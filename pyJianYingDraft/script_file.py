import json
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional

from . import assets
from . import exceptions
from . import util
from ._script_file_segments import _ScriptFileSegmentOps
from ._script_file_template import _ScriptFileTemplateOps
from ._script_file_tracks import _ScriptFileTrackOps
from .draft_content_loader import FallbackLoader, load_draft_content
from .ref_graph import RefReport, Violation, fix_content, validate_content
from .script_material import ScriptMaterial
from .template_mode import ImportedTrack, import_track
from .track import BaseTrack, Track


class ScriptFile(_ScriptFileTrackOps, _ScriptFileSegmentOps, _ScriptFileTemplateOps):
    """剪映草稿文件, 大部分接口定义在此"""

    save_path: Optional[str]
    """草稿文件保存路径, 仅在模板模式下有效"""
    content: Dict[str, Any]
    """草稿文件内容"""

    width: int
    """视频的宽度, 单位为像素"""
    height: int
    """视频的高度, 单位为像素"""
    fps: int
    """视频的帧率"""
    duration: int
    """视频的总时长, 单位为微秒"""

    maintrack_adsorb: bool
    """是否启用主轨道吸附（主轨磁吸）"""

    materials: ScriptMaterial
    """草稿文件中的素材信息部分"""
    tracks: Dict[str, Track]
    """轨道信息"""

    imported_materials: Dict[str, List[Dict[str, Any]]]
    """导入的素材原始信息, 读取时推荐走带自动补空的`_get_imported_material_list`方法"""
    imported_tracks: List[ImportedTrack]
    """导入的轨道信息"""
    _track_ref_owner_id: str
    """用于校验 TrackRef 归属的内部标识"""

    def __init__(self, width: int, height: int, fps: int, maintrack_adsorb: bool):
        """**创建剪映草稿推荐使用`DraftFolder.create_draft()`而非此方法**

        Args:
            width (int): 视频宽度, 单位为像素
            height (int): 视频高度, 单位为像素
            fps (int): 视频帧率
            maintrack_adsorb (bool): 是否启用主轨道吸附（主轨磁吸）

        Raises:
            无
        """
        self.save_path = None

        self.width = width
        self.height = height
        self.fps = fps
        self.duration = 0
        self.maintrack_adsorb = maintrack_adsorb

        self.materials = ScriptMaterial()
        self.tracks = {}

        self.imported_materials = {}
        self.imported_tracks = []
        self._track_ref_owner_id = uuid.uuid4().hex

        with open(assets.get_asset_path("DRAFT_CONTENT_TEMPLATE"), "r", encoding="utf-8") as f:
            self.content = json.load(f)

    @classmethod
    def _load_template(
        cls,
        json_path: str,
        fallback_loader: Optional[FallbackLoader] = None,
    ) -> "ScriptFile":
        obj = cls(**util.provide_ctor_defaults(cls))
        obj.save_path = json_path
        obj.content = load_draft_content(json_path, fallback_loader=fallback_loader)
        obj.content.setdefault("fps", 30.0)
        obj.content.setdefault("config", {})
        obj.content["config"].setdefault("maintrack_adsorb", True)
        obj.content.setdefault("tracks", [])
        obj.content.setdefault("materials", {})

        for track in obj.content["tracks"]:
            track.setdefault("segments", [])

        util.assign_attr_with_json(obj, ["fps", "duration"], obj.content)
        util.assign_attr_with_json(obj, ["maintrack_adsorb"], obj.content["config"])
        util.assign_attr_with_json(obj, ["width", "height"], obj.content["canvas_config"])

        obj.imported_materials = deepcopy(obj.content["materials"])
        obj.imported_tracks = [
            import_track(track_data, track_order)
            for track_order, track_data in enumerate(obj.content["tracks"])
        ]

        return obj

    def _build_export_content(self) -> Dict[str, Any]:
        """收集素材与轨道, 生成最终导出的草稿内容 dict (不进行 JSON 序列化)"""
        self.content["fps"] = self.fps
        self.content["duration"] = self.duration
        self.content["config"]["maintrack_adsorb"] = self.maintrack_adsorb
        self.content["canvas_config"] = {"width": self.width, "height": self.height, "ratio": "original"}
        self.content["materials"] = self.materials.export_json()

        for material_type, material_list in self.imported_materials.items():
            if material_type not in self.content["materials"]:
                self.content["materials"][material_type] = material_list
            else:
                self.content["materials"][material_type].extend(material_list)

        track_list: List[BaseTrack] = []
        track_list.extend(self.imported_tracks)
        track_list.extend(self.tracks.values())
        track_list.sort(key=lambda track: track.track_order)
        track_exports = [track.export_json() for track in track_list]
        for export_index, track_json in enumerate(track_exports):
            for segment_json in track_json["segments"]:
                segment_json["render_index"] = export_index
                segment_json["track_render_index"] = 0

        self.content["tracks"] = track_exports

        return self.content

    def check_references(
        self,
        *,
        fix: bool = False,
        drop_dangling: bool = False,
        rename: Optional[Dict[str, str]] = None,
        layer_convention: bool = True,
    ) -> RefReport:
        """序列化前建立类型化引用图并校验全部不变量

        校验内容: 全局 ID 唯一性、允许边类型、资源可达性、环、render_index、
        轨道顺序。报告中的每个违规带有精确 JSONPath、最小冲突子图及可选安全修复。

        Args:
            fix (`bool`): 是否应用安全修复 (重复 ID 去重并同步重写所有边、
                render_index 重写; 不删除任何不可达资源)
            drop_dangling (`bool`): 与`fix`配合, 移除悬空引用 (默认不移除)
            rename (`Dict[str, str]`, optional): 与`fix`配合, 显式 ID 改名,
                形如 {old_id: new_id}, 所有指向该 ID 的边会被同时重写
            layer_convention (`bool`): 是否检查轨道层级约定 (warning)

        Raises:
            `ValueError`: 提供了`rename`但未启用`fix`, 或 rename 参数非法

        Note:
            `fix=True` 的修复 (ID 去重/边重写) 作用于本次导出的内容 dict,
            不会回写到 segment/material 对象; 若随后重新调用无参数的`dumps()`,
            冲突会重新出现。需要"修复并落盘"时请使用
            `dump(path, validate=True, fix=True)` 或
            `dumps(validate=True, fix=True)`, 在同一序列化管道内闭环。
            track_order 并列修复属于对象状态修复, 会直接生效。

        Returns:
            `RefReport`: 校验报告
        """
        if rename is not None and not fix:
            raise ValueError("rename 仅能与 fix=True 一起使用")

        # 对象层轨道顺序不变量: 检查 track_order 是否有并列
        order_groups: Dict[int, List[BaseTrack]] = {}
        for track in self._list_all_tracks_in_order():
            order_groups.setdefault(track.track_order, []).append(track)
        tied_tracks: List[BaseTrack] = [
            track for group in order_groups.values() if len(group) > 1 for track in group
        ]
        order_fixed = False
        if tied_tracks and fix:
            self._reindex_track_orders(self._list_all_tracks_in_order())
            order_fixed = True

        content = self._build_export_content()

        if fix:
            report = fix_content(
                content, rename=rename, drop_dangling=drop_dangling,
                layer_convention=layer_convention,
            )
        else:
            report = validate_content(content, layer_convention=layer_convention)

        # 追加轨道顺序不变量结果 (路径来自已导出的 tracks 数组)
        if tied_tracks and not order_fixed:
            path_by_id = {
                track.get("id"): f"$.tracks[{i}]"
                for i, track in enumerate(content.get("tracks", []))
                if isinstance(track, dict)
            }
            report.append(Violation(
                code="track_order_conflict", severity="warning",
                message="存在 track_order 并列的轨道, 导出顺序依赖稳定排序, 可能与预期不一致",
                paths=[path_by_id[t.track_id] for t in tied_tracks if t.track_id in path_by_id],
                subgraph={
                    "nodes": [
                        {
                            "id": t.track_id,
                            "type": f"track:{t.track_type.name}",
                            "path": path_by_id.get(t.track_id, ""),
                        }
                        for t in tied_tracks
                    ],
                    "edges": [],
                },
                fix_available=True,
                fix={"kind": "reindex_track_order"},
            ))
        elif order_fixed:
            report.append(Violation(
                code="track_order_reindexed", severity="info",
                message="检测到 track_order 并列, 已按当前顺序重新编号为 0..n-1",
                paths=[],
                subgraph={"nodes": [], "edges": []},
            ))

        return report

    def dumps(
        self,
        *,
        validate: bool = False,
        fix: bool = False,
        drop_dangling: bool = False,
    ) -> str:
        """将草稿文件内容导出为JSON字符串

        Args:
            validate (`bool`): 序列化前建立引用图并校验, 存在 error 级冲突时抛出
                `ReferenceInvariantError`
            fix (`bool`): 与`validate`配合, 先应用安全修复再校验
            drop_dangling (`bool`): 与`fix`配合, 移除悬空引用
        """
        content = self._build_export_content()

        if validate:
            if fix:
                report = fix_content(content, drop_dangling=drop_dangling)
            else:
                report = validate_content(content)
            if not report.ok:
                raise exceptions.ReferenceInvariantError(
                    "草稿引用图不变量校验失败:\n" + report.to_json(),
                )

        return json.dumps(content, ensure_ascii=False, indent=4)

    def dump(
        self,
        file_path: str,
        *,
        validate: bool = False,
        fix: bool = False,
        drop_dangling: bool = False,
    ) -> None:
        """将草稿文件内容写入文件

        Args:
            validate (`bool`): 序列化前校验引用图, 失败时抛出`ReferenceInvariantError`
            fix (`bool`): 与`validate`配合, 先应用安全修复
            drop_dangling (`bool`): 与`fix`配合, 移除悬空引用
        """
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.dumps(validate=validate, fix=fix, drop_dangling=drop_dangling))

    def save(self) -> None:
        """保存草稿文件至打开时的路径

        Raises:
            `ValueError`: 没有设置保存路径
        """
        if self.save_path is None:
            raise ValueError("没有设置保存路径, 可能不在模板模式下")
        self.dump(self.save_path)
