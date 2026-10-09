"""草稿内容的类型化引用图

在序列化前对草稿内容（即最终导出的 JSON dict）建立由 track / segment / material /
speed / effect / animation / keyframe 组成的有向图, 并校验以下不变量:

- 全局 ID 唯一性 (global ID uniqueness)
- 允许的边类型 (allowed edge types)
- 资源可达性 (resource reachability)
- 环 (cycle)
- render_index 不变量 (render index invariants)
- 轨道顺序不变量 (track ordering invariants)

所有违规均返回精确的 JSONPath、最小冲突子图 (nodes + edges) 以及可选的安全修复方案。

注意: 对"看似不可达"的资源, 剪映可能隐式使用 (如顶层 keyframes/keyframe_graph_list
以及导入的素材桶), 本模块**永不自动删除**这类资源。
"""

from __future__ import annotations

import re
import uuid

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# --------------------------------------------------------------------------- #
# 常量定义
# --------------------------------------------------------------------------- #

KNOWN_TRACK_TYPES: Set[str] = {
    "video", "audio", "effect", "filter", "sticker", "text", "adjust",
}

"""segment 主素材边允许的目标节点类型 (按轨道/片段类型区分)"""
PRIMARY_MATERIAL_TARGETS: Dict[str, Set[str]] = {
    "video": {"material:videos"},
    "audio": {"material:audios"},
    "sticker": {"material:stickers"},
    "text": {"material:texts"},
    "effect": {"material:video_effects"},
    "filter": {"material:effects:filter"},
    # adjust 轨道由剪映内部生成, 不校验主素材边
}

"""extra_material_refs 边允许的目标节点类型 (含同桶任意 discriminator)"""
EXTRA_MATERIAL_TARGETS: Set[str] = {
    "material:speeds",
    "material:audio_effects",
    "material:audio_fades",
    "material:material_animations",
    "material:video_effects",
    "material:chromas",
    "material:masks",
    "material:transitions",
    "material:canvases",
    "material:effects",
}

"""剪映轨道层级约定 (自底向上), 仅用于 warning 级别的轨道顺序检查"""
TRACK_LAYER_RANK: Dict[str, int] = {
    "video": 0, "audio": 1, "text": 2, "sticker": 3, "effect": 4, "filter": 5,
}

_UUID_DASHED = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

# --------------------------------------------------------------------------- #
# JSONPath 工具
# --------------------------------------------------------------------------- #


def _p(path: str, key: str) -> str:
    return f"{path}.{key}"


def _pi(path: str, index: int) -> str:
    return f"{path}[{index}]"


def _is_within(path: str, ancestor: str) -> bool:
    """判断 path 是否位于 ancestor 节点子树内"""
    return path.startswith(ancestor + ".") or path.startswith(ancestor + "[")


def type_matches(actual: str, allowed: str) -> bool:
    """节点类型匹配; 不带 discriminator 的允许项可匹配同桶任意 discriminator

    例: "material:video_effects" 可匹配 "material:video_effects:face_effect"
    """
    if actual == allowed:
        return True
    return actual.startswith(allowed + ":")


def _new_id_like(sample: str) -> str:
    """生成与 sample 风格一致的新 ID (dashed UUID 或 hex)"""
    if _UUID_DASHED.match(sample):
        value = str(uuid.uuid4())
        if sample == sample.upper():
            return value.upper()
        return value
    return uuid.uuid4().hex


# --------------------------------------------------------------------------- #
# 图数据结构
# --------------------------------------------------------------------------- #


@dataclass
class GraphNode:
    """图节点: 任何拥有全局 ID 的 JSON 对象"""

    id: str
    ntype: str
    """节点类型, 如 track:video / segment:video / material:videos /
    material:effects:filter / keyframe_list / keyframe / keyframe_graph"""
    path: str
    """节点 dict 的 JSONPath"""
    id_path: str
    """节点 id 字段的 JSONPath"""
    container: Optional[str]
    """包含此节点的父节点 path (结构包含关系)"""
    obj: Dict[str, Any] = field(repr=False)
    """节点对应的原始 dict 对象"""


@dataclass
class GraphEdge:
    """图边: 一个 ID 引用 (字段或列表元素)"""

    kind: str
    """边类型: primary_material / extra_material / kf_material / graph / keyframe_ref"""
    path: str
    """引用所在的 JSONPath (列表元素精确到下标)"""
    source_path: str
    """边的属主节点 path"""
    target: str
    """被引用的 ID"""
    location: str
    """写入位置: "field" 表示 dict 字段, "list" 表示列表元素"""
    weak: bool = False
    """弱边: 意义不明确的引用 (keyframe_refs), 不做悬空/类型校验"""


@dataclass
class Violation:
    """单个不变量违规"""

    code: str
    """错误码, 如 duplicate_id / dangling_ref / invalid_edge_type / cycle /
    unreachable_material / render_index_mismatch / track_layer_order / ..."""
    severity: str
    """error / warning / info"""
    message: str
    paths: List[str]
    """相关的精确 JSONPath 列表"""
    subgraph: Dict[str, Any]
    """最小冲突子图: {"nodes": [...], "edges": [...]}"""
    fix_available: bool = False
    fix: Optional[Dict[str, Any]] = None
    """机器可读的安全修复方案; None 表示无安全修复"""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "paths": self.paths,
            "subgraph": self.subgraph,
            "fix_available": self.fix_available,
            "fix": self.fix,
        }


@dataclass
class RefReport:
    """引用图校验报告"""

    violations: List[Violation]
    graph: Optional["RefGraph"]
    actions: List[Dict[str, Any]] = field(default_factory=list)
    """fix_content 执行过的修复动作 (仅校验模式为空)"""

    @property
    def ok(self) -> bool:
        return not any(v.severity == "error" for v in self.violations)

    @property
    def errors(self) -> List[Violation]:
        return [v for v in self.violations if v.severity == "error"]

    @property
    def warnings(self) -> List[Violation]:
        return [v for v in self.violations if v.severity == "warning"]

    @property
    def infos(self) -> List[Violation]:
        return [v for v in self.violations if v.severity == "info"]

    def append(self, violation: Violation) -> None:
        self.violations.append(violation)

    def to_dict(self, *, include_graph: bool = False) -> Dict[str, Any]:
        ret: Dict[str, Any] = {
            "ok": self.ok,
            "counts": {
                "error": len(self.errors),
                "warning": len(self.warnings),
                "info": len(self.infos),
            },
            "violations": [v.to_dict() for v in self.violations],
        }
        if include_graph and self.graph is not None:
            ret["graph"] = self.graph.to_dict()
        if self.actions:
            ret["actions"] = self.actions
        return ret

    def to_json(self, *, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def format_text(self) -> str:
        """生成可读的文本报告"""
        lines = ["引用图校验结果: %s" % ("通过" if self.ok else "存在冲突")]
        for v in self.violations:
            lines.append(f"  [{v.severity}] {v.code}: {v.message}")
            for path in v.paths:
                lines.append(f"      @ {path}")
            if v.fix_available and v.fix is not None:
                lines.append(f"      fix: {v.fix.get('kind')}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 引用图构建
# --------------------------------------------------------------------------- #


class RefGraph:
    """草稿内容的类型化引用图"""

    content: Dict[str, Any]
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    nodes_by_path: Dict[str, GraphNode]
    nodes_by_id: Dict[str, List[GraphNode]]
    children: Dict[str, List[str]]
    """结构包含关系: 父节点 path -> 子节点 path 列表"""

    def __init__(self, content: Dict[str, Any]):
        self.content = content
        self.nodes = []
        self.edges = []
        self.nodes_by_path = {}
        self.nodes_by_id = {}
        self.children = {}
        self._build()

    # ----------------------- 内部注册 API ----------------------- #

    def _add_node(
        self,
        node_id: str,
        ntype: str,
        path: str,
        obj: Dict[str, Any],
        container: Optional[str],
    ) -> GraphNode:
        node = GraphNode(
            id=node_id, ntype=ntype, path=path,
            id_path=_p(path, "id"), container=container, obj=obj,
        )
        self.nodes.append(node)
        self.nodes_by_path[path] = node
        self.nodes_by_id.setdefault(node_id, []).append(node)
        if container is not None:
            self.children.setdefault(container, []).append(path)
        return node

    def _add_edge(self, edge: GraphEdge) -> None:
        self.edges.append(edge)

    def _ref_value(
        self,
        value: Any,
        kind: str,
        path: str,
        source: GraphNode,
        location: str,
        *,
        weak: bool = False,
    ) -> None:
        """登记一个引用值; 空字符串视为无引用, 非字符串登记为 malformed (调用方处理)"""
        if isinstance(value, str) and len(value) > 0:
            self._add_edge(GraphEdge(
                kind=kind, path=path, source_path=source.path,
                target=value, location=location, weak=weak,
            ))

    def _register_keyframe_lists(self, owner: GraphNode, lists: Any, path_prefix: str) -> None:
        """登记 (common_)keyframe 列表及其内部关键帧

        Args:
            owner: 包含 keyframe 列表的属主节点 (segment 或 material)
            lists: keyframe 列表数组
            path_prefix: 列表字段的 JSONPath, 如 $.tracks[0].segments[0].common_keyframes
        """
        if not isinstance(lists, list):
            return
        for k, kf_list in enumerate(lists):
            if not isinstance(kf_list, dict):
                continue
            list_path = _pi(path_prefix, k)
            list_id = kf_list.get("id", "")
            if isinstance(list_id, str) and len(list_id) > 0:
                list_node = self._add_node(list_id, "keyframe_list", list_path, kf_list, owner.path)
            else:
                # 无 id 的 keyframe 列表无法成为节点, 但其内部关键帧仍登记并挂在属主下
                list_node = owner

            # keyframe_list.material_id 边
            self._ref_value(
                kf_list.get("material_id"), "kf_material",
                _p(list_path, "material_id"), list_node, "field",
            )

            inner = kf_list.get("keyframe_list", [])
            if isinstance(inner, list):
                for m, keyframe in enumerate(inner):
                    self._register_keyframe(keyframe, _pi(_p(list_path, "keyframe_list"), m), list_node)

    def _register_keyframe(self, keyframe: Any, path: str, owner: GraphNode) -> None:
        """登记单个关键帧节点及其 graphID 边"""
        if not isinstance(keyframe, dict):
            return
        kf_id = keyframe.get("id", "")
        if not (isinstance(kf_id, str) and len(kf_id) > 0):
            return
        node = self._add_node(kf_id, "keyframe", path, keyframe, owner.path)
        self._ref_value(
            keyframe.get("graphID"), "graph",
            _p(path, "graphID"), node, "field",
        )

    # ----------------------- 构建流程 ----------------------- #

    def _build(self) -> None:
        self._build_tracks()
        self._build_materials()
        self._build_top_level_keyframes()
        self._build_keyframe_graphs()

    def _build_tracks(self) -> None:
        tracks = self.content.get("tracks", [])
        if not isinstance(tracks, list):
            return
        for i, track in enumerate(tracks):
            if not isinstance(track, dict):
                continue
            track_path = _pi("$.tracks", i)
            ttype = track.get("type", "unknown")
            ttype_name = ttype if isinstance(ttype, str) else "unknown"
            self._add_node(
                track.get("id", ""), f"track:{ttype_name}", track_path, track, None,
            )

            segments = track.get("segments", [])
            if not isinstance(segments, list):
                continue
            for j, segment in enumerate(segments):
                if not isinstance(segment, dict):
                    continue
                seg_path = _pi(_p(track_path, "segments"), j)
                seg_node = self._add_node(
                    segment.get("id", ""), f"segment:{ttype_name}", seg_path, segment, track_path,
                )
                self._ref_value(
                    segment.get("material_id"), "primary_material",
                    _p(seg_path, "material_id"), seg_node, "field",
                )
                extra = segment.get("extra_material_refs", [])
                if isinstance(extra, list):
                    for k, ref in enumerate(extra):
                        self._ref_value(
                            ref, "extra_material",
                            _pi(_p(seg_path, "extra_material_refs"), k),
                            seg_node, "list",
                        )
                # keyframe_refs 意义不明, 作为弱边登记
                kf_refs = segment.get("keyframe_refs", [])
                if isinstance(kf_refs, list):
                    for k, ref in enumerate(kf_refs):
                        self._ref_value(
                            ref, "keyframe_ref",
                            _pi(_p(seg_path, "keyframe_refs"), k),
                            seg_node, "list", weak=True,
                        )
                self._register_keyframe_lists(
                    seg_node, segment.get("common_keyframes"),
                    _p(seg_path, "common_keyframes"),
                )

    def _material_ntype(self, bucket: str, item: Dict[str, Any]) -> str:
        ntype = f"material:{bucket}"
        # effects / video_effects 桶是混合类型, 追加 type discriminator
        if bucket in ("effects", "video_effects"):
            disc = item.get("type")
            if isinstance(disc, str) and len(disc) > 0:
                ntype = f"{ntype}:{disc}"
        return ntype

    def _build_materials(self) -> None:
        materials = self.content.get("materials", {})
        if not isinstance(materials, dict):
            return
        for bucket, material_list in materials.items():
            if not isinstance(material_list, list):
                continue
            bucket_path = _p("$.materials", bucket)
            for k, item in enumerate(material_list):
                if not isinstance(item, dict):
                    continue
                item_path = _pi(bucket_path, k)
                node = self._add_node(
                    item.get("id", ""), self._material_ntype(bucket, item),
                    item_path, item, None,
                )
                # 素材内嵌的 keyframe 列表 (导入草稿中的 video_effect 等可能携带)
                self._register_keyframe_lists(
                    node, item.get("common_keyframes"),
                    _p(item_path, "common_keyframes"),
                )

    def _build_top_level_keyframes(self) -> None:
        """顶层 keyframes.<bucket>: 剪映的并行存储, 全部视为受保护的隐式根"""
        groups = self.content.get("keyframes", {})
        if not isinstance(groups, dict):
            return
        for bucket, kf_lists in groups.items():
            if not isinstance(kf_lists, list):
                continue
            bucket_path = _p("$.keyframes", bucket)
            for k, kf_list in enumerate(kf_lists):
                if not isinstance(kf_list, dict):
                    continue
                list_path = _pi(bucket_path, k)
                list_id = kf_list.get("id", "")
                if isinstance(list_id, str) and len(list_id) > 0:
                    list_node = self._add_node(list_id, "keyframe_list", list_path, kf_list, None)
                else:
                    continue
                self._ref_value(
                    kf_list.get("material_id"), "kf_material",
                    _p(list_path, "material_id"), list_node, "field",
                )
                inner = kf_list.get("keyframe_list", [])
                if isinstance(inner, list):
                    for m, keyframe in enumerate(inner):
                        self._register_keyframe(keyframe, _pi(_p(list_path, "keyframe_list"), m), list_node)

    def _build_keyframe_graphs(self) -> None:
        graphs = self.content.get("keyframe_graph_list", [])
        if not isinstance(graphs, list):
            return
        for i, graph in enumerate(graphs):
            if not isinstance(graph, dict):
                continue
            graph_path = _pi("$.keyframe_graph_list", i)
            graph_node = self._add_node(
                graph.get("id", ""), "keyframe_graph", graph_path, graph, None,
            )
            inner_keyframes = graph.get("keyframes", [])
            if isinstance(inner_keyframes, list):
                for j, keyframe in enumerate(inner_keyframes):
                    self._register_keyframe(
                        keyframe, _pi(_p(graph_path, "keyframes"), j), graph_node,
                    )

    # ----------------------- 输出 ----------------------- #

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {"id": n.id, "type": n.ntype, "path": n.path}
                for n in self.nodes
            ],
            "edges": [_edge_dict(e) for e in self.edges],
        }


def _edge_dict(edge: GraphEdge) -> Dict[str, Any]:
    return {
        "kind": edge.kind,
        "path": edge.path,
        "source_path": edge.source_path,
        "target": edge.target,
        "weak": edge.weak,
    }


def _node_dict(node: GraphNode) -> Dict[str, Any]:
    return {"id": node.id, "type": node.ntype, "path": node.path}


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #


def validate_content(content: Dict[str, Any], *, layer_convention: bool = True) -> RefReport:
    """对草稿内容执行全部不变量校验, 返回报告"""
    graph = RefGraph(content)
    violations: List[Violation] = []

    violations.extend(_check_track_types(graph))
    violations.extend(_check_duplicate_ids(graph))
    violations.extend(_check_edges(graph))
    violations.extend(_check_cycles(graph))
    violations.extend(_check_reachability(graph))
    violations.extend(_check_render_index(graph))
    if layer_convention:
        violations.extend(_check_track_layer_order(graph))

    return RefReport(violations=violations, graph=graph)


def _check_track_types(graph: RefGraph) -> List[Violation]:
    result: List[Violation] = []
    tracks = graph.content.get("tracks", [])
    if not isinstance(tracks, list):
        return result
    for i, track in enumerate(tracks):
        if not isinstance(track, dict):
            continue
        ttype = track.get("type")
        if not isinstance(ttype, str) or ttype not in KNOWN_TRACK_TYPES:
            path = _pi("$.tracks", i)
            node = graph.nodes_by_path.get(path)
            result.append(Violation(
                code="invalid_track_type", severity="error",
                message=f"未知的轨道类型 '{ttype}', 剪映无法识别",
                paths=[_p(path, "type")],
                subgraph={"nodes": [_node_dict(node)] if node else [], "edges": []},
            ))
    return result


def _check_duplicate_ids(graph: RefGraph) -> List[Violation]:
    result: List[Violation] = []
    for node_id, cluster in graph.nodes_by_id.items():
        if len(cluster) <= 1:
            continue
        cluster_sorted = sorted(cluster, key=lambda n: n.path)
        inbound = [_edge_dict(e) for e in graph.edges if e.target == node_id]
        result.append(Violation(
            code="duplicate_id", severity="error",
            message=f"全局 ID '{node_id}' 被 {len(cluster_sorted)} 个节点重复使用",
            paths=[n.id_path for n in cluster_sorted],
            subgraph={
                "nodes": [_node_dict(n) for n in cluster_sorted],
                "edges": inbound,
            },
            fix_available=True,
            fix={
                "kind": "deduplicate_id",
                "keep_path": cluster_sorted[0].path,
                "rename_paths": [n.path for n in cluster_sorted[1:]],
            },
        ))
    return result


def _allowed_targets(edge: GraphEdge, segment_type: Optional[str]) -> Optional[Set[str]]:
    """根据边类型与属主片段类型返回允许的目标类型; None 表示不校验"""
    if edge.kind == "primary_material":
        if segment_type is None:
            return None
        return PRIMARY_MATERIAL_TARGETS.get(segment_type)
    if edge.kind == "extra_material":
        return EXTRA_MATERIAL_TARGETS
    if edge.kind == "kf_material":
        return {"material"}
    if edge.kind == "graph":
        return {"keyframe_graph"}
    return None  # keyframe_ref 等弱边


def _owner_segment_type(graph: RefGraph, edge: GraphEdge) -> Optional[str]:
    """找到边属主链上最近的 segment 节点类型 (video/audio/...)"""
    path = edge.source_path
    while path:
        node = graph.nodes_by_path.get(path)
        if node is not None and node.ntype.startswith("segment:"):
            return node.ntype.split(":", 1)[1]
        if node is not None and node.container:
            path = node.container
        else:
            break
    return None


def _check_edges(graph: RefGraph) -> List[Violation]:
    result: List[Violation] = []
    for edge in graph.edges:
        source_node = graph.nodes_by_path.get(edge.source_path)

        # 弱边: 完全不校验
        if edge.weak:
            continue

        cluster = graph.nodes_by_id.get(edge.target, [])

        if len(cluster) == 0:
            result.append(Violation(
                code="dangling_ref", severity="error",
                message=f"{edge.kind} 引用了不存在的 ID '{edge.target}'",
                paths=[edge.path],
                subgraph={
                    "nodes": [_node_dict(source_node)] if source_node else [],
                    "edges": [_edge_dict(edge)],
                },
            ))
            continue

        if len(cluster) > 1:
            result.append(Violation(
                code="ambiguous_ref", severity="error",
                message=f"{edge.kind} 引用的 ID '{edge.target}' 存在 {len(cluster)} 个同名节点, 归属不明",
                paths=[edge.path],
                subgraph={
                    "nodes": [_node_dict(n) for n in sorted(cluster, key=lambda x: x.path)],
                    "edges": [_edge_dict(edge)],
                },
            ))

        # 类型校验
        allowed = _allowed_targets(edge, _owner_segment_type(graph, edge))
        if allowed is None:
            continue
        targets_ok = any(
            any(type_matches(n.ntype, t) for t in allowed) for n in cluster
        )
        if not targets_ok:
            actual_types = sorted({n.ntype for n in cluster})
            related_nodes: List[Dict[str, Any]] = []
            if source_node is not None:
                related_nodes.append(_node_dict(source_node))
            related_nodes.extend(_node_dict(n) for n in cluster)
            result.append(Violation(
                code="invalid_edge_type", severity="error",
                message=f"{edge.kind} 边不允许指向类型 {actual_types} (目标 ID '{edge.target}')",
                paths=[edge.path],
                subgraph={
                    "nodes": related_nodes,
                    "edges": [_edge_dict(edge)],
                },
            ))
    return result


def _check_cycles(graph: RefGraph) -> List[Violation]:
    """在已解析的强边图上做 DFS 环检测

    注: schema 合规的草稿中, 边总是从 segment/keyframe_list/keyframe 指向
    material/keyframe_graph, 结构上无环; 本检查作为防御性措施保留。
    """
    # 邻接表: 源节点 path -> [(edge, 目标节点 path)]
    adjacency: Dict[str, List[Tuple[GraphEdge, str]]] = {}
    for edge in graph.edges:
        if edge.weak:
            continue
        targets = graph.nodes_by_id.get(edge.target, [])
        for target in targets:
            adjacency.setdefault(edge.source_path, []).append((edge, target.path))

    color: Dict[str, int] = {}  # 0=未访问 1=在栈中 2=完成
    stack_nodes: List[str] = []
    # edge_stack[i] = 从 stack_nodes[i] 到 stack_nodes[i+1] 使用的边; 末尾恒为 None 占位
    stack_edges: List[Optional[GraphEdge]] = []
    found: Optional[Tuple[List[str], List[GraphEdge]]] = None

    def dfs(path: str) -> None:
        nonlocal found
        color[path] = 1
        stack_nodes.append(path)
        stack_edges.append(None)
        for edge, next_path in adjacency.get(path, []):
            if found is not None:
                return
            state = color.get(next_path, 0)
            if state == 0:
                stack_edges[-1] = edge
                dfs(next_path)
            elif state == 1:
                # 回边 edge: path -> next_path, 提取环
                start = stack_nodes.index(next_path)
                cyc_nodes = stack_nodes[start:] + [next_path]
                cyc_edges = [e for e in stack_edges[start:-1] if e is not None] + [edge]
                found = (cyc_nodes, cyc_edges)
                return
        stack_nodes.pop()
        stack_edges.pop()
        color[path] = 2

    for path in graph.nodes_by_path:
        if color.get(path, 0) == 0:
            dfs(path)
            if found is not None:
                break

    if found is None:
        return []

    cyc_nodes, cyc_edges = found
    nodes = [graph.nodes_by_path[p] for p in cyc_nodes[:-1]]
    return [Violation(
        code="cycle", severity="error",
        message="引用图中存在环: %s" % " -> ".join(cyc_nodes),
        paths=[e.path for e in cyc_edges],
        subgraph={
            "nodes": [_node_dict(n) for n in nodes],
            "edges": [_edge_dict(e) for e in cyc_edges],
        },
    )]


def _check_reachability(graph: RefGraph) -> List[Violation]:
    """资源可达性检查

    根: tracks; 隐式受保护根: keyframe_graph_list 与顶层 keyframes.* 中的节点。
    不可达的素材节点以 info 级别报告, 但永不提供删除修复 (剪映可能隐式使用)。
    """
    # 解析后的边邻接 (含弱边; 含歧义边的全部目标)
    edge_adj: Dict[str, List[str]] = {}
    for edge in graph.edges:
        for target in graph.nodes_by_id.get(edge.target, []):
            edge_adj.setdefault(edge.source_path, []).append(target.path)

    def traverse(roots: List[str]) -> Set[str]:
        reached: Set[str] = set()
        worklist = list(roots)
        while worklist:
            path = worklist.pop()
            if path in reached:
                continue
            reached.add(path)
            worklist.extend(graph.children.get(path, []))       # 结构包含
            worklist.extend(edge_adj.get(path, []))             # ID 边
        return reached

    track_roots = [
        n.path for n in graph.nodes if n.ntype.startswith("track:")
    ]
    reachable = traverse(track_roots)

    implicit_roots = [
        n.path for n in graph.nodes
        if n.ntype in ("keyframe_graph",)
        or (n.ntype == "keyframe_list" and n.path.startswith("$.keyframes."))
    ]
    implicit = traverse(implicit_roots) - reachable

    result: List[Violation] = []
    reported = 0
    for node in graph.nodes:
        if node.path in reachable or node.path in implicit:
            continue
        # 折叠不可达子树: 仅报告"不可达岛"的根节点 (container 不要求可达)
        if node.container is not None and node.container not in reachable:
            continue
        if node.ntype.startswith("material:"):
            result.append(Violation(
                code="unreachable_material", severity="info",
                message=f"素材节点 '{node.id}' ({node.ntype}) 不被任何轨道上的片段引用",
                paths=[node.path],
                subgraph={"nodes": [_node_dict(node)], "edges": []},
                fix_available=False,
                fix=None,
            ))
            reported += 1
        elif node.ntype == "keyframe_list" and not node.path.startswith("$.keyframes."):
            # 内嵌 keyframe 列表: 其属主素材不可达时会被折叠, 不会走到这里
            pass
    # 说明性警告: 存在不可达素材时提示"已保留, 未删除"
    if reported > 0:
        result.append(Violation(
            code="unreachable_retained", severity="info",
            message=f"共 {reported} 个不可达素材; 剪映可能隐式使用, 已全部保留不做删除",
            paths=[],
            subgraph={"nodes": [], "edges": []},
        ))
    return result


def _check_render_index(graph: RefGraph) -> List[Violation]:
    """render_index 不变量: segment.render_index == 所在轨道下标, track_render_index == 0"""
    result: List[Violation] = []
    tracks = graph.content.get("tracks", [])
    if not isinstance(tracks, list):
        return result
    for track_index, track in enumerate(tracks):
        if not isinstance(track, dict):
            continue
        track_path = _pi("$.tracks", track_index)
        segments = track.get("segments", [])
        if not isinstance(segments, list):
            continue
        for seg_index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                continue
            seg_path = _pi(_p(track_path, "segments"), seg_index)
            node = graph.nodes_by_path.get(seg_path)
            actual_ri = segment.get("render_index")
            actual_tri = segment.get("track_render_index")
            if actual_ri == track_index and actual_tri == 0:
                continue
            result.append(Violation(
                code="render_index_mismatch", severity="error",
                message=(
                    f"片段 render_index={actual_ri!r} (应为 {track_index}), "
                    f"track_render_index={actual_tri!r} (应为 0)"
                ),
                paths=[
                    _p(seg_path, "render_index"),
                    _p(seg_path, "track_render_index"),
                ],
                subgraph={"nodes": [_node_dict(node)] if node else [], "edges": []},
                fix_available=True,
                fix={
                    "kind": "rewrite_render_index",
                    "segment_path": seg_path,
                    "render_index": track_index,
                    "track_render_index": 0,
                },
            ))
    return result


def _check_track_layer_order(graph: RefGraph) -> List[Violation]:
    """轨道层级约定检查 (warning): 自底向上应为 video < audio < text < sticker < effect < filter

    仅报告相邻逆序对, 以精确定位顺序缺陷。
    """
    tracks = graph.content.get("tracks", [])
    if not isinstance(tracks, list) or len(tracks) < 2:
        return []

    inversions: List[Tuple[int, int]] = []
    for i in range(len(tracks) - 1):
        t1, t2 = tracks[i], tracks[i + 1]
        if not (isinstance(t1, dict) and isinstance(t2, dict)):
            continue
        r1 = TRACK_LAYER_RANK.get(t1.get("type", ""))  # type: ignore
        r2 = TRACK_LAYER_RANK.get(t2.get("type", ""))  # type: ignore
        if r1 is not None and r2 is not None and r1 > r2:
            inversions.append((i, i + 1))

    if not inversions:
        return []

    paths: List[str] = []
    nodes: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for i, j in inversions:
        for index in (i, j):
            path = _pi("$.tracks", index)
            node = graph.nodes_by_path.get(path)
            if node is not None and path not in seen:
                seen.add(path)
                paths.append(path)
                nodes.append(_node_dict(node))
    return [Violation(
        code="track_layer_order", severity="warning",
        message=f"存在 {len(inversions)} 处轨道层级逆序 (约定自底向上: video<audio<text<sticker<effect<filter), "
                "剪映可能静默重排或忽略",
        paths=paths,
        subgraph={"nodes": nodes, "edges": []},
    )]


# --------------------------------------------------------------------------- #
# 安全修复
# --------------------------------------------------------------------------- #


def fix_content(
    content: Dict[str, Any],
    *,
    rename: Optional[Dict[str, str]] = None,
    drop_dangling: bool = False,
    layer_convention: bool = True,
) -> RefReport:
    """在草稿内容上应用安全修复, 并返回修复后重新校验的报告

    安全修复内容:
    - duplicate_id: 保留每个 ID 簇中 path 最小的节点, 其余节点分配新 ID;
      同时在**一次遍历中重写所有边**: 位于被改名节点子树内、指向其旧 ID 的边
      跟随到新 ID; 其余边仍指向保留节点 (与剪映"复制片段共享素材"语义一致)。
    - 显式 rename (old_id -> new_id): 全局重写所有指向该 ID 的边及节点 ID。
    - render_index: 按轨道下标确定性重写。
    - drop_dangling (需显式开启): 移除悬空引用; 弱边不移除。

    永不执行: 删除不可达素材、打断环、自动修复类型错误或轨道层级逆序。
    """
    graph = RefGraph(content)
    actions: List[Dict[str, Any]] = []

    # 1. 校验显式 rename 参数
    global_renames: Dict[str, str] = {}
    if rename:
        existing_ids = set(graph.nodes_by_id.keys())
        for old_id, new_id in rename.items():
            if old_id not in existing_ids:
                raise ValueError(f"rename 失败: 草稿中不存在 ID '{old_id}'")
            if len(graph.nodes_by_id[old_id]) > 1:
                raise ValueError(
                    f"rename 失败: ID '{old_id}' 存在重复节点, 请先修复 duplicate_id"
                )
            if not isinstance(new_id, str) or len(new_id) == 0:
                raise ValueError(f"rename 失败: '{new_id}' 不是合法 ID")
            if new_id in existing_ids and new_id not in rename:
                raise ValueError(f"rename 失败: 新 ID '{new_id}' 已被占用")
            global_renames[old_id] = new_id

    # 2. 为重复 ID 建立 path -> new_id 的改名映射
    path_renames: Dict[str, Tuple[str, str]] = {}  # node path -> (old_id, new_id)
    for node_id, cluster in graph.nodes_by_id.items():
        if len(cluster) <= 1:
            continue
        cluster_sorted = sorted(cluster, key=lambda n: n.path)
        for node in cluster_sorted[1:]:
            path_renames[node.path] = (node_id, _new_id_like(node_id))

    renamed_nodes = [
        (graph.nodes_by_path[p], old_id, new_id)
        for p, (old_id, new_id) in path_renames.items()
    ]

    # 3. 一次遍历, 同步重写所有边
    for edge in graph.edges:
        new_target: Optional[str] = None

        if edge.target in global_renames:
            new_target = global_renames[edge.target]
        else:
            # 指向重复 ID 簇: 仅当边位于某个被改名节点子树内时跟随
            candidates = [
                (node, new_id) for node, old_id, new_id in renamed_nodes
                if old_id == edge.target and _is_within(edge.path, node.path)
            ]
            if candidates:
                # 多个祖先候选时取最深层节点
                candidates.sort(key=lambda item: len(item[0].path), reverse=True)
                new_target = candidates[0][1]

        if new_target is not None and new_target != edge.target:
            _write_edge_target(content, edge, new_target)
            actions.append({
                "kind": "rewrite_edge", "path": edge.path,
                "old": edge.target, "new": new_target,
            })

    # 4. 重写节点自身的 ID 字段 (重复 ID 修复 + 显式 rename)
    for node, old_id, new_id in renamed_nodes:
        node.obj["id"] = new_id
        actions.append({
            "kind": "rename_node", "path": node.path,
            "old": old_id, "new": new_id,
        })
    if global_renames:
        for old_id, new_id in global_renames.items():
            node = graph.nodes_by_id[old_id][0]
            node.obj["id"] = new_id
            actions.append({
                "kind": "rename_node", "path": node.path,
                "old": old_id, "new": new_id,
            })

    # 5. render_index 确定性重写
    tracks = content.get("tracks", [])
    if isinstance(tracks, list):
        for track_index, track in enumerate(tracks):
            if not isinstance(track, dict):
                continue
            for segment in track.get("segments", []):
                if not isinstance(segment, dict):
                    continue
                if segment.get("render_index") != track_index:
                    segment["render_index"] = track_index
                    actions.append({"kind": "rewrite_render_index", "value": track_index})
                if segment.get("track_render_index") != 0:
                    segment["track_render_index"] = 0
                    actions.append({"kind": "rewrite_track_render_index", "value": 0})

    # 6. 可选: 移除悬空引用 (弱边不移除)
    if drop_dangling:
        # 重新建图以反映改名后的解析关系
        current = RefGraph(content)
        list_removals: Dict[str, List[int]] = {}
        field_clears: List[Tuple[str, str]] = []
        for edge in current.edges:
            if edge.weak:
                continue
            if edge.target in current.nodes_by_id:
                continue
            if edge.location == "list":
                # 从 path 解析列表位置
                list_path, index = _split_list_path(edge.path)
                list_removals.setdefault(list_path, []).append(index)
            else:
                field_clears.append((edge.source_path, _field_name(edge.path)))
        for list_path, indices in list_removals.items():
            target_list = _resolve_path(content, list_path)
            if not isinstance(target_list, list):
                continue
            for index in sorted(indices, reverse=True):
                target_list.pop(index)
            actions.append({"kind": "drop_dangling", "path": list_path, "count": len(indices)})
        for source_path, field_name in field_clears:
            node = current.nodes_by_path.get(source_path)
            if node is not None:
                node.obj[field_name] = ""
                actions.append({"kind": "drop_dangling", "path": _p(source_path, field_name), "count": 1})

    # 7. 重新校验
    report = validate_content(content, layer_convention=layer_convention)
    report.actions = actions
    return report


def _write_edge_target(content: Dict[str, Any], edge: GraphEdge, new_target: str) -> None:
    """将边的目标 ID 写回草稿内容"""
    if edge.location == "field":
        node = _resolve_path(content, edge.source_path)
        if isinstance(node, dict):
            node[_field_name(edge.path)] = new_target
    else:
        target_list = _resolve_path(content, _split_list_path(edge.path)[0])
        index = _split_list_path(edge.path)[1]
        if isinstance(target_list, list) and 0 <= index < len(target_list):
            target_list[index] = new_target


_LIST_INDEX = re.compile(r"^(.*)\[(\d+)\]$")


def _split_list_path(path: str) -> Tuple[str, int]:
    match = _LIST_INDEX.match(path)
    assert match is not None, f"非法列表元素 path: {path}"
    return match.group(1), int(match.group(2))


def _field_name(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def _resolve_path(content: Dict[str, Any], path: str) -> Any:
    """按 JSONPath 取对象 (仅支持本模块生成的 path 形态)"""
    if not path.startswith("$."):
        return None
    current: Any = content
    for part in path[2:].split("."):
        # part 可能形如 "segments[0]" 或 "tracks[0]"
        while True:
            match = re.match(r"^([^\[\]]+)(\[(\d+)\])?$", part)
            if match is None:
                return None
            key = match.group(1)
            if isinstance(current, dict):
                current = current.get(key)
            else:
                return None
            if match.group(2) is None:
                break
            index = int(match.group(3))
            if isinstance(current, list) and 0 <= index < len(current):
                current = current[index]
            else:
                return None
            break
    return current


# --------------------------------------------------------------------------- #
# 便捷入口
# --------------------------------------------------------------------------- #


def build_ref_graph(content: Dict[str, Any]) -> RefGraph:
    """建立草稿内容的类型化引用图"""
    return RefGraph(content)
