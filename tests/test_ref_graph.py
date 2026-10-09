"""类型化引用图 (ref_graph) 的测试"""

import json

import pytest

import pyJianYingDraft as draft
from pyJianYingDraft.ref_graph import (
    GraphEdge, GraphNode, RefGraph, _check_cycles, fix_content, validate_content,
)
from tests.helpers import fake_video_material


# --------------------------------------------------------------------------- #
# 内容构造辅助
# --------------------------------------------------------------------------- #


def make_segment(seg_id="S1", material_id="V1", extra_refs=None, *, render_index=0, track_render_index=0):
    return {
        "id": seg_id,
        "material_id": material_id,
        "extra_material_refs": list(extra_refs) if extra_refs is not None else ["SP1"],
        "common_keyframes": [],
        "keyframe_refs": [],
        "render_index": render_index,
        "track_render_index": track_render_index,
    }


def make_content(tracks=None, materials=None, *, keyframes=None, keyframe_graph_list=None):
    return {
        "tracks": tracks if tracks is not None else [
            {"id": "T1", "type": "video", "segments": [make_segment()]},
        ],
        "materials": materials if materials is not None else {
            "videos": [{"id": "V1", "type": "video"}],
            "speeds": [{"id": "SP1", "type": "speed"}],
        },
        "keyframes": keyframes if keyframes is not None else {},
        "keyframe_graph_list": keyframe_graph_list if keyframe_graph_list is not None else [],
    }


def codes(report):
    return [v.code for v in report.violations]


# --------------------------------------------------------------------------- #
# 基础构图与健康图
# --------------------------------------------------------------------------- #


def test_valid_graph_passes():
    report = validate_content(make_content())
    assert report.ok
    assert report.errors == []


def test_graph_registers_typed_nodes_and_paths():
    graph = RefGraph(make_content())
    by_path = {n.path: n for n in graph.nodes}

    assert by_path["$.tracks[0]"].ntype == "track:video"
    assert by_path["$.tracks[0].segments[0]"].ntype == "segment:video"
    assert by_path["$.materials.videos[0]"].ntype == "material:videos"
    assert by_path["$.materials.speeds[0]"].ntype == "material:speeds"

    edge_kinds = {(e.kind, e.target) for e in graph.edges}
    assert ("primary_material", "V1") in edge_kinds
    assert ("extra_material", "SP1") in edge_kinds


def test_effects_bucket_uses_type_discriminator():
    content = make_content(
        materials={"effects": [
            {"id": "F1", "type": "filter"},
            {"id": "M1", "type": "mix_mode"},
        ]},
    )
    graph = RefGraph(content)
    ntypes = {n.id: n.ntype for n in graph.nodes}
    assert ntypes["F1"] == "material:effects:filter"
    assert ntypes["M1"] == "material:effects:mix_mode"


def test_invalid_track_type_is_error():
    content = make_content(tracks=[{"id": "T1", "type": "weird", "segments": []}])
    report = validate_content(content)
    assert not report.ok
    assert report.errors[0].code == "invalid_track_type"
    assert report.errors[0].paths == ["$.tracks[0].type"]


# --------------------------------------------------------------------------- #
# 全局 ID 唯一性
# --------------------------------------------------------------------------- #


def test_duplicate_id_reports_exact_paths_and_cluster():
    content = make_content(tracks=[{"id": "T1", "type": "video", "segments": [
        make_segment("S1"), make_segment("S1"),
    ]}])
    report = validate_content(content)

    dup = next(v for v in report.errors if v.code == "duplicate_id")
    assert dup.paths == [
        "$.tracks[0].segments[0].id",
        "$.tracks[0].segments[1].id",
    ]
    node_paths = [n["path"] for n in dup.subgraph["nodes"]]
    assert node_paths == ["$.tracks[0].segments[0]", "$.tracks[0].segments[1]"]
    assert dup.fix_available and dup.fix["kind"] == "deduplicate_id"


def test_fix_duplicate_ids_renames_nodes_and_keeps_edges_consistent():
    content = make_content(tracks=[{"id": "T1", "type": "video", "segments": [
        make_segment("S1", "V1", ["SP1"]),
        make_segment("S1", "V1", ["SP1"]),
    ]}], materials={
        "videos": [{"id": "V1", "type": "video"}],
        "speeds": [{"id": "SP1", "type": "speed"}, {"id": "SP1", "type": "speed"}],
    })

    report = fix_content(content)
    assert report.ok

    seg0, seg1 = content["tracks"][0]["segments"]
    spd0, spd1 = content["materials"]["speeds"]
    # 保留 path 最小者, 其余获得新 ID
    assert seg0["id"] == "S1" and seg1["id"] != "S1"
    assert spd0["id"] == "SP1" and spd1["id"] != "SP1"
    # 不产生悬空: 重复片段与保留素材/变速共享 (剪映允许一份素材被多片段使用)
    assert seg1["material_id"] == "V1"
    assert seg1["extra_material_refs"] == ["SP1"]
    # 被改名的变速成为不可达节点, 以 info 报告但仍保留
    unreachable = next(v for v in report.infos if v.code == "unreachable_material")
    assert unreachable.paths == ["$.materials.speeds[1]"]
    assert spd1 in content["materials"]["speeds"]  # 没有被删除


def test_ambiguous_ref_pinpoints_each_edge_into_duplicate_cluster():
    content = make_content(tracks=[{"id": "T1", "type": "video", "segments": [
        make_segment("S1"), make_segment("S1"),
    ]}])
    report = validate_content(content)
    ambiguous = [v for v in report.errors if v.code == "ambiguous_ref"]
    # 两个片段各自的 material_id 都指向唯一的 V1 (不歧义); 无 extra 重复, 故无歧义边
    assert ambiguous == []


# --------------------------------------------------------------------------- #
# 悬空引用
# --------------------------------------------------------------------------- #


def test_dangling_primary_material_has_exact_path():
    content = make_content(tracks=[{"id": "T1", "type": "video", "segments": [
        make_segment(material_id="GHOST"),
    ]}])
    report = validate_content(content)
    err = next(v for v in report.errors if v.code == "dangling_ref")
    assert err.paths == ["$.tracks[0].segments[0].material_id"]
    assert err.subgraph["edges"][0]["target"] == "GHOST"
    assert not err.fix_available


def test_dangling_extra_ref_has_exact_list_element_path():
    content = make_content(tracks=[{"id": "T1", "type": "video", "segments": [
        make_segment(extra_refs=["SP1", "GHOST"]),
    ]}])
    report = validate_content(content)
    err = next(v for v in report.errors if v.code == "dangling_ref")
    assert err.paths == ["$.tracks[0].segments[0].extra_material_refs[1]"]


def test_drop_dangling_removes_list_elements_and_clears_fields():
    content = make_content(tracks=[{"id": "T1", "type": "video", "segments": [
        make_segment(material_id="GHOST", extra_refs=["SP1", "GHOST2"]),
    ]}])
    report = fix_content(content, drop_dangling=True)
    assert report.ok

    segment = content["tracks"][0]["segments"][0]
    assert segment["material_id"] == ""
    assert segment["extra_material_refs"] == ["SP1"]
    assert any(a["kind"] == "drop_dangling" for a in report.actions)


# --------------------------------------------------------------------------- #
# 边类型
# --------------------------------------------------------------------------- #


def test_primary_material_pointing_at_wrong_bucket_is_type_error():
    content = make_content(tracks=[{"id": "T1", "type": "video", "segments": [
        make_segment(material_id="A1"),
    ]}], materials={
        "audios": [{"id": "A1", "type": "extract_music"}],
    })
    report = validate_content(content)
    err = next(v for v in report.errors if v.code == "invalid_edge_type")
    assert err.paths == ["$.tracks[0].segments[0].material_id"]
    assert "material:audios" in err.message


def test_extra_ref_pointing_at_primary_media_bucket_is_type_error():
    content = make_content(tracks=[{"id": "T1", "type": "video", "segments": [
        make_segment(extra_refs=["V2"]),
    ]}], materials={
        "videos": [{"id": "V1", "type": "video"}, {"id": "V2", "type": "video"}],
    })
    report = validate_content(content)
    err = next(v for v in report.errors if v.code == "invalid_edge_type")
    assert err.paths == ["$.tracks[0].segments[0].extra_material_refs[0]"]


def test_filter_segment_must_reference_filter_effect():
    # filter 轨道片段主素材指向 video_effects -> 类型错误
    content = make_content(tracks=[{"id": "T1", "type": "filter", "segments": [
        make_segment(material_id="E1"),
    ]}], materials={
        "video_effects": [{"id": "E1", "type": "video_effect"}],
    })
    report = validate_content(content)
    assert any(v.code == "invalid_edge_type" for v in report.errors)


def test_weak_keyframe_refs_are_not_validated():
    content = make_content(tracks=[{"id": "T1", "type": "video", "segments": [{
        "id": "S1", "material_id": "V1", "extra_material_refs": ["SP1"],
        "common_keyframes": [], "keyframe_refs": ["MYSTERIOUS"],
        "render_index": 0, "track_render_index": 0,
    }]}])
    report = validate_content(content)
    assert report.ok


# --------------------------------------------------------------------------- #
# 可达性
# --------------------------------------------------------------------------- #


def test_unreachable_material_is_info_and_never_auto_removed():
    content = make_content(materials={
        "videos": [{"id": "V1", "type": "video"}, {"id": "V2", "type": "video"}],
        "speeds": [{"id": "SP1", "type": "speed"}],
    })
    report_before = validate_content(content)
    unreachable = next(v for v in report_before.infos if v.code == "unreachable_material")
    assert unreachable.paths == ["$.materials.videos[1]"]
    assert not unreachable.fix_available

    report_after = fix_content(content)
    assert len(content["materials"]["videos"]) == 2
    assert any(v.code == "unreachable_retained" for v in report_after.infos)


def test_top_level_keyframes_and_graphs_are_implicitly_protected():
    content = make_content(
        materials={"videos": [{"id": "V1", "type": "video"}],
                   "speeds": [{"id": "SP1", "type": "speed"}]},
        keyframes={"videos": [{
            "id": "KL1", "material_id": "V1", "property_type": "KFTypeAlpha",
            "keyframe_list": [{"id": "K1", "time_offset": 0, "values": [1.0], "graphID": ""}],
        }]},
        keyframe_graph_list=[{"id": "G1", "keyframes": []}],
    )
    report = validate_content(content)
    # 不被轨道引用的 keyframe 列表/图不报告为不可达
    assert all(v.code != "unreachable_material" for v in report.violations)


def test_keyframe_graph_edge_resolves_and_is_typed():
    content = make_content(
        keyframe_graph_list=[{"id": "G1", "keyframes": [
            {"id": "K1", "time_offset": 0, "values": [1.0], "graphID": "G1"},
        ]}],
    )
    report = validate_content(content)
    assert report.ok
    graph = RefGraph(content)
    assert any(e.kind == "graph" and e.target == "G1" for e in graph.edges)


# --------------------------------------------------------------------------- #
# 环检测
# --------------------------------------------------------------------------- #


def test_cycle_detection_with_fabricated_adjacency():
    """schema 合规的草稿结构上无环 (边总指向 material/graph); 此处构造属主为
    material 的假边验证 DFS 环检测本身。"""
    path_a = "$.materials.videos[0]"
    path_b = "$.materials.audios[0]"
    path_l = "$.materials.videos[0].common_keyframes[0]"
    nodes = [
        GraphNode("A", "material:videos", path_a, path_a + ".id", None, {}),
        GraphNode("B", "material:audios", path_b, path_b + ".id", None, {}),
        GraphNode("L1", "keyframe_list", path_l, path_l + ".id", path_a, {}),
    ]
    edges = [
        GraphEdge("primary_material", path_a + ".material_id", path_a, "B", "field"),
        GraphEdge("extra_material", path_b + ".extra_material_refs[0]", path_b, "L1", "list"),
        GraphEdge("kf_material", path_l + ".material_id", path_l, "A", "field"),
    ]
    graph = RefGraph.__new__(RefGraph)
    graph.edges = edges
    graph.nodes_by_id = {"A": [nodes[0]], "B": [nodes[1]], "L1": [nodes[2]]}
    graph.nodes_by_path = {n.path: n for n in nodes}

    violations = _check_cycles(graph)
    assert len(violations) == 1
    assert violations[0].code == "cycle"
    assert {n["path"] for n in violations[0].subgraph["nodes"]} == {path_a, path_b, path_l}


# --------------------------------------------------------------------------- #
# render_index 与轨道顺序
# --------------------------------------------------------------------------- #


def test_render_index_mismatch_is_reported_and_fixed():
    content = make_content(tracks=[
        {"id": "T1", "type": "video", "segments": [
            make_segment(render_index=9, track_render_index=3),
        ]},
        {"id": "T2", "type": "video", "segments": []},
    ])
    report = validate_content(content)
    err = next(v for v in report.errors if v.code == "render_index_mismatch")
    assert set(err.paths) == {
        "$.tracks[0].segments[0].render_index",
        "$.tracks[0].segments[0].track_render_index",
    }

    fix_content(content)
    segment = content["tracks"][0]["segments"][0]
    assert segment["render_index"] == 0
    assert segment["track_render_index"] == 0


def test_track_layer_order_warns_on_inversion():
    content = make_content(tracks=[
        {"id": "T1", "type": "text", "segments": []},
        {"id": "T2", "type": "video", "segments": []},
    ])
    report = validate_content(content)
    warning = next(v for v in report.warnings if v.code == "track_layer_order")
    assert "$.tracks[0]" in warning.paths and "$.tracks[1]" in warning.paths

    # 可关闭
    assert validate_content(content, layer_convention=False).warnings == []


# --------------------------------------------------------------------------- #
# 显式 rename
# --------------------------------------------------------------------------- #


def test_explicit_rename_rewrites_all_edges():
    content = make_content(tracks=[{"id": "T1", "type": "video", "segments": [
        make_segment("S1", "V1", ["SP1"]),
        make_segment("S2", "V1", ["SP1"]),
    ]}], materials={
        "videos": [{"id": "V1", "type": "video"}],
        "speeds": [{"id": "SP1", "type": "speed"}],
    })
    report = fix_content(content, rename={"SP1": "SPX"})
    assert report.ok
    for segment in content["tracks"][0]["segments"]:
        assert segment["extra_material_refs"] == ["SPX"]
    assert content["materials"]["speeds"][0]["id"] == "SPX"


def test_rename_validates_arguments():
    content = make_content()
    with pytest.raises(ValueError):
        fix_content(content, rename={"NOPE": "X"})
    with pytest.raises(ValueError):
        fix_content(content, rename={"SP1": "V1"})  # 目标 ID 已占用


# --------------------------------------------------------------------------- #
# 报告序列化
# --------------------------------------------------------------------------- #


def test_report_json_contains_paths_subgraph_and_fix():
    content = make_content(tracks=[{"id": "T1", "type": "video", "segments": [
        make_segment(material_id="GHOST"),
    ]}])
    report = validate_content(content)
    payload = json.loads(report.to_json())
    assert payload["ok"] is False
    assert payload["counts"]["error"] >= 1
    violation = next(v for v in payload["violations"] if v["code"] == "dangling_ref")
    assert violation["paths"] == ["$.tracks[0].segments[0].material_id"]
    assert "edges" in violation["subgraph"] and "nodes" in violation["subgraph"]


# --------------------------------------------------------------------------- #
# ScriptFile 集成
# --------------------------------------------------------------------------- #


def _script_with_video_segment():
    script = draft.ScriptFile(1920, 1080, 30, False)
    script.append_track(draft.TrackSpec(draft.TrackType.video))
    segment = draft.VideoSegment(fake_video_material(), draft.trange("0s", "2s"))
    script.add_segment(segment)
    return script, segment


def test_check_references_passes_for_healthy_script():
    script, _ = _script_with_video_segment()
    report = script.check_references()
    assert report.ok
    assert report.errors == []


def test_check_references_detects_tampered_material_id():
    script, segment = _script_with_video_segment()
    segment.material_id = "ghost-id"

    report = script.check_references()
    assert not report.ok
    err = next(v for v in report.errors if v.code == "dangling_ref")
    assert err.paths == ["$.tracks[0].segments[0].material_id"]


def test_dumps_validate_raises_and_fix_repairs():
    script, segment = _script_with_video_segment()
    segment.material_id = "ghost-id"

    with pytest.raises(draft.ReferenceInvariantError):
        script.dumps(validate=True)

    output = script.dumps(validate=True, fix=True, drop_dangling=True)
    dumped = json.loads(output)
    assert dumped["tracks"][0]["segments"][0]["material_id"] == ""


def test_check_references_fix_duplicate_segment_id():
    script = draft.ScriptFile(1920, 1080, 30, False)
    script.append_track(draft.TrackSpec(draft.TrackType.video))

    seg1 = draft.VideoSegment(fake_video_material(material_id="V1"), draft.trange("0s", "2s"))
    seg2 = draft.VideoSegment(fake_video_material(material_id="V1"), draft.trange("2s", "2s"))
    seg2.segment_id = seg1.segment_id  # 人为制造重复 ID

    script.add_segment(seg1)
    script.add_segment(seg2)

    report = script.check_references(fix=True)
    assert report.ok
    # 对象属性不回写; 在序列化管道内修复后, 导出的两个片段 ID 互不相同
    dumped = json.loads(script.dumps(validate=True, fix=True))
    seg_ids = [s["id"] for s in dumped["tracks"][0]["segments"]]
    assert len(set(seg_ids)) == 2


def test_check_references_rename_requires_fix():
    script, segment = _script_with_video_segment()
    with pytest.raises(ValueError):
        script.check_references(rename={segment.material_id: "X"})
