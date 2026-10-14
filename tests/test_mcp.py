import asyncio
import base64
import json
import os
import sys
import urllib.request

import pytest

from smlibpy.mcp import discovery
from smlibpy.mcp.session import Session


@pytest.fixture(scope="module")
def session():
    s = Session()
    yield s
    s.close()


def test_session_keeps_state_and_survives_errors(session):
    session.clear()
    ok = session.run("plate = Box(40, 30, 10) - Cylinder(5, 20)\nprint(round(plate.volume, 3))")
    assert ok["ok"] and ok["output"].strip() == str(round(12000 - 3.141592653589793 * 25 * 10, 3))
    bad = session.run("plate = plate.fillet(plate.edges(), 50)")
    assert not bad["ok"] and "ModelingError" in bad["error"] and "line 1" in bad["error"]
    again = session.run("plate.volume")
    assert again["ok"] and float(again["value"]) == pytest.approx(12000 - 3.141592653589793 * 250, rel=1e-6)
    assert [item["name"] for item in again["scene"]] == ["plate"]


def test_session_stops_long_scripts_and_rebuilds(session):
    session.clear()
    assert session.run("a = Box(10, 10, 10)")["ok"]
    stuck = session.run("while True:\n    pass", timeout=2)
    assert not stuck["ok"] and "1 earlier scripts ran again" in stuck["error"]
    assert session.run("a.volume")["value"].startswith("1000")


def test_checkpoints(session):
    session.clear()
    session.run("p = Box(10, 10, 10)")
    session.checkpoint("cube")
    session.run("p = p - Cylinder(2, 20)")
    assert float(session.run("p.volume")["value"]) < 1000
    note = session.restore("cube")
    assert "cube" in note
    assert float(session.run("p.volume")["value"]) == pytest.approx(1000, rel=1e-9)


def test_session_save_and_load(session, tmp_path):
    session.clear()
    session.run("a = Box(20, 15, 10)", label="base plate")
    session.run("b = a - Cylinder(3, 20)", label="hole")
    session.checkpoint("with hole")
    session.run("c = b.fillet(b.edges().parallel_to(Z), 2)", label="fillet")
    path = session.save_to_file(str(tmp_path / "session.json"))
    assert os.path.getsize(path) > 0
    n_scripts = len(session.history)
    n_checkpoints = len(session.checkpoints)
    # A new session restores the saved one
    session.clear()
    assert session.run("x = Sphere(1)")["ok"]
    note = session.load_from_file(path)
    assert "3 scripts" in note and "1 checkpoints" in note
    assert len(session.history) == n_scripts
    assert len(session.checkpoints) == n_checkpoints
    assert float(session.run("c.volume")["value"]) < float(session.run("a.volume")["value"])
    assert abs(float(session.run("a.volume")["value"]) - 3000) < 1
    # Checkpoints still work after a restore
    session.restore("with hole")
    assert float(session.run("b.volume")["value"]) < 3000


def test_session_load_bad_files(session, tmp_path):
    session.clear()
    session.run("q = Box(5, 5, 5)")
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError):
        session.load_from_file(str(missing))
    not_session = tmp_path / "wrong.json"
    not_session.write_text('{"format": "other"}')
    with pytest.raises(ValueError, match="not a smlibpy-session file"):
        session.load_from_file(str(not_session))
    bad_json = tmp_path / "broken.json"
    bad_json.write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        session.load_from_file(str(bad_json))
    # The session is untouched by failed loads
    assert float(session.run("q.volume")["value"]) == pytest.approx(125, rel=1e-9)


def test_show_snapshot_save_inspect(session, tmp_path):
    session.clear()
    session.run("a = Box(10, 10, 10)\nb = Sphere(4).translate(20, 0, 0)\nshow(b)")
    scene = session.call("scene")
    assert [s["name"] for s in scene["scene"]] == ["b"] and scene["explicit"]
    shot = session.call("snapshot", view="sheet", size=(320, 240))
    assert shot["ok"] and shot["png"][:8] == b"\x89PNG\r\n\x1a\n"
    saved = session.call("save", format="step", path=str(tmp_path / "b.step"))
    assert saved["ok"] and os.path.getsize(saved["path"]) > 0
    info = session.call("inspect", name="a", faces=True)
    assert info["ok"] and len(info["info"]["faces"]) == 6 and info["info"]["valid"]


def test_snapshot_options_frame_and_compare(session, tmp_path):
    session.clear()
    session.run("part = Box(40, 30, 10) - Cylinder(5, 20)")
    shot = session.call("snapshot", size=(300, 200), path=str(tmp_path / "a.png"), highlight=["part.faces().top()"],
                        section="Plane.XZ", markers=[{"position": [0, 0, 5], "label": "bore"}], lines=True)
    assert shot["ok"] and shot["frame"]["scale"] > 0
    drawing = session.call("snapshot", size=(400, 300), drawing="third-angle", section={"origin": [0, 0, 0],
                                                                                         "normal": [0, -1, 0]})
    assert drawing["ok"] and drawing["frame"] is None
    opened = session.call("snapshot", size=(400, 300), drawing=True, section="Plane.XZ", cap=False)
    capped = session.call("snapshot", size=(400, 300), drawing=True, section="Plane.XZ")
    assert opened["ok"] and opened["png"] != capped["png"]
    bad = session.call("snapshot", highlight="nothing_here")
    assert not bad["ok"] and "highlight 'nothing_here'" in bad["error"]
    before = session.call("snapshot", size=(300, 200), path=str(tmp_path / "before.png"))
    session.run("part = part - Cylinder(3, 20).translate(12, 0, 0)")
    after = session.call("snapshot", size=(300, 200), path=str(tmp_path / "after.png"), frame=before["frame"],
                         compare=str(tmp_path / "before.png"))
    assert after["ok"] and after["frame"] == before["frame"]
    diff = after["diff"]
    assert diff["changed"] > 50 and diff["fraction"] < 0.05 and os.path.exists(diff["path"])
    same = session.call("snapshot", size=(300, 200), frame=before["frame"], compare=str(tmp_path / "after.png"))
    assert same["diff"]["changed"] == 0


def test_snapshot_zebra_and_curvature(session):
    session.clear()
    session.run("box = Box(40, 40, 20)\nsoft = box.fillet(box.edges().parallel_to(Z), 8, continuity='G2')\n"
                "show(soft)")
    zebra = session.call("snapshot", size=(300, 200), zebra={"stripes": 20, "direction": [0, 0, 1]},
                         highlight="soft.faces().top()", markers=[{"position": [0, 0, 10], "label": "top"}])
    curv = session.call("snapshot", size=(300, 200), curvature={"kind": "max", "range": 0.2})
    sheet = session.call("snapshot", size=(400, 300), sheet=True, curvature="mean")
    assert zebra["ok"] and curv["ok"] and sheet["ok"] and zebra["png"] != curv["png"]
    assert curv["frame"]["scale"] > 0 and sheet["frame"] is None
    bad = session.call("snapshot", curvature="sharpness")
    assert not bad["ok"] and "curvature" in bad["error"]


def test_discovery_finds_api_and_guides():
    keys = {e.key for e in discovery.methods()}
    assert {"Box", "Part.fillet", "ShapeList.top", "import_step"} <= keys
    assert "fillet" in [e.key.split(".")[-1] for _, e in discovery.search("round the edges", 5)]
    assert discovery.guide("SKILL") is not None and discovery.guide("recipes/verification") is not None


def test_live_viewer_serves_the_scene(session):
    import gzip
    import json

    from smlibpy.mcp.live import LiveViewer

    session.clear()
    session.run("c = Cylinder(5, 10).colored('#d9925a')", label="a cylinder")
    viewer = LiveViewer(session)
    try:
        page = urllib.request.urlopen(viewer.url, timeout=10).read()
        assert b"/assets/viewer.js" in page and b"__NONCE__" not in page
        port = viewer.url.split(":")[2].split("/")[0]
        three = urllib.request.urlopen(f"http://127.0.0.1:{port}/assets/vendor/three.module.min.js", timeout=10).read()
        assert len(three) > 100_000
        raw = urllib.request.urlopen(viewer.url + "/scene", timeout=30).read()
        payload = json.loads(gzip.decompress(raw))
        scene = payload["scene"]
        assert scene["kind"] == "PART" and scene["stats"]["bodies"] == 1 and scene["stats"]["faces"] == 3
        assert scene["bodies"][0]["color"] == [217, 146, 90]
        assert scene["features"] == [{"no": 1, "name": "a cylinder", "kind": "run_script"}]
        step = urllib.request.urlopen(viewer.url + "/export/c.step", timeout=60).read()
        assert step.startswith(b"ISO-10303-21;")
    finally:
        viewer.close()


def test_stdio_server_round_trip():
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-m", "smlibpy.mcp"],
                                       env={**os.environ, "SMLIBPY_VIEWER": "off"})
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                init = await s.initialize()
                assert "run_script" in (init.instructions or "")
                names = {t.name for t in (await s.list_tools()).tools}
                assert {"run_script", "snapshot", "docs", "save"} <= names
                res = await s.call_tool("run_script", {"script": "part = Box(20, 10, 5)\npart.volume"})
                text = res.content[0].text
                assert "Value: 1000" in text and "Viewer: http://127.0.0.1" in text
                shot = await s.call_tool("snapshot", {"width": 200, "height": 150})
                image = [c for c in shot.content if c.type == "image"][0]
                assert base64.b64decode(image.data)[:4] == b"\x89PNG"
                note = [c for c in shot.content if c.type == "text"][0].text
                assert "frame: {" in note
                drawing = await s.call_tool("snapshot", {"width": 400, "height": 300, "drawing": True,
                                                         "highlight": "part.faces().top()",
                                                         "view": {"azimuth": 30, "elevation": 20}})
                assert [c.type for c in drawing.content] == ["image", "text"]
                assert "frame:" not in drawing.content[1].text
                stripes = await s.call_tool("snapshot", {"width": 300, "height": 200, "zebra": True,
                                                         "curvature": None})
                assert stripes.content[0].type == "image"
                curv = await s.call_tool("snapshot", {"width": 300, "height": 200,
                                                      "curvature": {"kind": "gaussian", "range": 0.01}})
                assert curv.content[0].type == "image"

    asyncio.run(go())


def test_viewer_shows_a_read_step_file_and_imported_assemblies(session, tmp_path):
    from test_step_import import _tool_with_supplemental_geometry

    path = tmp_path / "tool.step"
    path.write_text(_tool_with_supplemental_geometry())
    session.clear()
    r = session.run(f"tool = read_step({str(path)!r})\nparts = tool.bodies")
    assert r["ok"], r.get("error")
    # shown without show(): the read file, and the list of its bodies
    assert [item["name"] for item in r["scene"]] == ["tool", "parts[0]"]
    assert r["scene"][0]["frames"] == ["MCS (TOOL)"]
    scene = session.call("scene_json")["scene"]
    assert scene["kind"] == "PART"
    names = [b["name"] for b in scene["bodies"]]
    assert names.count("cylinder") == 2 and {"CUTTING EDGE LINE", "ARC", "PATH"} <= set(names)
    edge_line = next(b for b in scene["bodies"] if b["name"] == "CUTTING EDGE LINE")
    assert edge_line["edgeColor"] == [255, 0, 0] and not edge_line["faces"]
    [mcs] = scene["marks"]["frames"]
    assert (mcs["name"], mcs["o"], mcs["x"]) == ("MCS", [0, 0, 120], [0, 1, 0])
    assert [p["name"] for p in scene["marks"]["points"]] == ["CRP"]
    assert scene["bounds"]["max"][2] == pytest.approx(120)
    assert session.call("snapshot", size=(300, 200))["ok"]
