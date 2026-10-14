"""The smlibpy MCP server: CAD modeling for agents, by running Python scripts on SMLib.

Run it with ``smlibpy-mcp`` (or ``python -m smlibpy.mcp``); it speaks MCP over stdio.

Environment:
    SMLIBPY_VIEWER        how the live viewer reaches the user: ``host`` (the agent opens the
                          link in its own browser pane; the default in the Claude desktop
                          app), ``browser`` (the user's default browser; the default
                          elsewhere) or ``off`` (only the link is handed over)
    SMLIBPY_SNAPSHOT_DIR  where snapshots are kept (default: a folder in the temp directory)
"""

from __future__ import annotations

import itertools
import json
import os
import tempfile
import time
import webbrowser
from typing import Annotated, Any

import anyio
from pydantic import Field

try:  # MCP SDK 2.x
    from mcp.server.mcpserver import Image, MCPServer
except ImportError:  # 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer  # type: ignore[no-redef]
    from mcp.server.fastmcp import Image  # type: ignore[no-redef]

from . import discovery
from .live import LiveViewer
from .session import DEFAULT_TIMEOUT, Session

DOCS_BUDGET = 48_000
_counter = itertools.count(1)


def _mode() -> str:
    mode = os.environ.get("SMLIBPY_VIEWER", "").strip().lower()
    if mode in ("host", "browser", "off"):
        return mode
    return "host" if os.environ.get("CLAUDE_CODE_ENTRYPOINT") == "claude-desktop" else "browser"


_VIEWER_NOTE = {
    "host": ("VIEWER: the first script that makes a model returns \"Viewer: <link>\". Open that link in your own "
             "browser pane right then (in the Claude desktop app: preview_start with url set to the link), once; "
             "the page follows every later script by itself. Tell the user in a few words that the model is in "
             "the browser pane. view() gives the link again."),
    "browser": ("VIEWER: the first script that makes a model opens a live 3D viewer in the user's browser and "
                "returns \"Viewer: <link>\". The page follows every later script by itself; do not open it again. "
                "view() gives the link again."),
    "off": ("VIEWER: the first script that makes a model returns \"Viewer: <link>\", a live 3D view that follows "
            "every later script. Hand the link to the user once."),
}

_GUIDE = """\
smlibpy: CAD modeling in Python on SMLib, the NURBS solid modeling kernel. You build exact solids by \
writing Python scripts; this server runs them, shows the result in a live 3D viewer and renders \
pictures for you to check.

WORKFLOW
1. Plan the part: its dimensions, its features, the order you build them in. Read the guides for \
the operations you will use (docs) before you write the script.
2. Build with run_script. `from smlibpy import *`, math and numpy (as np) are already imported. \
Variables persist between calls: build in steps, one feature or a few at a time.
3. Verify after every step: print numbers (part.volume, part.bbox(), part.is_valid(), \
len(part.faces())) and look with snapshot(view="sheet"): a technical drawing (front view, the view \
from the left, top view; hidden edges dashed) with a shaded iso view. A picture alone can mislead \
(it is scaled to fit); a number alone can too. Fix what is wrong before going on. snapshot also \
cuts sections (section=), shows what a selection picks (highlight="part.edges().concave()"), marks \
points (markers=), proves smooth blends (zebra=, curvature=) and compares before and after (frame= \
and compare=); see recipes/verification.
4. Deliver: save(format="step", path=...) writes the exact model (also stl, glb, obj, smb, png, and dxf \
for sketches and sections). Tell the user where the file is.

CONVENTIONS
- Millimeters and degrees, Z up. Primitives are centered on the origin unless align= says otherwise.
- Shapes are values: operations return new shapes and never change their input. `a + b` fuses, \
`a - b` cuts, `a & b` intersects. Place with .translate(), .rotate(), .moved(Location) or a \
pattern (`GridLocations(...) * Cylinder(...)`).
- Select faces and edges by geometry, not by index: part.faces().top(), \
part.edges().parallel_to(Z), part.faces().cylinders().largest(). Indices change when the model changes.
- Sketches are planar regions drawn on XY: shapes (Rectangle(w, h, fillet=r), Circle, Slot, ...) combined \
with + - &, or Profile(...) for outlines with tangent arcs, rounded corners or symmetry (recipes/profiles); \
then .on(plane or face) and extrude or revolve.
- show(a, b) at the end of a script chooses what the viewer and snapshots display. Without it, \
every Part and Shell variable of the session is shown, tools and intermediate bodies included.
- Errors are ModelingError (or KernelError) with a hint. Read the hint: it says what to change.
- A script that fails changes nothing: the session keeps the state before it. checkpoint() before \
a risky step and restore() to go back. undo() takes back the last step when it was wrong. \
export_script(path) writes the whole session as one runnable .py file.
- Sheet metal: start from a plate (or sketch.extrude(thickness)), add flange() and fold() with \
radius= and k_factor=; flat_pattern(part) gives the blank, its outline and bend lines (guide sheetmetal/generic).

DOCS
list_methods(search="fillet the top edges") finds names in the API and the guides. \
describe_method("Part.fillet") gives the signature, the docstring and the notes written for it. \
docs(["SKILL", "recipes/verification"]) returns guides by key (the index is below).

{viewer}

SMLIBPY OR CLASSCAD
smlibpy is for results a script produces once: a part or a few parts, written to STEP, STL or \
glTF, variants from parameters, batch jobs, checks. The script is the model; to change the part \
you change the script and run it again. Nothing else is kept: no feature history, no constraints, \
no references between features, no assemblies with mates.
When the user wants a model that stays editable and regenerates, use the ClassCAD MCP (server \
"classcad") instead: sketches with constraints and dimensions, features built on faces and edges \
of earlier features, parts driven by expressions, assemblies with mates, or an app built around \
such models (with buerli or Buerligons). Tell the user that smlibpy is the wrong tool for that and \
point them to the ClassCAD MCP: https://classcad.ai. The install steps for agents, for every host, are in https://classcad.ai/llms.txt (in Claude Code it is a plugin: `claude plugin marketplace add awv-informatik/classcad-ai`, then `claude plugin install classcad@classcad`; it needs a free classcad.ch account). If it is installed already, build \
those models there.
Both meet at STEP: a part from here opens in ClassCAD as a solid. To hand one to Buerligons, \
save(format="step", path=...) here, then call the classcad load tool with format STP and that path.

METHOD INDEX
{methods}

GUIDES
{guides}
"""


def instructions() -> str:
    return _GUIDE.format(viewer=_VIEWER_NOTE[_mode()], methods=discovery.method_index(),
                         guides=discovery.guide_index() or "(none yet)")


def _snapshot_dir() -> str:
    d = os.environ.get("SMLIBPY_SNAPSHOT_DIR") or os.path.join(tempfile.gettempdir(), "smlibpy-snapshots")
    os.makedirs(d, exist_ok=True)
    return d


def _fmt_scene(scene: list[dict[str, Any]], explicit: bool) -> str:
    if not scene:
        return "Shown: nothing yet."
    rows = []
    for item in scene:
        bits = [item["type"]]
        if item["type"] == "StepModel":
            bits.append(f"{item['bodies']} bodies")
            for key in ("frames", "curves"):
                if item.get(key):
                    bits.append(f"{key} " + ", ".join(item[key]))
            if item.get("points"):
                bits.append(f"{item['points']} points")
        if "volume" in item:
            bits.append(f"volume {item['volume']}")
        if "faces" in item:
            bits.append(f"{item['faces']} faces")
        if item.get("valid") is False:
            bits.append("NOT VALID")
        if "bbox" in item:
            lo, hi = item["bbox"]["min"], item["bbox"]["max"]
            bits.append("size " + " × ".join(f"{h - l:.6g}" for l, h in zip(lo, hi)))
        if "problem" in item:
            bits.append(item["problem"])
        rows.append(f"- {item['name']}: " + ", ".join(bits))
    head = "Shown (chosen with show):" if explicit else "Shown (every part, shell and read STEP file):"
    return head + "\n" + "\n".join(rows)


def build_server(session: Session | None = None) -> Any:
    import smlibpy

    session = session or Session()
    viewer = LiveViewer(session)
    mode = _mode()
    server = MCPServer(name="smlibpy", instructions=instructions())

    def tool() -> Any:
        """Tools answer with plain content: structured output would repeat every text and
        cannot carry images."""
        try:
            return server.tool(structured_output=False)
        except TypeError:  # older SDKs
            return server.tool()

    async def call(fn: Any, *args: Any, **kwargs: Any) -> Any:
        return await anyio.to_thread.run_sync(lambda: fn(*args, **kwargs))

    def viewer_note(scene: list[dict[str, Any]]) -> str | None:
        if viewer.opened or not scene or mode == "none":
            return None
        viewer.opened = True
        link = viewer.url
        if mode == "browser":
            try:
                webbrowser.open(link)
                return f"Viewer: {link}\nThe live viewer is open in the user's browser; it follows every script."
            except Exception:
                return f"Viewer: {link}\nGive the user this link: a live 3D view that follows every script."
        if mode == "host":
            return (f"Viewer: {link}\nOpen it now in your browser pane (preview_start with this url), once; it "
                    "follows every script by itself.")
        return f"Viewer: {link}\nHand this link to the user once: a live 3D view that follows every script."

    @tool()
    async def run_script(
        script: Annotated[str, Field(description="Python code. `from smlibpy import *`, math and numpy (np) are "
                                     "imported; variables persist between calls. The value of a last bare "
                                     "expression is returned.")],
        label: Annotated[str | None, Field(description="A short name for the step, for your own record.")] = None,
        timeout_s: Annotated[float, Field(description="Seconds before the script is stopped (1 to 900).")] = DEFAULT_TIMEOUT,
    ) -> str:
        """Run a Python script in the modeling session and report what it printed, its value and what is shown."""
        res = await call(session.run, script, label, timeout_s)
        if not res.get("ok"):
            text = "Script failed:\n" + res.get("error", "")
            if res.get("output"):
                text += "\n\nOutput before the error:\n" + res["output"]
            return text + "\n\nThe session is as it was before this script."
        parts = []
        if res.get("output"):
            parts.append("Output:\n" + res["output"].rstrip())
        if res.get("value") is not None:
            parts.append("Value: " + res["value"])
        scene = res.get("scene", [])
        scene_answer = await call(session.call, "scene")
        parts.append(_fmt_scene(scene, bool(scene_answer.get("explicit"))))
        note = viewer_note(scene)
        if note:
            parts.append(note)
        return "\n\n".join(parts)

    @tool()
    async def snapshot(
        view: Annotated[str | dict[str, Any], Field(description=(
            "The camera, orthographic, Z up: iso (default, a true isometric view from front, right and above), front, "
            "back, left, right, top, bottom; {\"azimuth\": a, \"elevation\": e} in degrees (azimuth 0 = from the "
            "front, 90 = from +X; elevation 90 = from above); {\"direction\": [x, y, z], \"up\": [x, y, z]}, the "
            "direction the camera looks in; or 'x,y,z', the direction from the model towards the camera. "
            "'sheet' is the same as drawing=true."))] = "iso",
        names: Annotated[list[str] | None, Field(description="Variables to render instead of what is shown.")] = None,
        width: Annotated[int, Field(ge=64, le=3000)] = 1000,
        height: Annotated[int, Field(ge=64, le=3000)] = 750,
        edges: Annotated[bool, Field(description="Draw the edges of faces (hidden ones stay hidden).")] = True,
        path: Annotated[str | None, Field(description="Where to keep the PNG (absolute path). Default: the "
                                          "snapshot folder.")] = None,
        drawing: Annotated[bool | str | dict[str, Any] | None, Field(description=(
            "A technical drawing in one picture: front, top and side view with visible edges solid and hidden edges "
            "dashed, aligned at one scale, and a shaded iso view. true or 'first-angle' (ISO method E, Europe): the "
            "top view below the front view, the view from the left to its right. 'third-angle' (method A, US): the "
            "top view above it, the view from the right to its right. {\"projection\": ..., \"side\": "
            "\"left\" or \"right\", \"iso\": false} picks the side view or leaves out the iso view. To check "
            "against a reference drawing, use its method and compare view by view."))] = None,
        sheet: Annotated[bool | list[str | dict[str, Any]] | None, Field(description=(
            "Shaded views in one picture: true (top, iso, front, right), 4 views (2 x 2), or 2 views side by side "
            "as panels A and B, e.g. a view and its mirror image for a handedness check: "
            "[{\"azimuth\": 30, \"elevation\": 30}, {\"azimuth\": -30, \"elevation\": 30}]. Named ortho views "
            "share one scale."))] = None,
        lines: Annotated[bool, Field(description=(
            "Line style: no fill, visible edges and silhouettes solid, hidden edges dashed. Hidden lines show bores, "
            "pockets, hollow or solid. Also for every panel of a sheet."))] = False,
        section: Annotated[dict[str, Any] | str | None, Field(description=(
            "Cut at a plane and leave out the side its normal points to; the cut faces are filled and hatched, bores "
            "and pockets stay open, so wall thicknesses read directly. {\"origin\": [x, y, z], \"normal\": "
            "[x, y, z], \"cap\": true} or a Python expression of the session giving a Plane or a planar face, "
            "e.g. \"Plane.XZ\" or \"part.faces().top()[0]\". Combines with drawing for section views."))] = None,
        cap: Annotated[bool, Field(description=(
            "With section: fill and hatch the cut faces (default); false leaves the cut open and shows the inside "
            "walls darker."))] = True,
        highlight: Annotated[str | list[str] | None, Field(description=(
            "Python expressions of the session for shapes drawn in signal colors, faces and bodies orange, edges red: "
            "\"part.edges().concave()\", [\"part.faces().top()\", \"rim\"]. Shows whether a selection picks "
            "what you meant before you fillet or cut with it."))] = None,
        highlight_at: Annotated[list[list[float]] | None, Field(description=(
            "Points [[x, y, z], ...]: the face closest to each point is highlighted."))] = None,
        markers: Annotated[list[dict[str, Any]] | None, Field(description=(
            "Crosshairs with labels, drawn on top: [{\"position\": [x, y, z], \"label\": \"A\", \"color\": "
            "\"green\"}]. For probe points and measured positions."))] = None,
        annotate: Annotated[bool, Field(description=(
            "Measurement overlay: the size of the bounding box, the axes as the view shows them, a scale bar."))] = False,
        xray: Annotated[bool, Field(description="Translucent bodies: hidden geometry shines through.")] = False,
        zebra: Annotated[bool | dict[str, Any] | None, Field(description=(
            "Zebra stripes, the reflection of parallel light lines, from the exact surface normals: they break "
            "where faces meet with an edge (G0), kink at a circular fillet (G1) and bend smoothly across a "
            "curvature-continuous blend (G2). true, or {\"stripes\": 16, \"width\": 0.5, \"direction\": "
            "\"vertical\" | \"horizontal\" | [x, y, z]}. A round that curves along the light lines shows no "
            "stripes: turn the direction."))] = None,
        curvature: Annotated[str | dict[str, Any] | None, Field(description=(
            "Curvature map of the exact surfaces: \"mean\", \"gaussian\", \"max\" or \"min\" (principal "
            "curvatures). Blue concave, white flat, red convex; a legend gives the range in 1/mm. "
            "{\"kind\": \"max\", \"range\": 0.2} fixes the range. A circular fillet of radius r is one "
            "color (1/r), a G2 blend a ramp from white at its edges."))] = None,
        colors: Annotated[str, Field(description=(
            "'native' (default): the shapes' own colors; 'distinct': one color per body, to tell bodies apart."))
                          ] = "native",
        zoom: Annotated[float, Field(gt=0, le=50, description="Factor on the scale that fits the model.")] = 1.0,
        look_at: Annotated[list[float] | None, Field(description="[x, y, z] that lands in the middle.")] = None,
        frame: Annotated[dict[str, float] | None, Field(description=(
            "The frame an earlier snapshot of the same view and size reported: keeps its scale and middle, so a "
            "snapshot of the changed model compares pixel by pixel. Overrides zoom and look_at."))] = None,
        compare: Annotated[str | None, Field(description=(
            "Path of an earlier snapshot of the same view and size (take this one with its frame): reports the "
            "pixels that changed and adds a picture with them in red."))] = None,
        supersample: Annotated[int, Field(ge=1, le=4, description="Antialiasing factor (default 2).")] = 2,
    ) -> list[Any]:
        """Render a picture of the model (Z up, orthographic, scaled to fit) to check what you built: a view, a
        technical drawing with hidden edges dashed, sheets of views, sections, highlights of selections, markers,
        x-ray, zebra stripes and curvature maps for surface quality, and before/after comparisons."""
        v: Any = view
        if isinstance(view, str) and "," in view:
            v = tuple(float(c) for c in view.split(","))
        if drawing not in (None, False) or (isinstance(view, str) and view == "sheet"):
            kind = "drawing"
        elif sheet not in (None, False):
            kind = "sheet"
        else:
            kind = view.replace(",", "_") if isinstance(view, str) else "view"
        target = path or os.path.join(_snapshot_dir(), f"{time.strftime('%H%M%S')}-{next(_counter)}-{kind}.png")
        options = dict(drawing=drawing, sheet=sheet, lines=lines, section=section, cap=cap, highlight=highlight,
                       highlight_at=highlight_at, markers=markers, annotate=annotate, xray=xray, colors=colors,
                       zoom=zoom, look_at=look_at, frame=frame, compare=compare, supersample=supersample,
                       zebra=zebra, curvature=curvature)
        res = await call(session.call, "snapshot", 300.0, view=v, size=(width, height), names=names, edges=edges,
                         path=target, **{k: x for k, x in options.items() if x is not None})
        if not res.get("ok"):
            return [f"Snapshot failed: {res.get('error')}"]
        lines_out = [f"Saved: {target}", f"Shown: {', '.join(res.get('shown', []))}."]
        if res.get("frame"):
            lines_out.append(f"frame: {json.dumps(res['frame'])} (pass it as frame, with the same view and size, to "
                             "compare a later snapshot pixel by pixel)")
        content: list[Any] = [Image(data=res["png"], format="png")]
        diff = res.get("diff")
        if diff:
            if diff["changed"]:
                x0, y0, x1, y1 = diff["bbox"]
                lines_out.append(f"Compared with {compare}: {diff['changed']} pixels changed "
                                 f"({100 * diff['fraction']:.3g} %), in x {x0}..{x1}, y {y0}..{y1}; red in "
                                 f"{diff['path']}")
                content.append(Image(data=diff["png"], format="png"))
            else:
                lines_out.append(f"Compared with {compare}: no pixel changed.")
        lines_out.append("Pictures are scaled to fit: compare sizes with numbers, not by eye. Send the file to the "
                         "user when they should see it.")
        content.append("\n".join(lines_out))
        return content

    @tool()
    async def scene() -> str:
        """What the viewer and snapshots show, and every shape variable of the session."""
        shown = await call(session.call, "scene")
        allv = await call(session.call, "vars")
        text = _fmt_scene(shown.get("scene", []), bool(shown.get("explicit")))
        rows = [f"- {v['name']}: {v['type']}" + (f", volume {v['volume']}" if "volume" in v else "")
                + (f", {v['count']} items" if "count" in v else "") for v in allv.get("vars", [])]
        return text + "\n\nVariables:\n" + ("\n".join(rows) if rows else "(none)") + \
            f"\n\nScripts in this session: {len(session.history)}; checkpoints: " + \
            (", ".join(k for k, _ in session.checkpoints) or "none")

    @tool()
    async def inspect(
        name: Annotated[str, Field(description="A variable, or an expression such as part.faces().top().")],
        faces: Annotated[bool, Field(description="List the faces: type, area, center, normal or radius.")] = False,
        edges: Annotated[bool, Field(description="List the edges: type, length, center, radius.")] = False,
    ) -> str:
        """Measurements of a shape: type, validity, volume, area, size, faces and edges."""
        res = await call(session.call, "inspect", name=name, faces=faces, edges=edges)
        if not res.get("ok"):
            return res.get("error", "failed")
        return json.dumps(res["info"], indent=1)

    @tool()
    async def list_methods(
        search: Annotated[str | list[str] | None, Field(description="Words for what you want to do, e.g. 'round "
                                                        "the top edges' or ['hole', 'pattern'].")] = None,
        limit: Annotated[int, Field(ge=1, le=200)] = 30,
    ) -> str:
        """Find functions, methods and guides of the API. Without search: the whole index."""
        if not search:
            return discovery.method_index() + "\n\nGUIDES\n" + (discovery.guide_index() or "(none yet)")
        hits = discovery.search(search, limit)
        if not hits:
            return "Nothing matches. list_methods() without search lists everything."
        return "\n".join(f"{e.key} [{e.kind}]: {e.line if e.kind not in ('guide',) else e.line}"
                         + (f" | {e.summary}" if e.summary and e.kind != "guide" else "") for _, e in hits)

    @tool()
    async def describe_method(
        method: Annotated[str, Field(description="A name such as 'fillet', 'Part.fillet' or 'GridLocations'.")],
    ) -> str:
        """Signature, docstring and the notes written for one function or method."""
        e = discovery.find_method(method)
        if e is None:
            hits = discovery.search(method, 8)
            return f"No method {method!r}. Close: " + ", ".join(h.key for _, h in hits)
        text = f"{e.line}\n\n{e.text or e.summary or '(no docstring)'}"
        g = discovery.guide_for(e)
        if g is not None:
            text += f"\n\n--- guide {g.key} ---\n{g.text}"
        return text

    @tool()
    async def docs(
        keys: Annotated[list[str], Field(description="Guide keys from the index, e.g. ['SKILL', "
                                         "'recipes/verification', 'features/fillet'].", min_length=1, max_length=24)],
    ) -> str:
        """Read guides of the smlibpy skill."""
        out, used = [], 0
        for k in keys:
            g = discovery.guide(k)
            if g is None:
                hits = discovery.search(k, 5)
                out.append(f"--- {k}: not found. Close: {', '.join(h.key for _, h in hits)}")
                continue
            piece = f"--- {g.key} ---\n{g.text}"
            if used + len(piece) > DOCS_BUDGET:
                out.append(f"--- {g.key}: left out (the answer is full); ask for it again on its own.")
                continue
            used += len(piece)
            out.append(piece)
        return "\n\n".join(out)

    @tool()
    async def save(
        format: Annotated[str, Field(description="step, stl, glb, gltf, obj, smb (native, one body), png, or dxf "
                                     "(2D: sketches, sections, wires; parts are left out).")] = "step",
        path: Annotated[str | None, Field(description="Absolute file path. Default: a file in the temp folder.")] = None,
        names: Annotated[list[str] | None, Field(description="Variables to write instead of what is shown.")] = None,
    ) -> str:
        """Write the shown shapes (or the named ones) to a file."""
        res = await call(session.call, "save", 600.0, format=format, path=path, names=names)
        if not res.get("ok"):
            return f"Save failed: {res.get('error')}"
        return f"Wrote {res['path']} ({res['bytes']} bytes): {', '.join(res['shapes'])}."

    @tool()
    async def load(
        path: Annotated[str, Field(description="Absolute path of a STEP (.step, .stp), SMLib (.smb), STL or DXF "
                                   "(.dxf, read as a sketch) file.")],
        name: Annotated[str | None, Field(description="Variable to hold it. Default: from the file name.")] = None,
    ) -> str:
        """Read a file into a session variable (a Part, a Shell, a list of them, a Mesh for STL, a Sketch for DXF)."""
        res = await call(session.load, path, name)
        if not res.get("ok"):
            return f"Load failed: {res.get('error')}"
        text = f"Loaded {path} as `{res['name']}`: {res.get('type')}"
        if "volume" in res:
            text += f", volume {res['volume']}"
        shown = await call(session.call, "scene")
        note = viewer_note(shown.get("scene", []))
        return text + (f"\n\n{note}" if note else "")

    @tool()
    async def undo() -> str:
        """Remove the last script and rebuild: one step back, without a checkpoint."""
        try:
            note = await call(session.undo)
        except KeyError as e:
            return str(e.args[0])
        shown = await call(session.call, "scene")
        scene_text = _fmt_scene(shown.get("scene", []), bool(shown.get("explicit")))
        return note + "\n\n" + scene_text

    @tool()
    async def export_script(
        path: Annotated[str, Field(description="Absolute file path for the .py export.")],
    ) -> str:
        """Write the session's scripts as one runnable Python file: the part as a script."""
        try:
            saved = await call(session.export_script, path)
            return f"Exported {len(session.history)} scripts to {saved}"
        except OSError as e:
            return f"Export failed: {type(e).__name__}: {e}"

    @tool()
    async def checkpoint(label: Annotated[str | None, Field(description="A name to restore it by.")] = None) -> str:
        """Remember the current state of the session, to go back to it with restore."""
        name = await call(session.checkpoint, label)
        return f"Checkpoint {name!r} after {len(session.history)} scripts."

    @tool()
    async def restore(label: Annotated[str | None, Field(description="Checkpoint name; default: the last.")] = None) -> str:
        """Go back to a checkpoint: the session is rebuilt from the scripts that led to it."""
        try:
            return await call(session.restore, label)
        except KeyError as e:
            return str(e.args[0])

    @tool()
    async def clear() -> str:
        """Start over: an empty session (checkpoints are dropped too)."""
        await call(session.clear)
        return "The session is empty."

    @tool()
    async def view(open: Annotated[bool, Field(description="Also open it in the user's default browser.")] = False) -> str:
        """The link of the live 3D viewer."""
        link = viewer.url
        viewer.opened = True
        if open:
            try:
                webbrowser.open(link)
            except Exception:
                pass
        return f"Viewer: {link}"

    server._smlibpy_session = session  # type: ignore[attr-defined]
    server._smlibpy_viewer = viewer  # type: ignore[attr-defined]
    _ = smlibpy
    return server


def main() -> None:
    server = build_server()
    try:
        server.run("stdio")
    finally:
        server._smlibpy_viewer.close()
        server._smlibpy_session.close()


if __name__ == "__main__":
    main()
