"""The process that runs the scripts of an MCP session.

Scripts run here and not in the server: a script that never ends, or a kernel that crashes,
costs a restart of this process and nothing else. The server talks to it over a pipe; every
request is a dict with an ``op`` and every answer a dict with ``ok``.
"""

from __future__ import annotations

import ast
import contextlib
import io
import linecache
import os
import sys
import tempfile
import traceback
from typing import Any

OUTPUT_LIMIT = 16_000
VALUE_LIMIT = 8_000

# Mesh quality for the live viewer and scene JSON: the angular tolerance (in degrees per
# unit) of the default tessellation. Higher is coarser and faster; lower is finer.
# Overridable for hosts that need finer or coarser default meshes.
_MESH_ANGLE_TOLERANCE = float(os.environ.get("SMLIBPY_MESH_ANGLE_TOLERANCE", "12.0"))


def main(conn: Any) -> None:
    # Anything the kernel or a library prints must not reach the server's stdout, which
    # carries the MCP protocol: send file descriptor 1 to stderr.
    try:
        os.dup2(2, 1)
    except OSError:
        pass
    sys.stdout = sys.stderr
    worker = _Worker()
    while True:
        try:
            request = conn.recv()
        except (EOFError, OSError):
            return
        op = request.pop("op", "")
        handler = getattr(worker, "op_" + op, None)
        try:
            answer = handler(**request) if handler else {"ok": False, "error": f"unknown request {op!r}"}
        except Exception as e:  # report, never die
            answer = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        try:
            conn.send(answer)
        except (EOFError, OSError):
            return


class _Worker:
    def __init__(self) -> None:
        import smlibpy

        self.sm = smlibpy
        self.ns: dict[str, Any] = {"__name__": "__session__", "__builtins__": __builtins__}
        exec("from smlibpy import *\nimport math\nimport numpy as np", self.ns)
        self.ns["show"] = self._show
        self.shown: list[Any] | None = None  # set by show(...); None: every part, shell and read STEP file
        self.count = 0

    # -- scripts ----------------------------------------------------------------
    def _show(self, *shapes: Any, **_ignored: Any) -> None:
        """In a session, show(...) chooses what the viewer and snapshots display; show()
        without arguments goes back to showing every part, shell (also in lists) and read STEP file."""
        flat: list[Any] = []
        for s in shapes:
            flat.extend(s) if isinstance(s, (list, tuple)) else flat.append(s)
        self.shown = flat or None

    def op_run(self, code: str, label: str | None = None) -> dict[str, Any]:
        self.count += 1
        filename = f"<script {self.count}>"
        linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
        out = io.StringIO()
        value: Any = None
        try:
            tree = ast.parse(code, filename, "exec")
            last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop().value)
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                exec(compile(tree, filename, "exec"), self.ns)
                if last is not None:
                    value = eval(compile(last, filename, "eval"), self.ns)
        except BaseException as e:  # SystemExit and KeyboardInterrupt too: the session goes on
            return {"ok": False, "error": _script_error(e, filename), "output": _tail(out.getvalue())}
        return {"ok": True, "output": _tail(out.getvalue()), "value": None if value is None else self._describe(value),
                "scene": self.op_scene()["scene"]}

    # -- what is on screen ----------------------------------------------------------
    def _named(self, obj: Any) -> str:
        for k, v in self.ns.items():
            if v is obj and not k.startswith("_"):
                return k
        return getattr(obj, "name", None) or type(obj).__name__

    def _items(self, names: list[str] | None = None) -> list[tuple[str, Any]]:
        """What is shown: the variables asked for, what show(...) chose, or else every part,
        shell, list of them (an imported assembly) and read STEP file of the session."""
        from smlibpy import Mesh, Part, Shape, ShapeList, Shell, StepModel

        if names:
            out = []
            for n in names:
                if n not in self.ns:
                    raise KeyError(f"no variable {n!r} in the session")
                out.append((n, self.ns[n]))
            return out
        if self.shown is not None:
            return [(self._named(s), s) for s in self.shown if isinstance(s, (Shape, Mesh, StepModel))]
        out = []
        for k, v in self.ns.items():
            if k.startswith("_") or isinstance(v, type):
                continue
            if isinstance(v, (Part, Shell, StepModel)):
                out.append((k, v))
            elif isinstance(v, ShapeList) and v and all(isinstance(s, (Part, Shell)) for s in v):
                out.extend((f"{k}[{i}]", s) for i, s in enumerate(v))
        return out

    def op_scene(self) -> dict[str, Any]:
        return {"ok": True, "scene": [{"name": n, **self._summary(s)} for n, s in self._items()],
                "explicit": self.shown is not None}

    def op_glb(self) -> dict[str, Any]:
        from smlibpy.viewer import _glb_bytes

        items = _drawn(self._items())
        if not items:
            return {"ok": True, "glb": None, "summary": ""}
        shapes = [s.named(n) if hasattr(s, "named") and not getattr(s, "name", None) else s for n, s, _ in items]
        glb, summary = _glb_bytes(shapes, None, _MESH_ANGLE_TOLERANCE, True)
        return {"ok": True, "glb": glb, "summary": summary}

    def op_scene_json(self) -> dict[str, Any]:
        """What the viewer page draws (the scene format of classcad-ai's viewer): per body its
        faces as meshes (positions, normals, triangles, planar or not), its edges as
        polylines, its color, and the bounds of everything. A read STEP file adds its curves
        as bodies of edges in their color, and its coordinate systems and points as
        ``marks``: ``{"frames": [{"name", "o", "x", "y", "z"}], "points": [{"name", "p",
        "color"}]}``."""
        import numpy as np

        from smlibpy import Edge, Face, Mesh, Wire
        from smlibpy.colors import to_rgb
        from smlibpy.exporters import _curve_points, _edge_polylines
        from smlibpy.mesh import face_meshes, tessellate

        def flat(a: Any) -> list[float]:
            return [round(float(v), 4) for v in np.asarray(a, dtype=float).ravel()]

        def mesh_face(m: Any, planar: bool) -> dict[str, Any]:
            normals = None
            if m.normals is not None and len(m.triangles):
                n = np.zeros_like(m.vertices)
                n[m.triangles.ravel()] = np.asarray(m.normals, dtype=float).reshape(-1, 3)
                normals = flat(n)
            return {"p": flat(m.vertices), "n": normals, "i": [int(v) for v in m.triangles.ravel()], "flat": planar}

        def rgb(color: Any) -> list[int] | None:
            return [round(c * 255) for c in to_rgb(color)] if color is not None else None

        bodies, placements = [], []
        stats = {"bodies": 0, "faces": 0, "triangles": 0, "edges": 0}
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        items = self._items()
        for name, shape, color in _drawn(items):
            body: dict[str, Any] = {"id": len(bodies), "name": getattr(shape, "name", None) or name,
                                    "color": rgb(color), "opacity": 1, "faces": [], "edges": []}
            try:
                if isinstance(shape, Mesh):
                    body["faces"].append(mesh_face(shape, False))
                elif isinstance(shape, (Edge, Wire)):
                    body["edges"] = [flat(pl) for pl in _curve_points(shape)]
                    # a curve is drawn in its color, unless white or black would vanish on the stage
                    if color is not None and max(to_rgb(color)) > 0.1 and min(to_rgb(color)) < 0.9:
                        body["edgeColor"] = body["color"]
                else:
                    faces = [shape] if isinstance(shape, Face) else list(shape.faces())
                    tol = max(shape.bbox().diagonal, 1e-6) * 1e-3
                    # the body is meshed in one piece and cut by face: imported faces the
                    # kernel cannot mesh on their own show too
                    try:
                        meshes = face_meshes(shape, tol, _MESH_ANGLE_TOLERANCE) if not isinstance(shape, Face) else None
                    except Exception:
                        meshes = None
                    for i, f in enumerate(faces):
                        try:
                            m = meshes[i] if meshes is not None else tessellate(f, tol, _MESH_ANGLE_TOLERANCE)
                            planar = f.geom_type == "plane"
                        except Exception:  # one face the kernel cannot mesh: the others still show
                            continue
                        if len(m.triangles):
                            body["faces"].append(mesh_face(m, planar))
                    try:
                        lines = _edge_polylines(shape) if not isinstance(shape, Face) else _curve_points(shape)
                        body["edges"] = [flat(pl) for pl in lines]
                    except Exception:
                        pass
            except Exception:
                continue
            if not body["faces"] and not body["edges"]:
                continue
            for f in body["faces"]:
                p = np.asarray(f["p"]).reshape(-1, 3)
                if len(p):
                    lo, hi = np.minimum(lo, p.min(axis=0)), np.maximum(hi, p.max(axis=0))
                stats["triangles"] += len(f["i"]) // 3
            for e in body["edges"]:
                p = np.asarray(e).reshape(-1, 3)
                if len(p):
                    lo, hi = np.minimum(lo, p.min(axis=0)), np.maximum(hi, p.max(axis=0))
            stats["faces"] += len(body["faces"])
            stats["edges"] += len(body["edges"])
            placements.append({"body": len(bodies), "m": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]})
            bodies.append(body)
        marks: dict[str, list[dict[str, Any]]] = {"frames": [], "points": []}
        for model in _step_models(items):
            for f in model.frames:
                p = f.plane
                marks["frames"].append({"name": f.name, "o": flat(p.origin.to_tuple()), "x": flat(p.x_dir.to_tuple()),
                                        "y": flat(p.y_dir.to_tuple()), "z": flat(p.z_dir.to_tuple())})
            for pt in model.points:
                marks["points"].append({"name": pt.name, "p": flat(pt.point.to_tuple()), "color": rgb(pt.color)})
        for p in [m["o"] for m in marks["frames"]] + [m["p"] for m in marks["points"]]:
            lo, hi = np.minimum(lo, p), np.maximum(hi, p)
        stats["bodies"] = len(bodies)
        names = [b["name"] for b in bodies]
        shown = bodies or marks["frames"] or marks["points"]
        scene = {"kind": "PART" if shown else "EMPTY",
                 "name": names[0] if len(names) == 1 else (f"{len(names)} shapes" if names else ""),
                 "bodies": bodies, "placements": placements, "features": [], "stats": stats, "marks": marks,
                 "bounds": {"min": [float(v) for v in lo], "max": [float(v) for v in hi]} if shown and np.isfinite(lo).all() else None}
        return {"ok": True, "scene": scene}

    def op_snapshot(self, view: Any = "iso", size: tuple[int, int] = (1000, 750), names: list[str] | None = None,
                    edges: bool = True, path: str | None = None, highlight: Any = None, section: Any = None,
                    compare: str | None = None, **options: Any) -> dict[str, Any]:
        """A picture of what is shown. ``highlight`` and a ``section`` given as text are Python
        expressions evaluated in the session; ``compare`` is the path of an earlier snapshot."""
        from smlibpy.render import diff_images, render

        items = self._items(names)
        if not items:
            return {"ok": False, "error": "nothing to show: no part or shell yet, and show(...) chose nothing"}
        if isinstance(highlight, str):
            highlight = [highlight]
        if highlight:
            highlight = [self._evaluate(e, "highlight") for e in highlight]
        if isinstance(section, str):
            section = self._evaluate(section, "section")
        pic = render(*[s for _, s in items], view=view, size=tuple(size), edges=edges, path=path, highlight=highlight,
                     section=section, **options)
        answer: dict[str, Any] = {"ok": True, "png": bytes(pic), "shown": [n for n, _ in items], "frame": pic.frame}
        if compare:
            d = diff_images(os.path.abspath(os.path.expanduser(compare)), pic)
            diff_path = os.path.splitext(path)[0] + "-diff.png" if path else None
            if diff_path:
                d.save(diff_path)
            answer["diff"] = {"changed": d.changed, "fraction": d.fraction, "bbox": d.bbox, "path": diff_path,
                              "png": bytes(d.picture)}
        return answer

    def _evaluate(self, expression: str, what: str) -> Any:
        try:
            return eval(expression, self.ns)
        except Exception as e:
            raise ValueError(f"{what} {expression!r}: {type(e).__name__}: {e}") from None

    # -- looking at things ----------------------------------------------------------
    def op_vars(self) -> dict[str, Any]:
        from smlibpy import Mesh, Shape, ShapeList, StepModel

        rows = []
        for k, v in self.ns.items():
            if k.startswith("_") or k in ("math", "np"):
                continue
            if isinstance(v, (Shape, ShapeList, Mesh, StepModel)) and k not in _LIBRARY_NAMES:
                rows.append({"name": k, **self._summary(v)})
        return {"ok": True, "vars": rows}

    def op_inspect(self, name: str, faces: bool = False, edges: bool = False) -> dict[str, Any]:
        if name not in self.ns:
            try:
                obj = eval(name, self.ns)
            except Exception:
                return {"ok": False, "error": f"no variable {name!r} in the session"}
        else:
            obj = self.ns[name]
        info = {"name": name, **self._summary(obj, full=True)}
        if faces and hasattr(obj, "faces"):
            info["faces"] = [_face_row(i, f) for i, f in enumerate(obj.faces())]
        if edges and hasattr(obj, "edges"):
            info["edges"] = [_edge_row(i, e) for i, e in enumerate(obj.edges())]
        return {"ok": True, "info": info}

    def _summary(self, obj: Any, full: bool = False) -> dict[str, Any]:
        from smlibpy import Edge, Face, Mesh, Part, Shape, ShapeList, Shell, Sketch, StepModel, Wire

        d: dict[str, Any] = {"type": type(obj).__name__}
        try:
            if isinstance(obj, StepModel):
                d.update(bodies=len(obj.bodies), frames=[f"{f.name} ({f.product})" for f in obj.frames],
                         curves=[f"{c.name} ({c.product})" for c in obj.curves], points=len(obj.points))
                return d
            if isinstance(obj, ShapeList):
                d["count"] = len(obj)
                kinds: dict[str, int] = {}
                for s in obj:
                    kinds[type(s).__name__] = kinds.get(type(s).__name__, 0) + 1
                d["items"] = kinds
                return d
            if isinstance(obj, Mesh):
                d.update(triangles=len(obj.triangles), volume=_r(obj.volume))
                return d
            if not isinstance(obj, Shape):
                return d
            if getattr(obj, "name", None):
                d["label"] = obj.name
            bb = obj.bbox()
            d["bbox"] = {"min": [_r(c) for c in bb.min.to_tuple()], "max": [_r(c) for c in bb.max.to_tuple()]}
            if isinstance(obj, Part):
                d["valid"] = bool(obj.is_valid())
                d["volume"] = _r(obj.volume)
            if isinstance(obj, (Part, Shell, Sketch, Face)):
                d["area"] = _r(obj.area)
            if isinstance(obj, (Part, Shell, Sketch)):
                d["faces"] = len(obj.faces())
                d["edges"] = len(obj.edges())
            if isinstance(obj, (Edge, Wire)):
                d["length"] = _r(obj.length)
            if isinstance(obj, (Face, Edge)):
                d["geom_type"] = obj.geom_type
            if full and isinstance(obj, (Part, Shell, Sketch)):
                kinds = {}
                for f in obj.faces():
                    kinds[f.geom_type] = kinds.get(f.geom_type, 0) + 1
                d["face_types"] = kinds
                d["center"] = [_r(c) for c in obj.center().to_tuple()]
                if getattr(obj, "color", None) is not None:
                    d["color"] = str(obj.color)
        except Exception as e:  # a broken body still gets a line
            d["problem"] = f"{type(e).__name__}: {e}"
        return d

    def _describe(self, value: Any) -> str:
        from smlibpy import Mesh, Shape, ShapeList, StepModel

        if isinstance(value, (Shape, ShapeList, Mesh, StepModel)):
            return repr(value)[:VALUE_LIMIT]
        text = repr(value)
        return text if len(text) <= VALUE_LIMIT else text[:VALUE_LIMIT] + f" … ({len(text)} characters)"

    # -- files --------------------------------------------------------------------
    def op_save(self, format: str, path: str | None = None, names: list[str] | None = None) -> dict[str, Any]:
        items = [(n, x) for n, x, _ in _drawn(self._items(names), curves=False)]
        if not items:
            return {"ok": False, "error": "nothing to save: no part or shell yet, and show(...) chose nothing"}
        ext = {"step": ".step", "stp": ".step", "stl": ".stl", "glb": ".glb", "gltf": ".gltf", "obj": ".obj",
               "smb": ".smb", "png": ".png", "dxf": ".dxf"}.get(format.lower())
        if ext is None:
            return {"ok": False, "error": f"unknown format {format!r}: step, stl, glb, gltf, obj, smb, png or dxf"}
        if path is None:
            base = items[0][0] if len(items) == 1 else "model"
            path = os.path.join(tempfile.gettempdir(), "smlibpy-files", base + ext)
        path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shapes = [s.named(n) if hasattr(s, "named") and not getattr(s, "name", None) else s for n, s in items]
        if ext == ".png":
            from smlibpy.render import render

            render(*shapes, path=path, size=(1200, 900))
        elif ext == ".smb":
            if len(shapes) != 1:
                return {"ok": False, "error": "an .smb file holds one body: name it with names=[...]"}
            self.sm.save(shapes[0], path)
        elif ext == ".dxf":
            flat = [s for s in shapes if not isinstance(s, self.sm.Part)]
            if not flat:
                return {"ok": False, "error": "a DXF file is a 2D drawing: name sketches, sections, wires or edges "
                                              "with names=[...] (a part's cross-section: part.section(plane))"}
            self.sm.export_dxf(flat, path)
            items = [(n, s) for n, s in items if not isinstance(s, self.sm.Part)]
        else:
            self.sm.export(shapes, path)
        return {"ok": True, "path": path, "bytes": os.path.getsize(path), "shapes": [n for n, _ in items]}

    def op_load(self, path: str, name: str | None = None) -> dict[str, Any]:
        path = os.path.abspath(os.path.expanduser(path))
        obj = self.sm.load(path)
        if name is None:
            stem = os.path.splitext(os.path.basename(path))[0]
            name = "".join(c if c.isalnum() else "_" for c in stem).strip("_") or "loaded"
            if name[0].isdigit():
                name = "part_" + name
        self.ns[name] = obj
        return {"ok": True, "name": name, **self._summary(obj)}

    def op_ping(self) -> dict[str, Any]:
        return {"ok": True, "pid": os.getpid()}


def _drawn(items: list[tuple[str, Any]], curves: bool = True) -> list[tuple[str, Any, Any]]:
    """(name, shape, color) of what is drawn: a read STEP file (``read_step``) gives its
    bodies and, with ``curves``, its curves in their colors."""
    from smlibpy import StepModel

    out: list[tuple[str, Any, Any]] = []
    for name, s in items:
        if isinstance(s, StepModel):
            out.extend((b.name or f"{name}.bodies[{i}]", b, b.color) for i, b in enumerate(s.bodies))
            if curves:
                out.extend((c.name or f"{name}.curves[{i}]", c.shape, c.color) for i, c in enumerate(s.curves))
        else:
            out.append((name, s, getattr(s, "color", None)))
    return out


def _step_models(items: list[tuple[str, Any]]) -> list[Any]:
    from smlibpy import StepModel

    return [s for _, s in items if isinstance(s, StepModel)]


_LIBRARY_NAMES = {"X", "Y", "Z"}


def _r(v: float) -> float:
    return float(f"{float(v):.6g}")


def _face_row(i: int, f: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"i": i, "type": f.geom_type, "area": _r(f.area),
                           "center": [_r(c) for c in f.center().to_tuple()]}
    try:
        if f.geom_type == "plane":
            row["normal"] = [_r(c) for c in f.normal_at().to_tuple()]
        elif f.geom_type in ("cylinder", "cone", "sphere", "torus"):
            r = f.radius
            if r is not None:
                row["radius"] = _r(r)
    except Exception:
        pass
    return row


def _edge_row(i: int, e: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"i": i, "type": e.geom_type, "length": _r(e.length),
                           "center": [_r(c) for c in e.center().to_tuple()]}
    try:
        if e.geom_type == "circle":
            row["radius"] = _r(e.radius)
    except Exception:
        pass
    return row


def _tail(text: str) -> str:
    if len(text) <= OUTPUT_LIMIT:
        return text
    return f"… ({len(text) - OUTPUT_LIMIT} characters left out)\n" + text[-OUTPUT_LIMIT:]


def _script_error(e: BaseException, filename: str) -> str:
    """The error with the lines of the script that led to it (library frames left out)."""
    if isinstance(e, SyntaxError):
        line = (e.text or "").rstrip()
        return f"SyntaxError: {e.msg} (line {e.lineno})\n    {line}"
    frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == filename]
    lines = [f"line {f.lineno}: {(f.line or '').strip()}" for f in frames]
    message = "".join(traceback.format_exception_only(type(e), e)).strip()
    return "\n".join(lines + [message]) if lines else message
