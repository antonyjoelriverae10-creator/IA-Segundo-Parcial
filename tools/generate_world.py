#!/usr/bin/env python3
"""Genera worlds/race.wbt (Webots R2023b, ENU) a partir de config/scenario.json.
Uso: python tools/generate_world.py [--scenario config/scenario.json] [--out worlds/race.wbt]"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "controllers", "race_supervisor"))
import layout  # noqa: E402

PROTO = "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/robots/robotis/darwin-op/protos/Darwin-op.proto"
ROBOT_Z = 0.33


def n(v):
    return "%.4f" % v


def solid(name, box, height, color, def_name=None):
    cx, cy, sx, sy = box
    d = "DEF %s " % def_name if def_name else ""
    return """%sSolid {
  translation %s %s %s
  children [ Shape { appearance PBRAppearance { baseColor %s roughness 1 metalness 0 } geometry Box { size %s %s %s } } ]
  name "%s"
  boundingObject Box { size %s %s %s }
}
""" % (d, n(cx), n(cy), n(height / 2), color, n(sx), n(sy), n(height), name, n(sx), n(sy), n(height))


def generate(cfg):
    w, tr, ep = cfg["world"], cfg["track"], cfg["episode"]
    cnt, L, sp = w["laneCount"], w["laneLength"], w["laneSpacing"]
    out = ["#VRML_SIM R2023b utf8", "", 'EXTERNPROTO "%s"' % PROTO, "",
           "WorldInfo {\n  basicTimeStep %d\n  coordinateSystem \"ENU\"\n}" % w.get("basicTimeStep", 16),
           "Viewpoint {\n  orientation 1 0 0 1.0\n  position %s %s %s\n}" % (n(L / 2), n(-0.5 * L), n(0.5 * L)),
           "Background { skyColor [ 0.6 0.8 1 ] }",
           "DirectionalLight { direction 0.3 0.5 -1 }",
           """Solid {
  translation %s %s -0.01
  children [ Shape { appearance PBRAppearance { baseColor 0.55 0.55 0.55 roughness 1 metalness 0 } geometry Box { size %s %s 0.02 } } ]
  name "floor"
  boundingObject Box { size %s %s 0.02 }
}""" % (n(L / 2), n((cnt - 1) * sp / 2), n(L + 4), n(cnt * sp), n(L + 4), n(cnt * sp)),
           'Robot {\n  name "race_supervisor"\n  controller "race_supervisor"\n  supervisor TRUE\n}']
    for i in range(cnt):
        oy = layout.lane_origin_y(w, i)
        for k, b in enumerate(layout.lane_walls(w, i)):
            out.append(solid("wall_%d_%d" % (i, k), b, 0.3, "0.8 0.8 0.8"))
        for k, b in enumerate(layout.lane_obstacles(cfg, i)):
            out.append(solid("obs_%d_%d" % (i, k), b, 0.3, "0.9 0.3 0.1"))
        # Marcadores visuales (sin boundingObject: no colisionan)
        out.append("""Solid {
  translation %s %s 0.003
  children [ Shape { appearance PBRAppearance { baseColor 1 1 1 roughness 1 metalness 0 } geometry Box { size 0.05 %s 0.006 } } ]
  name "start_line_%d"
}""" % (n(tr["startX"] - 0.2), n(oy), n(w["laneWidth"]), i))
        out.append("""DEF GOAL_%d Solid {
  translation %s %s 0.005
  children [ Shape { appearance PBRAppearance { baseColor 0 0.8 0 roughness 1 metalness 0 } geometry Cylinder { height 0.01 radius %s } } ]
  name "goal_%d"
}""" % (i, n(ep["goalXRange"][0]), n(oy), n(ep["goalRadius"]), i))
        out.append("""DEF ROBOT_%d Darwin-op {
  translation %s %s %s
  rotation 0 0 1 0
  name "darwin_%d"
  controller "locomotion"
  customData ""
}""" % (i, n(tr["startX"]), n(oy), n(ROBOT_Z), i))
    return "\n\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=os.path.join(ROOT, "config", "scenario.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "worlds", "race.wbt"))
    a = ap.parse_args()
    with open(a.scenario) as fh:
        text = generate(json.load(fh))
    with open(a.out, "w", newline="\n") as fh:
        fh.write(text)
    print("escrito", a.out)
