#!/usr/bin/env python3
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "worlds" / "circuito_curvo.wbt"

PROTO = "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/robots/robotis/darwin-op/protos/Darwin-op.proto"

# S sinusoidal: dos curvas amplias y transiciones suaves.
X0, X1 = -8.0, 8.0
AMPLITUDE = 2.4
HALF_WIDTH = 1.75
BARRIER_HEIGHT = 0.28
BARRIER_THICKNESS = 0.09
SAMPLES = 40

def f(v):
    return f"{v:.4f}"

def center(t):
    x = X0 + (X1-X0)*t
    y = AMPLITUDE * math.sin(2*math.pi*t)
    return x, y

def frame(t):
    x, y = center(t)
    dx = X1-X0
    dy = AMPLITUDE*2*math.pi*math.cos(2*math.pi*t)
    length = math.hypot(dx, dy)
    return x, y, -dy/length, dx/length

def box(name, x, y, z, sx, sy, sz, color, angle=0):
    return f'''Solid {{
  translation {f(x)} {f(y)} {f(z)}
  rotation 0 0 1 {f(angle)}
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor {color}
        roughness 0.9
        metalness 0
      }}
      geometry Box {{ size {f(sx)} {f(sy)} {f(sz)} }}
    }}
  ]
  name "{name}"
  boundingObject Box {{ size {f(sx)} {f(sy)} {f(sz)} }}
}}'''

def cylinder(name, x, y, radius, height, color):
    return f'''Solid {{
  translation {f(x)} {f(y)} {f(height/2+0.025)}
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor {color}
        roughness 0.85
      }}
      geometry Cylinder {{
        radius {f(radius)}
        height {f(height)}
        subdivision 24
      }}
    }}
  ]
  name "{name}"
  boundingObject Cylinder {{
    radius {f(radius)}
    height {f(height)}
  }}
}}'''

def road_surface():
    left, right = [], []
    for i in range(SAMPLES+1):
        t = i/SAMPLES
        x, y, nx, ny = frame(t)
        left.append((x+nx*HALF_WIDTH, y+ny*HALF_WIDTH))
        right.append((x-nx*HALF_WIDTH, y-ny*HALF_WIDTH))

    points = left + right
    coords = ",\n".join(
        f"            {f(x)} {f(y)} 0.025" for x, y in points
    )
    n = SAMPLES+1
    faces = []
    for i in range(SAMPLES):
        faces.append(f"{i} {i+1} {n+i+1} {n+i} -1")
    indices = "\n".join("            "+face for face in faces)

    return f'''Solid {{
  name "curved_track_surface"
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor 0.25 0.27 0.29
        roughness 1
      }}
      geometry IndexedFaceSet {{
        coord Coordinate {{
          point [
{coords}
          ]
        }}
        coordIndex [
{indices}
          ]
        creaseAngle 0.5
      }}
    }}
  ]
}}'''

def barriers():
    result = []
    for side in (-1, 1):
        for i in range(SAMPLES):
            t1, t2 = i/SAMPLES, (i+1)/SAMPLES
            x1, y1 = center(t1)
            x2, y2 = center(t2)
            xm, ym, nx, ny = frame((t1+t2)/2)
            bx = xm + side*nx*(HALF_WIDTH+0.07)
            by = ym + side*ny*(HALF_WIDTH+0.07)
            length = math.hypot(x2-x1, y2-y1)+0.08
            angle = math.atan2(y2-y1, x2-x1)
            color = "0.88 0.88 0.84" if i % 2 else "0.82 0.16 0.12"
            result.append(box(
                f"track_wall_{side}_{i}",
                bx, by, BARRIER_HEIGHT/2+0.025,
                length, BARRIER_THICKNESS, BARRIER_HEIGHT,
                color, angle
            ))
    return result

def main():
    nodes = [
        "#VRML_SIM R2023b utf8",
        f'EXTERNPROTO "{PROTO}"',
        '''WorldInfo {
  basicTimeStep 16
  coordinateSystem "ENU"
}''',
        '''Viewpoint {
  orientation 1 0 0 1.0
  position 0 -13 17
}''',
        '''Background { skyColor [ 0.6 0.8 1 ] }''',
        '''DirectionalLight {
  direction 0.3 0.5 -1
  intensity 1.2
  castShadows TRUE
}''',
        '''Solid {
  translation 0 0 -0.12
  children [
    Shape {
      appearance PBRAppearance {
        baseColor 0.48 0.52 0.46
        roughness 1
      }
      geometry Box { size 20 11 0.2 }
    }
  ]
  name "floor"
  boundingObject Box { size 20 11 0.2 }
}''',
        road_surface(),
    ]

    nodes.extend(barriers())

    # Salida y meta: ambas perpendiculares al recorrido.
    nodes.append(box(
        "start_line", -7.65, 0, 0.045,
        0.10, 3.5, 0.025, "1 1 1"
    ))
    nodes.append(box(
        "finish_line", 7.65, 0, 0.045,
        0.10, 3.5, 0.025, "1 0.82 0.05"
    ))

    # Cuatro DARwIn-OP alineados sobre la salida.
    for i, offset in enumerate((-1.2, -0.4, 0.4, 1.2)):
        nodes.append(f'''DEF ROBOT_{i} Darwin-op {{
  translation -8.25 {f(offset)} 0.33
  rotation 0 0 1 0
  name "darwin_{i}"
  controller "<none>"
  customData ""
}}''')

    # Solo tres obstáculos geométricos, separados entre sí.
    nodes.append(box(
        "obstacle_block_1", -3.4, 1.35, 0.20,
        0.42, 0.42, 0.35, "0.92 0.30 0.10"
    ))
    nodes.append(cylinder(
        "obstacle_cylinder", 0.0, -1.55,
        0.24, 0.38, "0.12 0.42 0.85"
    ))
    nodes.append(box(
        "obstacle_block_2", 4.1, 1.15, 0.20,
        0.48, 0.38, 0.35, "0.95 0.65 0.08"
    ))

    OUT.write_text("\n\n".join(nodes)+"\n", encoding="utf-8")
    print("MUNDO ACTUALIZADO:", OUT)
    print("Forma: S continua con dos curvas amplias")
    print("Robots:", 4)
    print("Barreras laterales:", 2*SAMPLES)
    print("Obstaculos geometricos: 3")
    print("Salida y meta: colocadas en los extremos del recorrido")

if __name__ == "__main__":
    main()

