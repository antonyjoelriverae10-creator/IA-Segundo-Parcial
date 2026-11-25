"""Geometria de la pista, compartida por tools/generate_world.py y el entorno.
Cajas alineadas a los ejes como (cx, cy, sx, sy) en coordenadas de mundo (ENU, z arriba)."""

WALL_T = 0.05  # grosor de las paredes


def lane_origin_y(world, i):
    return i * world["laneSpacing"]


def lane_walls(world, i):
    L, W, oy, t = world["laneLength"], world["laneWidth"], lane_origin_y(world, i), WALL_T
    return [
        (L / 2, oy + W / 2 + t / 2, L, t),   # lateral +y
        (L / 2, oy - W / 2 - t / 2, L, t),   # lateral -y
        (L + t / 2, oy, t, W),               # fondo (tras la meta)
        (-t / 2, oy, t, W),                  # espalda (tras la salida)
    ]


def lane_obstacles(cfg, i):
    oy = lane_origin_y(cfg["world"], i)
    return [(o["x"], oy + o["y"], o["sx"], o["sy"]) for o in cfg["track"]["obstacles"]]
