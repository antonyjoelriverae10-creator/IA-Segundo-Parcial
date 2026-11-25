"""Geometria 2D minima (sin dependencias): distancia punto-caja y rayo-caja."""
import math


def point_box_distance(px, py, box):
    cx, cy, sx, sy = box
    dx = max(abs(px - cx) - sx / 2, 0.0)
    dy = max(abs(py - cy) - sy / 2, 0.0)
    return math.hypot(dx, dy)


def ray_box(ox, oy, dx, dy, box, max_range):
    """Distancia a la que el rayo corta la caja (metodo de slabs), o None."""
    cx, cy, sx, sy = box
    tmin, tmax = 0.0, max_range
    for o, d, c, s in ((ox, dx, cx, sx), (oy, dy, cy, sy)):
        lo, hi = c - s / 2, c + s / 2
        if abs(d) < 1e-9:
            if o < lo or o > hi:
                return None
        else:
            t1, t2 = (lo - o) / d, (hi - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin, tmax = max(tmin, t1), min(tmax, t2)
            if tmin > tmax:
                return None
    return tmin


def scan(x, y, yaw, angles, boxes, max_range):
    """Lecturas de 'lidar virtual': una distancia por angulo (positivo = izquierda)."""
    out = []
    for a in angles:
        dx, dy = math.cos(yaw + a), math.sin(yaw + a)
        best = max_range
        for b in boxes:
            t = ray_box(x, y, dx, dy, b, max_range)
            if t is not None and t < best:
                best = t
        out.append(best)
    return out


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
