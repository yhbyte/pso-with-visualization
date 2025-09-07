import numpy as np
import shapely
from shapely.geometry import Polygon, Point, box
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
import matplotlib.pyplot as plt


def rect(center, w, h, angle_deg=0.0):
    cx, cy = center
    g = box(-w/2, -h/2, w/2, h/2)
    if angle_deg:
        g = rotate(g, angle_deg, origin=(0, 0))
    return translate(g, cx, cy)


def iso_triangle(center, direction, length, width):
    cx, cy = center
    dx, dy = direction
    n = (dx*dx + dy*dy) ** 0.5 or 1.0
    dx, dy = dx/n, dy/n
    px, py = -dy, dx
    tip = (cx + 0.5*length*dx, cy + 0.5*length*dy)
    base_c = (cx - 0.5*length*dx, cy - 0.5*length*dy)
    base_l = (base_c[0] + 0.5*width*px, base_c[1] + 0.5*width*py)
    base_r = (base_c[0] - 0.5*width*px, base_c[1] - 0.5*width*py)
    return Polygon([tip, base_l, base_r])


def composite_region(smooth=0.03):
    body = rect(center=(0.30, 0.05), w=1.80, h=1.30, angle_deg=12)
    arrowhead = iso_triangle(center=(1.35, 0.05), direction=(1, 0), length=1.20, width=2.00)
    shoulder_t = rect(center=(-0.55, 0.85), w=1.20, h=0.80, angle_deg=30)
    region = unary_union([body, arrowhead, shoulder_t])
    scoop_l = iso_triangle(center=(-0.55, 0.10), direction=(-1, -0.25), length=1.00, width=0.85)
    notch_br = iso_triangle(center=(0.60, -0.60), direction=(1, -0.40), length=0.95, width=0.95)
    region = region.difference(unary_union([scoop_l, notch_br]))
    if smooth and smooth > 0:
        region = region.buffer(+smooth).buffer(-smooth)
    region = region.buffer(0)
    if hasattr(region, "geoms"):
        polys = [g for g in region.geoms if isinstance(g, Polygon)]
        if polys:
            region = max(polys, key=lambda g: g.area)
    return region


def f(x, y, geom=None):
    if geom is None:
        geom = composite_region()
    xv = np.atleast_1d(x).astype(float)
    yv = np.atleast_1d(y).astype(float)
    pts = shapely.points(xv, yv)
    dist = shapely.distance(geom.boundary, pts)
    inside = shapely.contains(geom, pts)
    sdist = np.where(inside, dist, -dist)
    return sdist.reshape(np.broadcast(xv, yv).shape) if (xv.size > 1 or yv.size > 1) else float(sdist)


def is_in_shape(x, y, geom=None):
    if geom is None:
        geom = composite_region()
    return bool(geom.covers(Point(float(x), float(y))))


def plot_region_and_levelsets(geom, n=520, padding=0.5):
    minx, miny, maxx, maxy = geom.bounds
    minx -= padding; miny -= padding; maxx += padding; maxy += padding
    xs = np.linspace(minx, maxx, n)
    ys = np.linspace(miny, maxy, n)
    GX, GY = np.meshgrid(xs, ys)
    pts = shapely.points(GX.ravel(), GY.ravel())
    dist = shapely.distance(geom.boundary, pts)
    inside = shapely.contains(geom, pts)
    SD = np.where(inside, dist, -dist).reshape(GX.shape)
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    cf = ax.contourf(GX, GY, SD, levels=50, alpha=0.95)
    ax.contour(GX, GY, SD, levels=[0.0], linewidths=2)
    x, y = geom.exterior.xy
    ax.plot(x, y, linewidth=2)
    for ring in geom.interiors:
        xi, yi = ring.xy
        ax.plot(xi, yi, linewidth=1, linestyle="--")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Arrow-ish region from a few BIG pieces: f(x,y)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(cf, ax=ax, label="f(x,y)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    region = composite_region(smooth=0.03)
    tests = [(-0.6, 0.8), (1.75, 0.1), (1.95, 0.0), (0.6, -0.6), (-0.55, 0.10)]
    for pt in tests:
        print(f"{pt} ->", "inside" if is_in_shape(*pt, geom=region) else "outside")
    plot_region_and_levelsets(region, n=520)
