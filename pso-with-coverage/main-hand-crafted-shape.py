import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely.affinity import rotate, translate, scale
from shapely.geometry import Polygon, Point


def shape_polygon(
    rotation_deg: float = 0.0,
    scale_xy: tuple = (1.0, 1.0),
    offset_xy: tuple = (0.0, 0.0),
    smooth: float = 0.0,
):
    P = np.array([
        (-2.10,  1.05),
        (-1.05,  2.35),
        (-0.20,  0.95),
        ( 1.45,  0.55),
        ( 2.80,  1.00),
        ( 3.00,  0.00),
        ( 1.70, -1.85),
        ( 0.55, -0.45),
        (-0.30, -2.70),
        (-0.40, -0.75),
        (-1.60,  0.30),
    ], dtype=float)

    poly = Polygon(P).buffer(0)
    if rotation_deg:
        poly = rotate(poly, rotation_deg, origin="center")
    if scale_xy != (1.0, 1.0):
        poly = scale(poly, xfact=scale_xy[0], yfact=scale_xy[1], origin="center")
    if offset_xy != (0.0, 0.0):
        poly = translate(poly, xoff=offset_xy[0], yoff=offset_xy[1])
    if smooth and smooth > 0:
        poly = poly.buffer(+smooth).buffer(-smooth)
    return poly


def f(x, y, geom=None):
    if geom is None:
        geom = shape_polygon()
    xv = np.atleast_1d(x).astype(float)
    yv = np.atleast_1d(y).astype(float)
    pts = shapely.points(xv, yv)

    # Distance to boundary and inside test (Shapely 2 array API)
    dist = shapely.distance(geom.boundary, pts)
    inside = shapely.contains(geom, pts)
    sdist = np.where(inside, dist, -dist)
    return sdist.reshape(np.broadcast(xv, yv).shape) if (xv.size > 1 or yv.size > 1) else float(sdist)

def is_in_shape(x, y, geom=None):
    if geom is None:
        geom = shape_polygon()
    return bool(geom.covers(Point(float(x), float(y))))

# ----------------- Visualization -----------------

def plot_region_and_levelsets(geom, n=500, padding=0.6):
    minx, miny, maxx, maxy = geom.bounds
    minx -= padding; miny -= padding; maxx += padding; maxy += padding

    xs = np.linspace(minx, maxx, n)
    ys = np.linspace(miny, maxy, n)
    GX, GY = np.meshgrid(xs, ys)

    # Signed distance field
    pts = shapely.points(GX.ravel(), GY.ravel())
    dist = shapely.distance(geom.boundary, pts)
    inside = shapely.contains(geom, pts)
    SD = np.where(inside, dist, -dist).reshape(GX.shape)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    cf = ax.contourf(GX, GY, SD, levels=50, alpha=0.95)
    ax.contour(GX, GY, SD, levels=[0.0], linewidths=2)  # boundary

    # Draw polygon edges explicitly
    x, y = geom.exterior.xy
    ax.plot(x, y, linewidth=2)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Single non-convex region: f(x,y) signed distance")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(cf, ax=ax, label="f(x,y)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    region = shape_polygon(rotation_deg=0, scale_xy=(1.0, 1.0), offset_xy=(0.0, 0.0), smooth=0.0)

    tests = [(-2.0, 1.0), (2.6, 0.6), (3.2, 0.0), (0.2, -1.0)]
    for pt in tests:
        print(pt, "->", "inside" if is_in_shape(*pt, geom=region) else "outside")

    plot_region_and_levelsets(region, n=520)
