import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Vertices of the main area
vertices = [
    (0.0, 0.0),
    (7.7, 2.6),
    (12.8, 0.0),
    (17.9, 5.1),
    (20.4, 12.8),
    (17.9, 20.4),
    (12.8, 23.0),
    (7.7, 20.4),
    (2.6, 15.3),
    (0.0, 7.7)
]

#Forbidden zones (x, y, radius)
forbidden_zones = [
    (7.5, 10, 1.7),
    (7, 13.5, 2),
    (16.5, 13, 1.5),
    (10, 17.5, 1.5)
]

# Ellipses parameters (a, b)
ellipse_params = [
    (2.8, 1.4),
    (1.6, 3.1),
    (2.1, 4.3),
    (1.2, 2.4),
    (1.9, 3.7),
    (0.9, 1.9),
    (2.3, 4.1),
    (1.3, 2.6),
    (1.7, 3.4),
    (1.1, 2.2),
    (2.0, 3.9),
    (1.0, 2.0),
    (2.2, 4.4),
    (1.5, 2.7),
    (1.8, 3.5),
    (1.2, 2.3),
    (2.4, 4.0),
    (1.4, 2.5),
    (1.9, 3.6),
    (1.0, 2.1)
]

# Ellipses positioning (1D array: x1, y1, tau1, x2, y2, tau2, ...)
positions_1d = [
    4.60287213, 6.56948361, 1.39,
    16.40025067, 5.80806444, 5.32,
    7.42986517, 17.13576316, 5.25,
    3.45091372, 14.40095728, 2,
    3.04411913, 2.89474381, 2.7,
    17.32894582, 19.29402437, 1.05,
    10.12438042, 9.34606282, 2.9,
    13.16129433, 12.54019689, 0.03,
    12.05187157, 6.08922536, 2.86,
    14.89149996, 19.02671709, 0.5,
    15.77051954, 15.57448831, 2.9,
    13.24965242, 17.54692457, 2.67,
    17.11055134, 11.02177492, 2.25,
    11.47772851, 19.8462347, 1.8,
    9.73344362, 14.15171157, 2.2,
    0.9744613, 5.87553023, 1.5,
    4.0163689, 10.45795883, 2.5,
    7.52194973, 4.70302566, 1.8,
    11.7488999, 2.67996205, 0.1,
    14.32409305, 21.32373706, 0.6
]

# Reshape positions into (x, y, tau) tuples
ellipse_positions = [(positions_1d[i * 3], positions_1d[i * 3 + 1], positions_1d[i * 3 + 2])
                     for i in range(20)]

# Create plot
fig, ax = plt.subplots(figsize=(12, 12))

# Draw main area
polygon = patches.Polygon(vertices, closed=True,
                          facecolor='blue', edgecolor='navy',
                          linewidth=2, alpha=0.25)
ax.add_patch(polygon)

# Draw forbidden zones
for i, (x, y, r) in enumerate(forbidden_zones, 1):
    circle = patches.Circle((x, y), r,
                            facecolor='red', edgecolor='darkred',
                            linewidth=1.5, alpha=0.25)

    center = patches.Circle((x, y), 0.1, facecolor='red', edgecolor='darkred',
                   linewidth=1.5)
    ax.add_patch(circle)
    ax.add_patch(center)
    #ax.text(x, y, f'Z{i}', ha='center', va='center', fontweight='bold')

# Draw covering ellipses
for i, ((a, b), (x, y, tau)) in enumerate(zip(ellipse_params, ellipse_positions), 1):
    angle_deg_user = math.degrees(tau)
    angle_deg_matplotlib = 90 - angle_deg_user
    print(f'ellipps {i} angle: {angle_deg_user}')

    ellipse = patches.Ellipse((x, y), width=2 * a, height=2 * b, angle=angle_deg_matplotlib,
                              facecolor='green', edgecolor='darkgreen',
                              linewidth=1.5, alpha=0.25)
    center = patches.Circle((x, y), 0.1, facecolor='green', edgecolor='darkgreen',
                            linewidth=1.5)

    ax.add_patch(ellipse)
    ax.add_patch(center)
    #ax.text(x, y-0.5, f'{i}', ha='center', va='center', fontweight='bold', fontsize=7, color='white')

# Mark vertices
for i, (x, y) in enumerate(vertices, 1):
    ax.plot(x, y, 'o', color='navy', markersize=6)
    #ax.text(x, y - 0.5, f'V{i}', ha='center', va='top', color='navy', fontsize=8)

# Configure plot
ax.set_xlim(-2, 24)
ax.set_ylim(-2, 25)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_title('Main Area with Forbidden Zones and 20 Covering Ellipses')

plt.tight_layout()
plt.show()