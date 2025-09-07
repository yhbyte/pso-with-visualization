import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Vertices of the main area
vertices = [
    (0.00, 0.00),
    (4.47, 1.49),
    (7.45, 0.00),
    (10.43, 2.98),
    (11.92, 7.45),
    (10.43, 11.92),
    (7.45, 13.41),
    (4.47, 11.92),
    (1.49, 8.94),
    (0.00, 4.47)
]

# Forbidden zones (x, y, radius)
forbidden_zones = [
    (2.98, 2.98, 1.49),
    (11.92, 11.92, 2.23),
    (7.45, 4.47, 1.79),
    (10.43, 7.45, 1.19),
    (5.96, 8.94, 1.79)
]

# Covering ellipses (a, b, x, y, theta)
ellipses = [
    (3.0, 1.5, 1.34, 2.45, 0.89),
    (2.5, 1.0, 3.56, 1.67, 1.34),
    (4.0, 2.0, 5.78, 3.89, 2.56),
    (3.5, 1.2, 7.90, 5.01, 3.25),
    (2.0, 0.8, 9.12, 6.23, 4.67),
    (4.5, 2.5, 10.34, 7.45, 5.78),
    (3.2, 1.8, 8.56, 8.67, 0.23),
    (2.8, 1.4, 6.89, 9.89, 1.45),
    (3.8, 2.2, 5.01, 11.01, 2.67),
    (4.2, 1.6, 2.23, 12.23, 3.89)
]

# Create plot
fig, ax = plt.subplots(figsize=(10, 10))

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
    ax.add_patch(circle)
    ax.text(x, y, f'Z{i}', ha='center', va='center', fontweight='bold')

# Draw covering ellipses
for i, (a, b, x, y, theta) in enumerate(ellipses, 1):
    # Convert angle from radians to degrees
    angle_deg = np.degrees(theta)
    ellipse = patches.Ellipse((x, y), width=2*a, height=2*b, angle=angle_deg,
                             facecolor='green', edgecolor='darkgreen',
                             linewidth=1.5, alpha=0.25)
    ax.add_patch(ellipse)
    ax.text(x, y, f'E{i}', ha='center', va='center', fontweight='bold', fontsize=8)

# Mark vertices
for i, (x, y) in enumerate(vertices, 1):
    ax.plot(x, y, 'o', color='navy', markersize=6)
    ax.text(x, y - 0.3, f'V{i}', ha='center', va='top', color='navy')

# Configure plot
ax.set_xlim(-1, 14)
ax.set_ylim(-1, 15)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_title('Main Area with Forbidden Zones and Covering Ellipses')

plt.tight_layout()
plt.show()