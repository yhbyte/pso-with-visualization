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
    (2.8, 1.4, 1.23, 2.34, 0.78),
    (3.1, 1.6, 3.45, 1.56, 1.23),
    (4.3, 2.1, 5.67, 3.78, 2.45),
    (2.4, 1.2, 7.89, 4.90, 3.14),
    (3.7, 1.9, 9.01, 6.12, 4.56),
    (1.9, 0.9, 11.23, 7.34, 5.67),
    (4.1, 2.3, 8.45, 8.56, 0.12),
    (2.6, 1.3, 6.67, 9.78, 1.34),
    (3.4, 1.7, 4.89, 10.90, 2.56),
    (2.2, 1.1, 3.01, 12.12, 3.78),
    (3.9, 2.0, 1.12, 3.23, 0.90),
    (2.0, 1.0, 3.34, 2.45, 1.45),
    (4.4, 2.2, 5.56, 4.67, 2.67),
    (2.7, 1.5, 7.78, 5.89, 3.89),
    (3.5, 1.8, 9.90, 7.01, 5.01),
    (2.3, 1.2, 11.12, 8.23, 0.12),
    (4.0, 2.4, 8.34, 9.45, 1.56),
    (2.5, 1.4, 6.56, 10.67, 2.78),
    (3.6, 1.9, 4.78, 11.89, 3.90),
    (2.1, 1.0, 2.90, 13.01, 4.12)
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
    ax.text(x, y, f'{i}', ha='center', va='center', fontweight='bold', fontsize=7, color='white')

# Mark vertices
for i, (x, y) in enumerate(vertices, 1):
    ax.plot(x, y, 'o', color='navy', markersize=6)
    ax.text(x, y - 0.3, f'V{i}', ha='center', va='top', color='navy', fontsize=8)

# Configure plot
ax.set_xlim(-1, 14)
ax.set_ylim(-1, 15)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_title('Main Area with Forbidden Zones and 20 Covering Ellipses')

plt.tight_layout()
plt.show()