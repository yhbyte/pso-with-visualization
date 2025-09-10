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

# Forbidden zones (x, y, radius)
forbidden_zones = [
    (7.5, 10, 1.7),
    (7, 13.5, 2),
    (16.5, 13, 1.5),
    (10, 17.5, 1.5),
    (14, 9.5, 1.7),
    (10, 5, 1.5)
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

# Remove default axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Draw custom axes starting from (0, 0)
# X-axis from 0 to max x
ax.arrow(0, 0, 22, 0, head_width=0.3, head_length=0.3, fc='black', ec='black', linewidth=1.5)
# Y-axis from 0 to max y
ax.arrow(0, 0, 0, 22.5, head_width=0.3, head_length=0.3, fc='black', ec='black', linewidth=1.5)

# Add axis labels
ax.text(23, -0.5, 'x', fontsize=18, ha='center')
ax.text(-0.5, 23, 'y', fontsize=18, ha='center')

# Add origin label
ax.text(-0.5, -0.5, '0', fontsize=18, ha='center')

# Add tick marks and numbers on x-axis
for x in range(5, 21, 5):
    ax.plot([x, x], [-0.2, 0.2], 'k-', linewidth=1)  # tick mark
    ax.text(x, -0.8, str(x), fontsize=18, ha='center')

# Add tick marks and numbers on y-axis
for y in range(5, 21, 5):
    ax.plot([-0.2, 0.2], [y, y], 'k-', linewidth=1)  # tick mark
    ax.text(-0.8, y, str(y), fontsize=18, ha='center', va='center')

# Draw main area in YELLOW (this will be the uncovered area)
polygon_yellow = patches.Polygon(vertices, closed=True,
                                 facecolor='yellow', edgecolor='navy',
                                 linewidth=2, alpha=0.5)
ax.add_patch(polygon_yellow)

# Draw all ellipse fills (white) first
for i, ((a, b), (x, y, tau)) in enumerate(zip(ellipse_params, ellipse_positions), 1):
    angle_deg_user = math.degrees(tau)
    angle_deg_matplotlib = 90 - angle_deg_user

    # Draw only the fill, no border
    ellipse_fill = patches.Ellipse((x, y), width=2 * a, height=2 * b, angle=angle_deg_matplotlib,
                                   facecolor='white', edgecolor='none',
                                   linewidth=0)
    ax.add_patch(ellipse_fill)

# Draw all ellipse borders on top (so they're not covered)
for i, ((a, b), (x, y, tau)) in enumerate(zip(ellipse_params, ellipse_positions), 1):
    angle_deg_user = math.degrees(tau)
    angle_deg_matplotlib = 90 - angle_deg_user

    # Draw only the border, no fill
    ellipse_border = patches.Ellipse((x, y), width=2 * a, height=2 * b, angle=angle_deg_matplotlib,
                                     facecolor='none', edgecolor='darkgreen',
                                     linewidth=1.5)
    ax.add_patch(ellipse_border)

    # Draw center point
    center = patches.Circle((x, y), 0.1, facecolor='green', edgecolor='darkgreen',
                            linewidth=1.5)
    ax.add_patch(center)

# Draw main area outline in blue with transparency
polygon_outline = patches.Polygon(vertices, closed=True,
                                  facecolor='none', edgecolor='blue',
                                  linewidth=3, alpha=0.7)
ax.add_patch(polygon_outline)

# Draw forbidden zones
for i, (x, y, r) in enumerate(forbidden_zones, 1):
    circle = patches.Circle((x, y), r,
                            facecolor='red', edgecolor='darkred',
                            linewidth=1.5, alpha=0.5)

    center = patches.Circle((x, y), 0.1, facecolor='red', edgecolor='darkred',
                            linewidth=1.5)
    ax.add_patch(circle)
    ax.add_patch(center)

# Mark vertices
for i, (x, y) in enumerate(vertices, 1):
    ax.plot(x, y, 'o', color='navy', markersize=6)

# Configure plot
ax.set_xlim(-2, 24)
ax.set_ylim(-2, 24)
ax.set_aspect('equal')
# Remove grid - no grid() call
# Remove tick marks
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
plt.show()