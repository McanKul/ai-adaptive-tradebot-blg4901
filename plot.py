import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Data derived from the HTML file
tasks = [
    "Project Planning & Research",
    "System Design",
    "Core Development",
    "Integration & Testing",
    "Dashboard & Deployment",
    "Doc & Final Presentation"
]

# Time range: Oct 2025 - Apr 2026
months = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]

# Matrix representing which months contain work (1 = active, 0 = inactive)
# Based on the HTML dates:
# 1. Planning: Oct 01 - Oct 22 -> Oct
# 2. Design: Oct 22 - Nov 18 -> Oct, Nov
# 3. Core Dev: Nov 19 - Feb 09 -> Nov, Dec, Jan, Feb
# 4. Integration: Feb 10 - Mar 09 -> Feb, Mar
# 5. Dashboard: Mar 10 - Apr 06 -> Mar, Apr
# 6. Documentation: Apr 07 - Apr 24 -> Apr
data = [
    [1, 0, 0, 0, 0, 0, 0], # Planning
    [1, 1, 0, 0, 0, 0, 0], # Design
    [0, 1, 1, 1, 1, 0, 0], # Core Dev
    [0, 0, 0, 0, 1, 1, 0], # Integration
    [0, 0, 0, 0, 0, 1, 1], # Dashboard
    [0, 0, 0, 0, 0, 0, 1], # Documentation
]

# Reference colors (Blue and Yellow/Orange)
colors = ['#5b9bd5', '#ffc000']

# Setup Figure
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, len(months) + 3.5) # Extra space for "Work Package" column
ax.set_ylim(0, len(tasks) + 1)

# Function to draw the grid
def draw_table():
    # Draw Headers
    ax.text(0.1, len(tasks) + 0.3, "Work Package", weight='bold', fontsize=12, va='center')
    for i, month in enumerate(months):
        ax.text(i + 3.5 + 0.5, len(tasks) + 0.3, month, weight='bold', fontsize=12, ha='center', va='center')
        
        # Draw Header Grid Box
        rect = Rectangle((i + 3.5, len(tasks)), 1, 1, fill=False, edgecolor='black')
        ax.add_patch(rect)
        
    # Draw "Work Package" Header Box
    rect = Rectangle((0, len(tasks)), 3.5, 1, fill=False, edgecolor='black')
    ax.add_patch(rect)

    # Draw Rows
    for row_idx, task in enumerate(tasks):
        y_pos = len(tasks) - 1 - row_idx
        
        # Draw Task Name
        ax.text(0.1, y_pos + 0.5, task, fontsize=11, va='center')
        
        # Draw Task Name Box boundary
        rect = Rectangle((0, y_pos), 3.5, 1, fill=False, edgecolor='black')
        ax.add_patch(rect)

        # Draw Month Cells
        current_color = colors[row_idx % 2] # Alternating colors
        
        for col_idx, is_active in enumerate(data[row_idx]):
            x_pos = col_idx + 3.5
            
            # Fill color if active, otherwise white
            fill_color = current_color if is_active else "white"
            
            rect = Rectangle((x_pos, y_pos), 1, 1, facecolor=fill_color, edgecolor='black')
            ax.add_patch(rect)

draw_table()

# Clean up axes
ax.axis('off')
plt.tight_layout()
plt.savefig('./docs/formal_gantt_chart.png', dpi=300, bbox_inches='tight')
plt.show()