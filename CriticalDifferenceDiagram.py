
import matplotlib.pyplot as plt
import numpy as np

# Data from Nemenyi test
algorithms = ["WORL_RTGS", "DRAS", "mDRL", "MAGRL", "MIGMPS", "RL_Hybrid"]
avg_ranks = [1.485, 1.98, 2.92, 4.0, 4.7, 5.9]
colors = ["#ff7f0e", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
critical_difference = 1.260  # CD value from Nemenyi test (alpha=0.05)

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 4))
y_pos = np.arange(len(algorithms))

# Plot horizontal bars
bars = ax.barh(y_pos, avg_ranks, color=colors, edgecolor=colors, height=0.5, label=algorithms)

# Customize plot
ax.set_yticks(y_pos)
ax.set_yticklabels(algorithms, fontsize=10)
ax.invert_yaxis()  # Highest rank at top
ax.set_xlabel("Average Rank", fontsize=12)  # Adapted from 'Nodes number'
ax.set_ylabel("Algorithm", fontsize=12)  # Adapted from 'Makespan'
#ax.set_title("Critical Difference Diagram", fontsize=12)
ax.set_xlim(1, 6)
ax.set_xticks(np.arange(1, 6.1, 0.5))  # Adjusted ticks for ranks
ax.grid(True, axis='x', linestyle='--', alpha=0.7)


# Adjust layout and display
plt.tight_layout()
plt.show()

