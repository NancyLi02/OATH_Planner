import matplotlib.pyplot as plt
import numpy as np


fig, ax1 = plt.subplots()

labels = ("10", "20", "50", "100")
label_means = {
    'LTL-D*': (1440, 3340, 8490, 16840),
    'Local Revise': (1580, 3820, 10120, 20300),
}

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars
multiplier = -0.5


for attribute, measurement in label_means.items():
    offset = width * multiplier
    rects = ax1.bar(x + offset, measurement, width, label=attribute, alpha=0.5)
    # ax.bar_label(rects, padding=2)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_ylabel('Total Cost')
# ax.set_title('Penguin attributes by species')
ax1.set_xticks(x + width, labels)
ax1.yaxis.set_ticks(np.arange(0, 22500, 2500))
ax1.legend(loc='upper left')
#ax.set_ylim(0, 250)


ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Ratio', color=color)  # we already handled the x-label with ax1
ax2.plot(labels, [1.09, 1.11, 1.189, 1.193], color=color, linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)
# ax2.set_ylim(1, 1.2)
# ax2.yaxis.set_ticks(np.arange(1, 1.2, 0.05))
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()