import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse

parser = argparse.ArgumentParser(description='Plot matching evaluation results')
parser.add_argument('results_file', nargs=1, help='matching results file')
args = parser.parse_args()

with h5py.File(args.results_file[0], 'r') as f:
    stats = f['match_stats'][:]
    radial_overlaps_errors = f['radial_overlaps_errors'][:]
    delta_radius = f['delta_radius'][:]
    attrs = dict(f['attributes'].attrs.items())

df = pd.DataFrame(stats)
stats = df.groupby(0).mean().to_records()
stats = stats.view(np.float64).reshape(len(stats), -1)
framediff, nmatch, prec, rec = stats.T

fig = plt.figure()
fig.suptitle('Matching - detector={}, descriptor={}, chi={}, alpha={}, focal_length={}'.format(attrs["detector_type"], attrs["descriptor_type"], attrs["chi"][0], attrs["alpha"][0], attrs["fx"][0]))
ax1 = fig.add_subplot(4, 1, 1)
ax2 = fig.add_subplot(4, 1, 2)
ax3 = fig.add_subplot(4, 1, 3)
ax4 = fig.add_subplot(4, 1, 4)
cm = plt.get_cmap('gist_ncar')
numdesc = 1

ax1.set_prop_cycle(color=[cm(1. * i / numdesc) for i in range(numdesc)])

handles = []
color = next(ax1._get_lines.prop_cycler)['color']
h1, = ax1.plot(framediff * attrs["rate"][0], prec, color=color)
h2, = ax1.plot(framediff * attrs["rate"][0], rec, linestyle='dashed', color=color)
handles.append((h1, h2))

l1 = ax1.legend(handles, ['{}+{}'.format(attrs["detector_type"], attrs["descriptor_type"])], loc=1, title='Detector+Descriptor', fontsize='small')
l2 = ax1.legend([h1, h2], ['Precision', 'Recall'], loc=4, fontsize='small')
l2.legendHandles[0].set_color('black')
l2.legendHandles[1].set_color('black')
ax1.add_artist(l1)
ax1.set_xlabel('Frame difference')
ax1.set_ylabel('Precision / recall')
ax1.set_title('Precision-recall over frame difference')

handles = []
ax2.set_prop_cycle(color=[cm(1. * i / numdesc) for i in range(numdesc)])
color = next(ax2._get_lines.prop_cycler)['color']
ax2.plot(framediff * attrs["rate"][0], nmatch, color=color)
handles.append((h1, h2))

l1 = ax2.legend(handles, ['{}+{}'.format(attrs["detector_type"], attrs["descriptor_type"])], loc=1, title='Detector+Descriptor', fontsize='small')
ax2.add_artist(l1)
ax2.set_xlabel('Frame difference')
ax2.set_ylabel('Number of matches')
ax2.set_title('Number of matches over frame difference')

ax3.set_title('Distribution of matches over changes in radial distance')
ax3.set_ylabel('Number of matches')
ax3.set_xlabel('Delta Radial distance')
ax3.hist(delta_radius.ravel(), color='c')

binned_overlaps = [[0] for i in range(10)]
for r, o, _ in radial_overlaps_errors:
    binned_overlaps[int(min(r, 0.499999) / 0.05)].append(o)
ax4.set_title('Match IOU distribution for various radial distances')
ax4.set_ylabel('IOU')
ax4.set_xlabel('Radial distance')
ax4.set_xticks(np.arange(1, 11))
ax4.set_xticklabels(['{}-{}'.format(i * 0.05, (i + 1) * 0.05) for i in range(10)])
ax4.violinplot(binned_overlaps)
ax4.plot([i for i in range(1, 11)], [np.array(binned_overlaps[i]).mean() for i in range(10)], '-o', markersize=4, c='black')

plt.show()
