import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse

parser = argparse.ArgumentParser(description='Plot tracking evaluation results')
parser.add_argument('results_file', nargs=1, help='tracking results file')
args = parser.parse_args()

with h5py.File(args.results_file[0], 'r') as f:
    failures = f['failures'][:]
    radial_errors = f['radial_errors'][:]
    length_errors = f['length_errors'][:]
    track_counts = f['track_counts'][:]
    track_lengths = f['track_lengths'][:]
    attrs = dict(f['attributes'].attrs.items())

fig = plt.figure()
fig.suptitle('KLT Tracking - chi={}, alpha={}, focal_length={}'.format(attrs["chi"][0], attrs["alpha"][0], attrs["fx"][0]))
ax1 = fig.add_subplot(5, 1, 1)
ax2 = fig.add_subplot(5, 1, 2)
ax3 = fig.add_subplot(5, 1, 3)
ax4 = fig.add_subplot(5, 1, 4)
ax5 = fig.add_subplot(5, 1, 5)

binned_errors = [[0] for i in range(10)]
for r, e in radial_errors:
    if e < 50:
        binned_errors[int(min(r, 0.499999) / 0.05)].append(e)
ax1.set_title('Pixel error distribution for various radial distances')
ax1.set_ylabel('Pixel error')
ax1.set_xlabel('Radial distance')
ax1.set_xticks(np.arange(1, 11))
ax1.set_xticklabels(['{}-{}'.format(i * 0.05, (i + 1) * 0.05) for i in range(10)])
ax1.set_ylim([0, 20])
ax1.violinplot(binned_errors)
ax1.plot([i for i in range(1, 11)], [np.array(binned_errors[i]).mean() for i in range(10)], '-o', markersize=4, c='black')

ax2.set_title('Pixel errors over track lifetime')
ax2.set_ylim([0, 30])
ax2.set_ylabel('Pixel error')
ax2.set_xlabel('Frame')
err = dict()
rs = dict()
num_tracks = 0
for row in length_errors:
    if row[1] not in err:
        err[row[1]] = []
    if row[1] not in rs:
        rs[row[1]] = []
    err[row[1]].append((row[0], row[3]))
    rs[row[1]].append((row[2]))
    if row[1] > num_tracks:
        num_tracks = int(row[1])
i = 0
while i < num_tracks:
    if i not in err or i not in rs:
        i += 1
        continue
    e = err[i]
    r = np.asarray(rs[i])
    colors = cm.rainbow(r * 2)
    if len(e) > 0:
        ax2.scatter(*zip(*e), s=2, color=colors.reshape(-1, 4))
        for j in range(0, len(e) - 1):
            ax2.plot([e[j][0], e[j+1][0]], [e[j][1], e[j+1][1]], color=np.squeeze(colors[j]))
    i += int(num_tracks / 2000)
tl_err = [(tl, e) for tl, _, _, e in length_errors]
lim = int(max(tl_err, key=lambda item:item[0])[0])
binned_errors = [[] for i in range(lim + 1)]
for tl, _, _, e in length_errors:
    binned_errors[int(tl)].append(e)
ax2.plot([i for i in range(0, lim)], [np.array(binned_errors[i]).mean() for i in range(0, lim)], '-o', markersize=4, c='black')

ax3.set_title('Number of tracks over frames')
ax3.set_xlabel('Frame')
ax3.set_ylabel('Count')
frames = np.array([f for f, _ in track_counts])
counts = np.array([c for _, c in track_counts])
ax3.plot(frames, counts, color='blue', linewidth=2)

ax4.set_title('Distribution of track lifetimes')
ax4.set_xlabel('Lifetime (frames)')
ax4.set_ylabel('Count')
track_lengths = track_lengths[track_lengths > 0]
ax4.hist(track_lengths.ravel(), color='c', bins=int(track_lengths.max() / 10))
ax4.axvline(track_lengths.mean(), color='k', linestyle='dashed', linewidth=1)
ax4.axvline(np.median(track_lengths), color='k', linestyle='dotted', linewidth=1)

ax5.set_title('Number of failures at various radial distances')
ax5.set_xlabel('Radial distance')
ax5.set_ylabel('Count')
ax5.hist([r for r in failures], bins=[0.0125 * a for a in range(0, 41)], color='palegreen')
plt.show()
