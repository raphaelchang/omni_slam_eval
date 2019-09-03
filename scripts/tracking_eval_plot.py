import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from parse import parse
import argparse

parser = argparse.ArgumentParser(description='Plot tracking evaluation results')
parser.add_argument('results_path', help='tracking results file or working directory')
args = parser.parse_args()

if not os.path.isdir(args.results_path) and args.results_path.endswith('.hdf5'):
    with h5py.File(args.results_path, 'r') as f:
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

elif os.path.isdir(args.results_path):
    fig = plt.figure()
    fig.suptitle('KLT Tracking Performance for Various FOVs and Motions')
    motion_count = 0
    fov_dict = dict()
    last_fov_num = 0
    for motion in os.listdir(args.results_path):
        if os.path.isdir(os.path.join(args.results_path, motion)):
            bag_dir = os.path.join(args.results_path, motion)
            motion_count += 1
            for fov in os.listdir(bag_dir):
                if os.path.isdir(os.path.join(bag_dir, fov)):
                    for filename in os.listdir(os.path.join(bag_dir, fov)):
                        if filename.endswith('.tracking.hdf5'):
                            results_file = os.path.join(bag_dir, fov, filename)
                            with h5py.File(results_file, 'r') as f:
                                attrs = dict(f['attributes'].attrs.items())
                                fov = int(round(attrs['fov']))
                                if fov not in fov_dict.keys():
                                    fov_dict[fov] = last_fov_num
                                    last_fov_num += 1
                                break

    last_fov_num = 0
    for fov in sorted(fov_dict.keys()):
        fov_dict[fov] = last_fov_num
        last_fov_num += 1

    num_rows = motion_count
    num_cols = last_fov_num
    motion_inx = 0
    for motion in os.listdir(args.results_path):
        if os.path.isdir(os.path.join(args.results_path, motion)):
            bag_dir = os.path.join(args.results_path, motion)
            for fov in os.listdir(bag_dir):
                if os.path.isdir(os.path.join(bag_dir, fov)):
                    chi, alpha, fx, fy, cx, cy = parse('chi{:f}_alpha{:f}_fx{:f}_fy{:f}_cx{:f}_cy{:f}', fov)
                    failures = np.empty(shape=(1,0))
                    radial_errors = np.empty(shape=(0,2))
                    file_exists = False
                    for filename in os.listdir(os.path.join(bag_dir, fov)):
                        if filename.endswith('.tracking.hdf5'):
                            results_file = os.path.join(bag_dir, fov, filename)
                            with h5py.File(results_file, 'r') as f:
                                failures = np.hstack((failures, f['failures'][:]))
                                radial_errors = np.vstack((radial_errors, f['radial_errors'][:]))
                                attrs = dict(f['attributes'].attrs.items())
                                file_exists = True
                                fov = int(round(attrs['fov']))
                    if file_exists:
                        ax = fig.add_subplot(num_rows, num_cols, motion_inx * num_cols + fov_dict[fov] + 1)
                        ax.hist([[r for r, e in radial_errors if e <= 5], [r for r, e in radial_errors if 5 < e <= 20], [r for r, e in radial_errors if 20 < e <= 50], [r for r in failures]], bins=[i * 0.05 for i in range(11)], alpha=0.5, label=['<5', '5-20', '20-50', 'Failures'], stacked=False)
                        ax.legend(loc='best', title='Pixel error', fontsize='x-small')
                    else:
                        print '[WARNING] No results files found in directory {}'.format(os.path.join(bag_dir, fov))

                    if motion_inx == 0 and file_exists:
                        ax.set_title('FOV {} degrees'.format(fov))
                    elif motion_inx == num_rows - 1:
                        ax.set_xlabel('Radial distance')
                    if fov_dict[fov] == 0:
                        ax.set_ylabel(motion, size='large')
                    elif fov_dict[fov] == num_cols - 1:
                        ax.set_ylabel('Number of tracks')
                        ax.yaxis.set_label_position("right")
            motion_inx += 1
    plt.show()

else:
    print "[ERROR] Invalid path specified"
