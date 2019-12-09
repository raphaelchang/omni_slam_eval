import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas
import os
from parse import parse
import argparse

parser = argparse.ArgumentParser(description='Plot tracking evaluation results')
parser.add_argument('results_path', help='tracking results file or working directory')
args = parser.parse_args()

if not os.path.isdir(args.results_path) and args.results_path.endswith('.hdf5'):
    with h5py.File(args.results_path, 'r') as f:
        failures = f['failures'][:]
        successes = f['successes'][:]
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

    df = pandas.DataFrame()
    re_filt = radial_errors[np.where((radial_errors[:, 1] < 50) & (radial_errors[:, 2] > 0))]
    df = df.append(pandas.DataFrame({'Radial distance': np.round(re_filt[:, 0] / 0.005) * 0.005, 'Relative endpoint error (pixels)': re_filt[:, 2]}))
    # binned_errors = [[0] for i in range(30)]
    # for r, aee, ree, ae, rbe in radial_errors:
        # if aee < 50:
            # binned_errors[int(r / 0.025)].append(aee)
    # ax1.set_title('Pixel error distribution for various radial distances')
    # ax1.set_ylabel('Pixel error')
    # ax1.set_xlabel('Radial distance')
    # ax1.set_xticks(np.arange(1, 22))
    # ax1.set_xticklabels(['{}-{}'.format(i * 0.025, (i + 1) * 0.025) for i in range(21)])
    # ax1.set_ylim([0, 10])
    # ax1.violinplot(binned_errors[:21])
    # ax1.plot([i for i in range(1, 22)], [np.array(binned_errors[i]).mean() for i in range(21)], '-o', markersize=4, c='black')
    sns.relplot(x="Radial distance", y="Relative endpoint error (pixels)", kind="line", data=df, ci="sd", ax=ax1)

    ax2.set_title('Pixel errors over track lifetime')
    ax2.set_ylim([0, 0.1])
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
        err[row[1]].append((row[0], row[4]))
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
    tl_err = [(tl, be) for tl, _, _, e, be in length_errors]
    lim = int(max(tl_err, key=lambda item:item[0])[0])
    binned_errors = [[] for i in range(lim + 1)]
    for tl, _, _, e, be in length_errors:
        binned_errors[int(tl)].append(be)
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

    ax5.set_title('Ratio of inliers over radial distances')
    ax5.set_xlabel('Radial distance')
    ax5.set_ylabel('Ratio of inliers')
    lim = int(successes.max(axis=0)[1])
    binned_inliers = [[0 for j in range(50)] for i in range(lim)]
    binned_outliers = [[0 for j in range(50)] for i in range(lim)]
    for r, f in successes:
        binned_inliers[int(f) - 1][int(r / 0.0125)] += 1
    for r, f in failures:
        binned_outliers[int(f) - 1][int(r / 0.0125)] += 1
    binned_ratio = [0. for i in range(lim)]
    for r in range(50):
        for f in range(lim):
            total = binned_outliers[f][r] + binned_inliers[f][r]
            if total > 0:
                binned_ratio[r] += binned_inliers[f][r] / float(total)
        binned_ratio[r] /= lim
    # ax5.hist([r for r, _ in failures], bins=[0.0125 * a for a in range(0, 43)], color='palegreen')
    ax5.bar([0.0125 * a for a in range(0, 43)], [binned_ratio[a] for a in range(43)], 0.0125)
    plt.show()

elif os.path.isdir(args.results_path):
    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    fig3 = plt.figure(3)
    fig1.suptitle('KLT Tracking Accuracy for Various FOVs and Motions')
    fig2.suptitle('KLT Tracking Lifetime Distributions for Various FOVs and Motions')
    fig3.suptitle('KLT Tracking Track Counts for Varying Rates')
    sns.set()
    motion_count = 0
    fov_dict = dict()
    last_fov_num = 0
    fovs = []
    for yaml in os.listdir(args.results_path):
        if not os.path.isdir(os.path.join(args.results_path, yaml)) and yaml.endswith('.yaml'):
            fov = os.path.splitext(os.path.basename(yaml))[0]
            fovs.append(fov)
    fovs.sort(key=int)
    for motion in os.listdir(args.results_path):
        if os.path.isdir(os.path.join(args.results_path, motion)):
            bag_dir = os.path.join(args.results_path, motion)
            motion_exists = False
            for fovstr in fovs:
                for filename in os.listdir(bag_dir):
                    if filename.split('.')[1] == fovstr and filename.endswith('.tracking.hdf5'):
                        results_file = os.path.join(bag_dir, filename)
                        with h5py.File(results_file, 'r') as f:
                            attrs = dict(f['attributes'].attrs.items())
                            # fov = int(round(attrs['fov']))
                            fov = int(fovstr)
                            if fov not in fov_dict.keys():
                                fov_dict[fov] = last_fov_num
                                last_fov_num += 1
                            motion_exists = True
                            break
            if motion_exists:
                motion_count += 1

    last_fov_num = 0
    for fov in sorted(fov_dict.keys()):
        fov_dict[fov] = last_fov_num
        last_fov_num += 1

    num_rows = motion_count
    num_cols = last_fov_num
    motion_inx = 0
    df_ee = pandas.DataFrame()
    df_ae = pandas.DataFrame()
    df_be = pandas.DataFrame()
    df_inlier = pandas.DataFrame()
    df_length = pandas.DataFrame()
    for motion in os.listdir(args.results_path):
        if os.path.isdir(os.path.join(args.results_path, motion)):
            df_lifetime = pandas.DataFrame()
            df_rate = pandas.DataFrame()
            df_rate_errors = pandas.DataFrame()
            bag_dir = os.path.join(args.results_path, motion)
            motion_exists = False
            num_fovs = 0
            for fovstr in fovs:
                failures = dict()
                successes = dict()
                radial_errors_dict = dict()
                track_lengths = np.empty(shape=(1,0))
                length_errors = np.empty(shape=(0,5))
                file_exists = False
                for filename in os.listdir(bag_dir):
                    if filename.split('.')[1] == fovstr and filename.endswith('.tracking.hdf5'):
                        results_file = os.path.join(bag_dir, filename)
                        with h5py.File(results_file, 'r') as f:
                            attrs = dict(f['attributes'].attrs.items())
                            rate = int(attrs['rate'])
                            if rate not in successes:
                                successes[rate] = np.empty(shape=(0,2))
                            if rate not in failures:
                                failures[rate] = np.empty(shape=(0,2))
                            if rate not in radial_errors_dict:
                                radial_errors_dict[rate] = np.empty(shape=(0,5))
                            successes[rate] = np.vstack((successes[rate], f['successes'][:]))
                            failures[rate] = np.vstack((failures[rate], f['failures'][:]))
                            radial_errors_dict[rate] = np.vstack((radial_errors_dict[rate], f['radial_errors'][:]))
                            if rate == 1:
                                tl = f['track_lengths'][:]
                                track_lengths = np.hstack((track_lengths, tl[:, tl[0, :] > 0]))
                                length_errors = np.vstack((length_errors, f['length_errors'][:]))
                            file_exists = True
                            motion_exists = True
                            # fov = int(round(attrs['fov']))
                fov = int(fovstr)
                if file_exists:
                    radial_errors = radial_errors_dict[1]
                    ax = fig1.add_subplot(num_rows, num_cols, motion_inx * num_cols + fov_dict[fov] + 1)
                    ax.hist([[r for r, aee, ree, ae, rbe in radial_errors if aee <= 5], [r for r, aee, ree, ae, rbe in radial_errors if 5 < aee <= 20], [r for r, aee, ree, ae, rbe in radial_errors if 20 < aee <= 50], [r for r, _ in failures[1]]], bins=[i * 0.05 for i in range(11)], alpha=0.5, label=['<5', '5-20', '20-50', 'Failures'], stacked=False)
                    ax.legend(loc='best', title='Pixel error', fontsize='x-small')

                    if motion_inx == 0:
                        ax.set_title('FOV {} degrees'.format(fov))
                    elif motion_inx == num_rows - 1:
                        ax.set_xlabel('Radial distance')
                    if fov_dict[fov] == 0:
                        ax.set_ylabel(motion, size='large')
                    elif fov_dict[fov] == num_cols - 1:
                        ax.set_ylabel('Number of tracks')
                        ax.yaxis.set_label_position("right")

                    df_lifetime = df_lifetime.append(pandas.DataFrame({'FOV': [fov for i in range(len(track_lengths[0]))], 'Track lifetime (frames)': track_lengths[0, :]}))
                    re_filt_ee = radial_errors[np.where((radial_errors[:, 1] < 50) & (radial_errors[:, 2] > 0))]
                    re_filt_ae = radial_errors[np.where(radial_errors[:, 1] < 50)]
                    re_filt_be = radial_errors[np.where((radial_errors[:, 1] < 50) & (radial_errors[:, 4] > 0))]
                    df_ee = df_ee.append(pandas.DataFrame({'Radial distance': np.round(re_filt_ee[:, 0] / 0.005) * 0.005, 'FOV': [fovstr for i in range(len(re_filt_ee))], 'Motion': [motion for i in range(len(re_filt_ee))], 'Relative endpoint error (pixels)': re_filt_ee[:, 2]}))
                    df_ae = df_ae.append(pandas.DataFrame({'Radial distance': np.round(re_filt_ae[:, 0] / 0.005) * 0.005, 'FOV': [fovstr for i in range(len(re_filt_ae))], 'Motion': [motion for i in range(len(re_filt_ae))], 'Angular error (radians)': re_filt_ae[:, 3]}))
                    df_be = df_be.append(pandas.DataFrame({'Radial distance': np.round(re_filt_be[:, 0] / 0.005) * 0.005, 'FOV': [fovstr for i in range(len(re_filt_be))], 'Motion': [motion for i in range(len(re_filt_be))], 'Relative bearing error (radians)': re_filt_be[:, 4]}))
                    # df_inlier = df_inlier.append(pandas.DataFrame({'Inliers': [float(len(successes[1][successes[1][:, 1] == i + 1])) / (len(successes[1][successes[1][:, 1] == i + 1]) + len(failures[1][failures[1][:, 1] == i + 1])) for i in range(frame_num)], 'FOV': [fov for i in range(frame_num)]}))
                    for rate, _ in failures.iteritems():
                        frame_num = max(int(failures[rate].max(axis=0)[1]), int(successes[rate].max(axis=0)[1]))
                        df_rate = df_rate.append(pandas.DataFrame({'Rate': [rate for i in range(frame_num)], 'Inliers': [float(len(successes[rate][successes[rate][:, 1] == i + 1])) / (len(successes[rate][successes[rate][:, 1] == i + 1]) + len(failures[rate][failures[rate][:, 1] == i + 1])) for i in range(frame_num)], 'FOV': [fov for i in range(frame_num)]}))
                        df_rate_errors = df_rate_errors.append(pandas.DataFrame({'Rate': [rate for i in range(len(radial_errors_dict[rate]))], 'Endpoint error': [aee for _, aee, _, _, _ in radial_errors_dict[rate]], 'FOV': [fov for i in range(len(radial_errors_dict[rate]))]}))
                    df_length = df_length.append(pandas.DataFrame({'Track length (frames)': length_errors[:, 0], 'Endpoint error': length_errors[:, 3], 'Bearing error': length_errors[:, 4], 'FOV': fovstr, 'Motion': motion}))
                    num_fovs += 1
                else:
                    print '[WARNING] No results files found in directory {} for FOV {}'.format(os.path.join(bag_dir), fovstr)
            if motion_exists:
                ax2 = fig2.add_subplot(num_rows, 1, motion_inx + 1)
                sns.violinplot(x='FOV', y='Track lifetime (frames)', data=df_lifetime, ax=ax2, palette="muted", inner='quart')
                ax2.set_title('Motion {}'.format(motion))
                ax_rate = fig3.add_subplot(motion_count, 2, 2 * motion_inx + 1)
                ax_rate_errors = fig3.add_subplot(motion_count, 2, 2 * motion_inx + 2)
                ax_rate.set_title('Motion {}'.format(motion))
                sns.lineplot(x="Rate", y="Inliers", hue="FOV", data=df_rate, ax=ax_rate, marker='o', ci='sd', legend='full', palette=sns.color_palette('muted', n_colors=num_fovs))
                sns.lineplot(x="Rate", y="Endpoint error", hue="FOV", data=df_rate_errors, ax=ax_rate_errors, marker='o', ci='sd', legend='full', palette=sns.color_palette('muted', n_colors=num_fovs))
                ax_rate.set_xlabel('Degrees per frame')
                ax_rate.set_ylabel('Inlier rate')
                motion_inx += 1

    # ax_ee = fig3.add_subplot(1, 3, 1)
    sns.relplot(x="Radial distance", y="Relative endpoint error (pixels)", kind="line", data=df_ee, ci="sd", row='Motion', hue='FOV', legend="full", hue_order=fovs, palette=sns.color_palette("muted", n_colors=len(fovs)))
    # ax_ee.set_title('Relative endpoint error (pixels)')
    # ax_ae = fig3.add_subplot(1, 3, 2)
    sns.relplot(x="Radial distance", y="Angular error (radians)", kind="line", data=df_ae, ci="sd", row='Motion', hue='FOV', legend="full", hue_order=fovs, palette=sns.color_palette("muted", n_colors=len(fovs)))
    # ax_ae.set_title('Angular error (radians)')
    # ax_be = fig3.add_subplot(1, 3, 3)
    sns.relplot(x="Radial distance", y="Relative bearing error (radians)", kind="line", data=df_be, ci="sd", row='Motion', hue='FOV', legend="full", hue_order=fovs, palette=sns.color_palette("muted", n_colors=len(fovs)))
    # ax_be.set_title('Relative bearing error (radians)')

    sns.relplot(x='Track length (frames)', y='Endpoint error', kind='line', data=df_length, ci='sd', row='Motion', hue='FOV', legend='full', hue_order=fovs, palette=sns.color_palette('muted', n_colors=len(fovs)))

    plt.show()

else:
    print "[ERROR] Invalid path specified"
