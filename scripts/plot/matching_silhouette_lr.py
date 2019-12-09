import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
import argparse
import os

parser = argparse.ArgumentParser(description='Plot matching evaluation results')
parser.add_argument('results_path',  help='matching results file, source bag file, or working directory')
args = parser.parse_args()

fovs = []
for yaml in os.listdir(args.results_path):
    if not os.path.isdir(os.path.join(args.results_path, yaml)) and yaml.endswith('.yaml'):
        fov = os.path.splitext(os.path.basename(yaml))[0]
        fovs.append(fov)
fovs.sort(key=int)

lr_list = [250, 195, 160]

detdesclist = []
good_radial_distances = dict()
bad_radial_distances = dict()
good_radial_distances_lr = dict()
bad_radial_distances_lr = dict()
lrexists = dict()
detdescset = set()
for motion in os.listdir(args.results_path):
    if os.path.isdir(os.path.join(args.results_path, motion)):
        bag_dir = os.path.join(args.results_path, motion)
        for fovstr in fovs:
            for filename in os.listdir(bag_dir):
                if filename.split('.')[1] == fovstr and filename.endswith('.matching.hdf5') and "+LR" not in filename:
                    results_file = os.path.join(bag_dir, filename)
                    with h5py.File(results_file, 'r') as f:
                        attrs = dict(f['attributes'].attrs.items())
                        detdesc = (attrs['detector_type'], attrs['descriptor_type'], int(fovstr))
                        detdescset.add(attrs['detector_type'] + '+' + attrs['descriptor_type'])
                        detdesclist.append(detdesc)
                        good_sampled = f['good_radial_distances'][:]
                        # good_sampled = good_sampled[np.random.choice(good_sampled.shape[0], 1000, replace=False)]
                        bad_sampled = f['bad_radial_distances'][:]
                        # bad_sampled = bad_sampled[np.random.choice(bad_sampled.shape[0], 1000, replace=False)]
                        if detdesc not in good_radial_distances:
                            good_radial_distances[detdesc] = good_sampled
                        else:
                            good_radial_distances[detdesc] = np.vstack((good_radial_distances[detdesc], good_sampled))
                        if detdesc not in bad_radial_distances:
                            bad_radial_distances[detdesc] = bad_sampled
                        else:
                            bad_radial_distances[detdesc] = np.vstack((bad_radial_distances[detdesc], bad_sampled))
                elif filename.split('.')[1] == fovstr and filename.endswith('.matching.hdf5') and "+LR" in filename:
                    results_file = os.path.join(bag_dir, filename)
                    with h5py.File(results_file, 'r') as f:
                        attrs = dict(f['attributes'].attrs.items())
                        detdesc = (attrs['detector_type'], attrs['descriptor_type'], int(fovstr))
                        detdescset.add(attrs['detector_type'] + '+' + attrs['descriptor_type'])
                        detdesclist.append(detdesc)
                        good_sampled = f['good_radial_distances'][:]
                        # good_sampled = good_sampled[np.random.choice(good_sampled.shape[0], 1000, replace=False)]
                        bad_sampled = f['bad_radial_distances'][:]
                        # bad_sampled = bad_sampled[np.random.choice(bad_sampled.shape[0], 1000, replace=False)]
                        if detdesc not in good_radial_distances_lr:
                            good_radial_distances_lr[detdesc] = good_sampled
                        else:
                            good_radial_distances_lr[detdesc] = np.vstack((good_radial_distances_lr[detdesc], good_sampled))
                        if detdesc not in bad_radial_distances_lr:
                            bad_radial_distances_lr[detdesc] = bad_sampled
                        else:
                            bad_radial_distances_lr[detdesc] = np.vstack((bad_radial_distances_lr[detdesc], bad_sampled))
                        lrexists[detdesc] = True

if len(detdesclist) > 0:
    sns.set()

    detdesclist = sorted(list(set(detdesclist)))

    df_r = pd.DataFrame()
    num_bins = 20
    num_samples = 1000
    for detdesc in detdesclist:
        if detdesc in lrexists:
            sr = [0 for i in range(num_bins)]
            X_good = [np.array([]) for i in range(num_bins)]
            labels_good = [np.array([]) for i in range(num_bins)]
            X_bad = [np.array([]) for i in range(num_bins)]
            labels_bad = [np.array([]) for i in range(num_bins)]
            valid_good = [False for i in range(num_bins)]
            valid_bad = [False for i in range(num_bins)]
            valid = [False for i in range(num_bins)]
            for row in good_radial_distances[detdesc]:
                r = int(min(row[0], 0.499999) / (0.5 / num_bins))
                X_good[r] = np.append(X_good[r], row[2])
                labels_good[r] = np.append(labels_good[r], 0)
                valid_good[r] = True
            for row in bad_radial_distances[detdesc]:
                r = int(min(row[0], 0.499999) / (0.5 / num_bins))
                X_bad[r] = np.append(X_bad[r], row[2])
                labels_bad[r] = np.append(labels_bad[r], 1)
                valid_bad[r] = True
            for i in range(num_bins):
                valid[i] = valid_good[i] and valid_bad[i]
                if not valid[i]:
                    continue
                idx_good = np.arange(len(X_good[i]))
                idx_bad = np.arange(len(X_bad[i]))
                if len(X_good[i]) > num_samples:
                    idx_good = np.random.choice(np.arange(len(X_good[i])), num_samples, replace=False)
                if len(X_bad[i]) > num_samples:
                    idx_bad = np.random.choice(np.arange(len(X_bad[i])), num_samples, replace=False)
                sr[i] = sklearn.metrics.silhouette_score(np.concatenate((X_good[i][idx_good], X_bad[i][idx_bad])).reshape(-1, 1), np.concatenate((labels_good[i][idx_good], labels_bad[i][idx_bad])), metric = 'l1')
                df_r = df_r.append(pd.DataFrame({'Change in radial distance': [i * 0.5 / num_bins + 0.5 / num_bins / 2 for i in range(0, len(sr)) if valid[i]], 'Silhouette coefficient': [sr[i] for i in range(len(sr)) if valid[i]], 'Descriptor': '{}'.format(detdesc[1]), 'FOV': detdesc[2], 'LR': 'Without'}))

            if int(detdesc[2]) not in lr_list:
                continue
            sr = [0 for i in range(num_bins)]
            X_good = [np.array([]) for i in range(num_bins)]
            labels_good = [np.array([]) for i in range(num_bins)]
            X_bad = [np.array([]) for i in range(num_bins)]
            labels_bad = [np.array([]) for i in range(num_bins)]
            valid_good = [False for i in range(num_bins)]
            valid_bad = [False for i in range(num_bins)]
            valid = [False for i in range(num_bins)]
            for row in good_radial_distances_lr[detdesc]:
                r = int(min(row[0], 0.499999) / (0.5 / num_bins))
                X_good[r] = np.append(X_good[r], row[2])
                labels_good[r] = np.append(labels_good[r], 0)
                valid_good[r] = True
            for row in bad_radial_distances_lr[detdesc]:
                r = int(min(row[0], 0.499999) / (0.5 / num_bins))
                X_bad[r] = np.append(X_bad[r], row[2])
                labels_bad[r] = np.append(labels_bad[r], 1)
                valid_bad[r] = True
            for i in range(num_bins):
                valid[i] = valid_good[i] and valid_bad[i]
                if not valid[i]:
                    continue
                idx_good = np.arange(len(X_good[i]))
                idx_bad = np.arange(len(X_bad[i]))
                if len(X_good[i]) > num_samples:
                    idx_good = np.random.choice(np.arange(len(X_good[i])), num_samples, replace=False)
                if len(X_bad[i]) > num_samples:
                    idx_bad = np.random.choice(np.arange(len(X_bad[i])), num_samples, replace=False)
                sr[i] = sklearn.metrics.silhouette_score(np.concatenate((X_good[i][idx_good], X_bad[i][idx_bad])).reshape(-1, 1), np.concatenate((labels_good[i][idx_good], labels_bad[i][idx_bad])), metric = 'l1')
                df_r = df_r.append(pd.DataFrame({'Change in radial distance': [i * 0.5 / num_bins + 0.5 / num_bins / 2 for i in range(0, len(sr)) if valid[i]], 'Silhouette coefficient': [sr[i] for i in range(len(sr)) if valid[i]], 'Descriptor': '{}'.format(detdesc[1]), 'FOV': detdesc[2], 'LR': 'With'}))

    g1 = sns.relplot(x='Change in radial distance', y="Silhouette coefficient", hue='FOV', marker='o', palette=sns.color_palette('muted', n_colors=len(fovs)), col='Descriptor', kind='line', style='LR', data=df_r, estimator=None, aspect=1.6, height=3, dashes=['', (5, 5)], style_order=['With', 'Without'], markers=['o', '.'])
    [plt.setp(ax.texts, text="") for ax in g1.axes.flat]
    g1.set_titles(col_template='{col_name}')
    g1.fig.subplots_adjust(hspace=0.2, right=0.87)
    g1.savefig('silhouette_lr.png')

    plt.show()
