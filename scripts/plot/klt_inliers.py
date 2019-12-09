import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
import os
from parse import parse
import argparse

parser = argparse.ArgumentParser(description='Plot tracking evaluation results')
parser.add_argument('results_path', help='tracking results file or working directory')
args = parser.parse_args()

sns.set()
fovs = []
for yaml in os.listdir(args.results_path):
    if not os.path.isdir(os.path.join(args.results_path, yaml)) and yaml.endswith('.yaml'):
        fov = os.path.splitext(os.path.basename(yaml))[0]
        fovs.append(fov)
fovs.sort(key=int)

df = pandas.DataFrame()
for motion in os.listdir(args.results_path):
    if os.path.isdir(os.path.join(args.results_path, motion)):
        bag_dir = os.path.join(args.results_path, motion)
        for fovstr in fovs:
            radial_errors = np.empty(shape=(0,5))
            for filename in os.listdir(bag_dir):
                if filename.split('.')[1] == fovstr and filename.endswith('.tracking.hdf5'):
                    results_file = os.path.join(bag_dir, filename)
                    with h5py.File(results_file, 'r') as f:
                        attrs = dict(f['attributes'].attrs.items())
                        rate = int(attrs['rate'])
                        if rate > 1:
                            continue
                        failures = f['failures'][:]
                        successes = f['successes'][:]
                        file_exists = True
            if file_exists:
                lim = max(int(successes.max(axis=0)[1]), int(failures.max(axis=0)[1]))
                binned_inliers = [[0 for j in range(20)] for i in range(lim)]
                binned_outliers = [[0 for j in range(20)] for i in range(lim)]
                for r, f in successes:
                    if np.isnan(r):
                        continue
                    binned_inliers[int(f) - 1][int(r / 0.05)] += 1
                for r, f in failures:
                    if np.isnan(r):
                        continue
                    binned_outliers[int(f) - 1][int(r / 0.05)] += 1
                binned_ratio = [0. for i in range(lim)]
                for r in range(20):
                    for f in range(lim):
                        total = binned_outliers[f][r] + binned_inliers[f][r]
                        if total > 0:
                            binned_ratio[r] += binned_inliers[f][r] / float(total)
                    binned_ratio[r] /= lim
                df = df.append(pandas.DataFrame({'FOV': fovstr, 'Radial distance': [r / 20. for r in range(11) if binned_ratio[r] > 0], 'Inlier ratio': [binned_ratio[r] for r in range(11) if binned_ratio[r] > 0]}))


g = sns.catplot(x="Radial distance", y="Inlier ratio", kind="bar", data=df, hue='FOV', legend="full", hue_order=fovs, palette=sns.color_palette("muted", n_colors=len(fovs)), aspect=3, height=3, facet_kws={'despine': True}, ci='sd')
g.fig.subplots_adjust(hspace=0.25, right=0.9)
g.set(ylim=(0,1))
g.savefig('inlier_dist.png')

plt.show()

