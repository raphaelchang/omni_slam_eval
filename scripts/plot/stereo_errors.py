import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
import os
from parse import parse
import argparse

parser = argparse.ArgumentParser(description='Plot stereo evaluation results')
parser.add_argument('results_path', help='stereo results file or working directory')
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
            depth_errors = np.empty(shape=(0,3))
            file_exists = False
            for filename in os.listdir(bag_dir):
                if filename.split('.')[1] == fovstr and filename.endswith('.stereo.hdf5'):
                    results_file = os.path.join(bag_dir, filename)
                    with h5py.File(results_file, 'r') as f:
                        attrs = dict(f['attributes'].attrs.items())
                        rate = int(attrs['rate'])
                        if rate > 1:
                            continue
                        depth_errors = np.vstack((depth_errors, f['depth_errors'][:]))
                        file_exists = True
            if file_exists:
                df = df.append(pandas.DataFrame({'Radial distance': [str(x) for x in np.round(np.minimum(depth_errors[:, 0], 0.499999) / 0.1) / 10.], 'FOV': fovstr, 'Disparity': depth_errors[:, 1], 'Normalized depth error': depth_errors[:, 2]}))

df = df[np.isfinite(df['Normalized depth error'])]
# g = sns.relplot(x="Radial distance", y="Normalized depth error", kind="line", data=df, ci="sd", hue='FOV', legend="full", hue_order=fovs, palette=sns.color_palette("muted", n_colors=len(fovs)), aspect=2.2, height=2.5, facet_kws={'despine': True}, err_kws={'alpha': 0.15})
g = sns.catplot(x="Radial distance", y="Normalized depth error", kind="box", data=df, hue='FOV', legend="full", orient='v', hue_order=fovs, palette=sns.color_palette("muted", n_colors=len(fovs)), aspect=2.5, height=4, dodge=True, notch=True, fliersize=0.5)
g.fig.subplots_adjust(left=0.08, right=0.9)
g.set(ylim=(0, 100), yscale='log')
g.savefig('depth_error.png')

plt.show()

