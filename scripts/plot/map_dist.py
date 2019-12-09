import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
import os
from parse import parse
import argparse

parser = argparse.ArgumentParser(description='Plot reconstruction evaluation results')
parser.add_argument('results_path', help='reconstruction results file or working directory')
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
            landmarks = np.empty(shape=(0,7))
            file_exists = False
            for filename in os.listdir(bag_dir):
                if filename.split('.')[1] == fovstr and filename.endswith('.reconstruction.hdf5'):
                    results_file = os.path.join(bag_dir, filename)
                    with h5py.File(results_file, 'r') as f:
                        attrs = dict(f['attributes'].attrs.items())
                        rate = int(attrs['rate'])
                        if rate > 1:
                            continue
                        landmarks = np.vstack((landmarks, f['landmarks'][:]))
                        file_exists = True
            if file_exists:
                landmarks_correct = landmarks[np.linalg.norm(landmarks[:, 0:3] - landmarks[:, 3:6], axis=1) < 0.1]
                new = pandas.DataFrame({'FOV': fovstr, 'x (m)': landmarks[:, 0], 'y (m)': landmarks[:, 1], 'z (m)': landmarks[:, 2]})
                print 'FOV: {}, X range: {}m, Y range: {}m, Z range: {}m'.format(fovstr, new['x (m)'].max() - new['x (m)'].min(), new['y (m)'].max() - new['y (m)'].min(), new['z (m)'].max() - new['z (m)'].min())
                df = df.append(new.sample(n=50000, replace=False))

# g = sns.FacetGrid(df, hue='FOV', palette='Set3', legend_out=True)
# g = g.map(sns.distplot, 'x', hist=False)
g = sns.relplot(x="x (m)", y="y (m)", kind="scatter", data=df, col='FOV', hue='FOV', col_order=fovs, col_wrap=3, hue_order=fovs, legend="full", s=12, palette=sns.color_palette("muted", n_colors=len(fovs)), aspect=1, height=3, facet_kws={'despine': True})

g.savefig('map_dist.png')

plt.show()


