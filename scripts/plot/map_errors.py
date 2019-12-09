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

df_gnd = pandas.DataFrame()
df_est = pandas.DataFrame()
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
                df_gnd = df_gnd.append(pandas.DataFrame({'FOV': fovstr, 'Error (meters)': np.linalg.norm(landmarks[:, 0:3] - landmarks[:, 3:6], axis=1)}))
            landmarks = np.empty(shape=(0,7))
            file_exists = False
            for filename in os.listdir(bag_dir):
                if filename.split('.')[1] == fovstr and filename.endswith('.slam.hdf5'):
                    results_file = os.path.join(bag_dir, filename)
                    with h5py.File(results_file, 'r') as f:
                        attrs = dict(f['attributes'].attrs.items())
                        rate = int(attrs['rate'])
                        if rate > 1:
                            continue
                        landmarks = np.vstack((landmarks, f['landmarks'][:]))
                        file_exists = True
            if file_exists:
                df_est = df_est.append(pandas.DataFrame({'FOV': fovstr, 'Error (meters)': np.linalg.norm(landmarks[:, 0:3] - landmarks[:, 3:6], axis=1)}))

df_gnd = df_gnd[np.isfinite(df_gnd['Error (meters)'])]
df_est = df_est[np.isfinite(df_est['Error (meters)'])]
g1 = sns.catplot(x="FOV", y="Error (meters)", kind="box", data=df_est, order=fovs, legend=False, orient='v', palette=sns.color_palette("muted", n_colors=len(fovs)), aspect=1.5, height=4, dodge=True, notch=True, fliersize=0.001)
g2 = sns.catplot(x="FOV", y="Error (meters)", kind="box", data=df_gnd, order=fovs, legend=False, orient='v', palette=sns.color_palette("muted", n_colors=len(fovs)), aspect=1.5, height=4, dodge=True, notch=True, fliersize=0.001)
g1.fig.subplots_adjust(left=0.08, right=0.9)
g1.set(ylim=(0, 3))
g1.savefig('map_errors_gnd.png')
g2.fig.subplots_adjust(left=0.08, right=0.9)
g2.set(ylim=(0, 3))
g2.savefig('map_errors_est.png')

plt.show()

