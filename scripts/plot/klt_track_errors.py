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

motion_map = {'yaw': 'Yaw/pitch', 'roll': 'Roll', 'strafe_side': 'Sideways translate', 'strafe_forward': 'Forward translate', 'strafe_back': 'Backward translate', 'composite': 'Composite'}

df_length = pandas.DataFrame()
for motion in os.listdir(args.results_path):
    if os.path.isdir(os.path.join(args.results_path, motion)):
        bag_dir = os.path.join(args.results_path, motion)
        for fovstr in fovs:
            length_errors = np.empty(shape=(0,5))
            for filename in os.listdir(bag_dir):
                if filename.split('.')[1] == fovstr and filename.endswith('.tracking.hdf5'):
                    results_file = os.path.join(bag_dir, filename)
                    with h5py.File(results_file, 'r') as f:
                        attrs = dict(f['attributes'].attrs.items())
                        rate = int(attrs['rate'])
                        if rate > 1:
                            continue
                        length_errors = np.vstack((length_errors, f['length_errors'][:]))
                        file_exists = True
            if file_exists:
                if motion in motion_map:
                    motion = motion_map[motion]
                    length_errors = length_errors[np.where((length_errors[:, 3] > 0) & (length_errors[:, 4] > 0) & (length_errors[:, 3] < 20))]
                df_length = df_length.append(pandas.DataFrame({'Track length (frames)': length_errors[:, 0], 'Endpoint error (pixels)': length_errors[:, 3], 'Bearing error (radians)': length_errors[:, 4], 'FOV': fovstr, 'Motion': motion}))

motion_order = ['Yaw/pitch', 'Roll', 'Sideways translate', 'Forward translate', 'Backward translate', 'Composite']
g1 = sns.relplot(x="Track length (frames)", y="Endpoint error (pixels)", kind="line", data=df_length, ci="sd", col='Motion', col_wrap=2, col_order=motion_order, hue='FOV', legend="full", hue_order=fovs, palette=sns.color_palette("muted", n_colors=len(fovs)), aspect=2.2, height=2.5, facet_kws={'despine': True, 'sharex': False, 'sharey': False}, err_kws={'alpha': 0.15})
g2 = sns.relplot(x="Track length (frames)", y="Bearing error (radians)", kind="line", data=df_length, ci="sd", col='Motion', col_wrap=2, col_order=motion_order, hue='FOV', legend="full", hue_order=fovs, palette=sns.color_palette("muted", n_colors=len(fovs)), aspect=2.2, height=2.5, facet_kws={'despine': True, 'sharex': False, 'sharey': False}, err_kws={'alpha': 0.15})
g1.facet_axis(0, 0).set_ylabel('')
g1.facet_axis(0, 4).set_ylabel('')
g2.facet_axis(0, 0).set_ylabel('')
g2.facet_axis(0, 4).set_ylabel('')
g1.facet_axis(0, 0).set(ylim=(-5, 15))
g2.facet_axis(0, 0).set(ylim=(-0.01, 0.04))
g1.facet_axis(0, 1).set(ylim=(-2, 8))
g2.facet_axis(0, 1).set(ylim=(-0.005, 0.03))
g1.facet_axis(0, 5).set(ylim=(-10, 40))
g2.facet_axis(0, 5).set(ylim=(-0.02, 0.08))
g1.fig.subplots_adjust(hspace=0.3, right=0.9)
g2.fig.subplots_adjust(hspace=0.3, right=0.9)
g1.savefig('epe_length.png')
g2.savefig('be_length.png')

plt.show()
