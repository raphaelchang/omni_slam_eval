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

motion_map = {'yaw': 'Yaw/pitch',  'strafe_side': 'Sideways translate', 'strafe_back': 'Backward translate'}
unit_map = {'yaw': 'degrees/frame',  'strafe_side': 'meters/frame', 'strafe_back': 'meters/frame'}
scale_map = {'yaw': 1, 'strafe_side': 0.2, 'strafe_back': 0.5}

df_rate = pandas.DataFrame()
df_rate_errors = pandas.DataFrame()
for motion in os.listdir(args.results_path):
    if os.path.isdir(os.path.join(args.results_path, motion)):
        bag_dir = os.path.join(args.results_path, motion)
        for fovstr in fovs:
            failures = dict()
            successes = dict()
            radial_errors_dict = dict()
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
                        file_exists = True
            if file_exists:
                if motion in motion_map:
                    motion_new = motion_map[motion]
                    for rate, _ in failures.iteritems():
                        frame_num = max(int(failures[rate].max(axis=0)[1]), int(successes[rate].max(axis=0)[1]))
                        df_rate = df_rate.append(pandas.DataFrame({'Rate': [rate * scale_map[motion] for i in range(frame_num)], 'Inlier ratio': [float(len(successes[rate][successes[rate][:, 1] == i + 1])) / (len(successes[rate][successes[rate][:, 1] == i + 1]) + len(failures[rate][failures[rate][:, 1] == i + 1])) for i in range(frame_num)], 'FOV': [fovstr for i in range(frame_num)], 'Motion': motion_new}))
                        re_filt_ee = radial_errors_dict[rate][np.where((radial_errors_dict[rate][:, 1] < 50) & (radial_errors_dict[rate][:, 2] > 0))]
                        df_rate_errors = df_rate_errors.append(pandas.DataFrame({'Rate': [rate * scale_map[motion] for i in range(len(re_filt_ee))], 'Relative endpoint error (pixels)': [ree for _, _, ree, _, _ in re_filt_ee], 'FOV': [fovstr for i in range(len(re_filt_ee))], 'Motion': motion_new}))


motion_order = ['Yaw/pitch', 'Sideways translate', 'Backward translate']
g1 = sns.relplot(x="Rate", y="Inlier ratio", hue="FOV", kind="line", data=df_rate, marker='o', ci='sd', row="Motion", row_order=motion_order, legend='full', hue_order=fovs, palette=sns.color_palette('muted', n_colors=len(fovs)), aspect=2.2, height=2.8, facet_kws={'sharex': False, 'ylim': (0, 1)}, err_kws={'alpha': 0.15})
g2 = sns.relplot(x="Rate", y="Relative endpoint error (pixels)", hue="FOV", kind="line", data=df_rate_errors, marker='o', ci='sd', row="Motion", row_order=motion_order, legend='full', hue_order=fovs, palette=sns.color_palette('muted', n_colors=len(fovs)), aspect=2.2, height=2.8, facet_kws={'sharex': False, 'sharey': False}, err_kws={'alpha': 0.15})
g1.facet_axis(0, 0).set_ylabel('')
g1.facet_axis(2, 0).set_ylabel('')
g1.facet_axis(0, 0).set_xlabel('Rate ({})'.format(unit_map['yaw']))
g1.facet_axis(1, 0).set_xlabel('Rate ({})'.format(unit_map['strafe_side']))
g1.facet_axis(2, 0).set_xlabel('Rate ({})'.format(unit_map['strafe_back']))
g2.facet_axis(0, 0).set_ylabel('')
g2.facet_axis(2, 0).set_ylabel('')
g2.facet_axis(0, 0).set_xlabel('Rate ({})'.format(unit_map['yaw']))
g2.facet_axis(1, 0).set_xlabel('Rate ({})'.format(unit_map['strafe_side']))
g2.facet_axis(2, 0).set_xlabel('Rate ({})'.format(unit_map['strafe_back']))
g1.fig.subplots_adjust(hspace=0.4, right=0.84)
g2.fig.subplots_adjust(hspace=0.4, right=0.84)
g1.savefig('rate_inliers.png')
g2.savefig('rate_errors.png')

plt.show()
