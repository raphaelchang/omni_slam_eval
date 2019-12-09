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

df_ee = pandas.DataFrame()
df_ae = pandas.DataFrame()
df_be = pandas.DataFrame()
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
                        radial_errors = np.vstack((radial_errors, f['radial_errors'][:]))
                        file_exists = True
            if file_exists:
                if motion in motion_map:
                    motion = motion_map[motion]

                re_filt_ee = radial_errors[np.where((radial_errors[:, 1] < 50) & (radial_errors[:, 2] > 0))]
                re_filt_ae = radial_errors[np.where(radial_errors[:, 1] < 50)]
                re_filt_be = radial_errors[np.where((radial_errors[:, 1] < 50) & (radial_errors[:, 4] > 0))]
                df_ee = df_ee.append(pandas.DataFrame({'Radial distance': np.round(re_filt_ee[:, 0] / 0.005) * 0.005, 'FOV': [fovstr for i in range(len(re_filt_ee))], 'Motion': [motion for i in range(len(re_filt_ee))], 'Relative endpoint error (pixels)': re_filt_ee[:, 2]}))
                df_ae = df_ae.append(pandas.DataFrame({'Radial distance': np.round(re_filt_ae[:, 0] / 0.005) * 0.005, 'FOV': [fovstr for i in range(len(re_filt_ae))], 'Motion': [motion for i in range(len(re_filt_ae))], 'Angular error (radians)': re_filt_ae[:, 3]}))
                df_be = df_be.append(pandas.DataFrame({'Radial distance': np.round(re_filt_be[:, 0] / 0.005) * 0.005, 'FOV': [fovstr for i in range(len(re_filt_be))], 'Motion': [motion for i in range(len(re_filt_be))], 'Relative bearing error (radians)': re_filt_be[:, 4]}))

motion_order = ['Yaw/pitch', 'Roll', 'Sideways translate', 'Forward translate', 'Backward translate', 'Composite']
g1 = sns.relplot(x="Radial distance", y="Relative endpoint error (pixels)", kind="line", data=df_ee, ci="sd", col='Motion', col_wrap=2, col_order=motion_order, hue='FOV', legend="full", hue_order=fovs, palette=sns.color_palette("muted", n_colors=len(fovs)), aspect=2.2, height=2.5, facet_kws={'despine': True, 'sharey': False}, err_kws={'alpha': 0.15})
g2 = sns.relplot(x="Radial distance", y="Angular error (radians)", kind="line", data=df_ae, ci="sd", col='Motion', col_wrap=2, col_order=motion_order, hue='FOV', legend="full", hue_order=fovs, palette=sns.color_palette("muted", n_colors=len(fovs)), aspect=2.2, height=2.5, facet_kws={'despine': True, 'sharey': False}, err_kws={'alpha': 0.15})
g3 = sns.relplot(x="Radial distance", y="Relative bearing error (radians)", kind="line", data=df_be, ci="sd", col='Motion', col_wrap=2, col_order=motion_order, hue='FOV', legend="full", hue_order=fovs, palette=sns.color_palette("muted", n_colors=len(fovs)), aspect=2.2, height=2.5, facet_kws={'despine': True, 'sharey': False}, err_kws={'alpha': 0.15})
g1.facet_axis(0, 0).set_ylabel('')
g1.facet_axis(0, 4).set_ylabel('')
g2.facet_axis(0, 0).set_ylabel('')
g2.facet_axis(0, 4).set_ylabel('')
g3.facet_axis(0, 0).set_ylabel('')
g3.facet_axis(0, 4).set_ylabel('')
g1.facet_axis(0, 0).set(ylim=(-0.4,1.0))
g1.facet_axis(0, 1).set(ylim=(-0.4,1.0))
g1.facet_axis(0, 2).set(ylim=(-0.4,1.2))
g1.facet_axis(0, 3).set(ylim=(-0.5,1.5))
g1.facet_axis(0, 4).set(ylim=(-0.5,2.0))
g2.facet_axis(0, 0).set(ylim=(-0.1,0.15))
g2.facet_axis(0, 2).set(ylim=(-0.2,0.5))
g2.facet_axis(0, 4).set(ylim=(-0.2,1.0))
g2.facet_axis(0, 5).set(ylim=(-0.15,0.3))
g3.facet_axis(0, 0).set(ylim=(-0.001,0.0025))
g3.facet_axis(0, 1).set(ylim=(-0.001,0.0025))
g3.facet_axis(0, 3).set(ylim=(-0.002,0.005))
g3.facet_axis(0, 4).set(ylim=(-0.002,0.008))
# g3.facet_axis(0, 0).ticklabel_format(style='sci', axis='y', scilimits=(0,0), useOffset=True)
# g3.facet_axis(0, 1).ticklabel_format(style='sci', axis='y', scilimits=(0,0), useOffset=True)
g1.fig.subplots_adjust(hspace=0.25, right=0.9)
g2.fig.subplots_adjust(hspace=0.25, right=0.9)
g3.fig.subplots_adjust(hspace=0.25, right=0.9)
g1.savefig('epe.png')
g2.savefig('ae.png')
g3.savefig('be.png')

plt.show()
