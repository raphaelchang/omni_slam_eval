import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

parser = argparse.ArgumentParser(description='Plot matching evaluation results')
parser.add_argument('results_path',  help='matching results file, source bag file, or working directory')
args = parser.parse_args()

motion_map = {'yaw': 'Yaw'}
unit_map = {'yaw': 'degrees'}
scale_map = {'yaw': 1}

fovs = []
for yaml in os.listdir(args.results_path):
    if not os.path.isdir(os.path.join(args.results_path, yaml)) and yaml.endswith('.yaml'):
        fov = os.path.splitext(os.path.basename(yaml))[0]
        fovs.append(fov)
fovs.sort(key=int)

rot = dict()
detdesclist = []
detdescset = set()
for motion in os.listdir(args.results_path):
    if os.path.isdir(os.path.join(args.results_path, motion)):
        if motion not in motion_map:
            continue
        bag_dir = os.path.join(args.results_path, motion)
        for fovstr in fovs:
            for filename in os.listdir(bag_dir):
                if filename.split('.')[1] == fovstr and filename.endswith('.matching.hdf5') and "+LR" not in filename:
                    results_file = os.path.join(bag_dir, filename)
                    with h5py.File(results_file, 'r') as f:
                        attrs = dict(f['attributes'].attrs.items())
                        detdesc = (attrs['detector_type'], attrs['descriptor_type'], int(fovstr), motion)
                        detdescset.add(attrs['detector_type'] + '+' + attrs['descriptor_type'])
                        detdesclist.append(detdesc)
                        rot[detdesc] = f['rotation_errors'][:]

if len(detdesclist) > 0:
    sns.set()

    detdesclist = sorted(list(set(detdesclist)))

    df = pd.DataFrame()
    df_inlier = pd.DataFrame()
    for detdesc in detdesclist:
        rot[detdesc][:, 2] = np.abs(rot[detdesc][:, 2])
        rot[detdesc][:, 2][rot[detdesc][:, 2] > np.pi / 2.] = np.pi - rot[detdesc][:, 2][rot[detdesc][:, 2] > np.pi / 2.]
        rot[detdesc][:, 2] = np.abs(rot[detdesc][:, 2])
        rot[detdesc][:, 2] *= 180. / np.pi
        for i in range(1, int(rot[detdesc][:, 1].max(axis=0))):
            if detdesc[3] == 'yaw' and i > int(detdesc[2]) and i < 360 - int(detdesc[2]) and int(detdesc[2]) < 180:
                df = df.append(pd.DataFrame({'Detector+Descriptor': ['{}+{}'.format(detdesc[0], detdesc[1])], 'FOV': [int(detdesc[2])], 'Rotation error (degrees)': 180, 'Baseline ({})'.format(unit_map[detdesc[3]]): [np.round(i / 5.) * 5. * scale_map[detdesc[3]]], 'Motion': [motion_map[detdesc[3]]]}))
                df_inlier = df_inlier.append(pd.DataFrame({'Detector+Descriptor': ['{}+{}'.format(detdesc[0], detdesc[1])], 'FOV': [int(detdesc[2])], 'Inlier ratio': 0, 'Baseline ({})'.format(unit_map[detdesc[3]]): [i * scale_map[detdesc[3]]], 'Motion': [motion_map[detdesc[3]]]}))
                continue
            df = df.append(pd.DataFrame({'Detector+Descriptor': ['{}+{}'.format(detdesc[0], detdesc[1])], 'FOV': [int(detdesc[2])], 'Rotation error (degrees)': rot[detdesc][:, 2][rot[detdesc][:, 1] == i], 'Baseline ({})'.format(unit_map[detdesc[3]]): [np.round(i / 5.) * 5. * scale_map[detdesc[3]]], 'Motion': [motion_map[detdesc[3]]]}))
            df_inlier = df_inlier.append(pd.DataFrame({'Detector+Descriptor': ['{}+{}'.format(detdesc[0], detdesc[1])], 'FOV': [int(detdesc[2])], 'Inlier ratio': rot[detdesc][:, 3][rot[detdesc][:, 1] == i] / rot[detdesc][:, 4][rot[detdesc][:, 1] == i], 'Baseline ({})'.format(unit_map[detdesc[3]]): [i * scale_map[detdesc[3]]], 'Motion': [motion_map[detdesc[3]]]}))

    for motion, _ in motion_map.iteritems():
        g1 = sns.relplot(y='Rotation error (degrees)', x='Baseline ({})'.format(unit_map[motion]), hue='FOV', row='Detector+Descriptor', kind='line', data=df.loc[df['Motion'] == motion_map[motion]], ci='sd', legend='full', palette=sns.color_palette('muted', n_colors=df.FOV.unique().shape[0]), aspect=3, height=1.8)
        g1.fig.subplots_adjust(hspace=0.25, right=0.84)
        g2 = sns.relplot(y='Inlier ratio', x='Baseline ({})'.format(unit_map[motion]), hue='FOV', row='Detector+Descriptor', kind='line', data=df_inlier.loc[df_inlier['Motion'] == motion_map[motion]], estimator=None, legend='full', palette=sns.color_palette('muted', n_colors=df_inlier.FOV.unique().shape[0]), aspect=3, height=1.8)
        g2.fig.subplots_adjust(hspace=0.25, right=0.84)
        for i in range(len(detdescset)):
            if i != len(detdescset) / 2:
                g1.facet_axis(i, 0).set_ylabel('')
                g1.set(ylim=(0, 5))
        g1.savefig('rotation.png')
        g2.savefig('rotation_inliers.png')

    plt.show()

