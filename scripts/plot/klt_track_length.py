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

df = pandas.DataFrame()
for motion in os.listdir(args.results_path):
    if os.path.isdir(os.path.join(args.results_path, motion)):
        bag_dir = os.path.join(args.results_path, motion)
        for fovstr in fovs:
            track_lengths = np.empty(shape=(1,0))
            for filename in os.listdir(bag_dir):
                if filename.split('.')[1] == fovstr and filename.endswith('.tracking.hdf5'):
                    results_file = os.path.join(bag_dir, filename)
                    with h5py.File(results_file, 'r') as f:
                        attrs = dict(f['attributes'].attrs.items())
                        rate = int(attrs['rate'])
                        if rate > 1:
                            continue
                        tl = f['track_lengths'][:]
                        track_lengths = np.hstack((track_lengths, tl[:, tl[0, :] > 2]))
                        file_exists = True
            if file_exists:
                if motion in motion_map:
                    motion = motion_map[motion]

                df = df.append(pandas.DataFrame({'Motion': motion, 'FOV': [fovstr for i in range(len(track_lengths[0]))], 'Track lifetime (frames)': track_lengths[0, :]}))

latex = ''
for _, motion in motion_map.iteritems():
    latex += motion
    for fov in fovs:
        latex += ' & '
        rows = df.loc[(df['Motion'] == motion) & (df['FOV'] == fov)]
        latex += '{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\'.format(fov, rows['Track lifetime (frames)'].mean(), rows['Track lifetime (frames)'].median(), rows['Track lifetime (frames)'].quantile(0.75), rows['Track lifetime (frames)'].std())
        print latex
        latex = ''
