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

fovs = []
for yaml in os.listdir(args.results_path):
    if not os.path.isdir(os.path.join(args.results_path, yaml)) and yaml.endswith('.yaml'):
        fov = os.path.splitext(os.path.basename(yaml))[0]
        fovs.append(fov)
fovs.sort(key=int)

detdesclist = []
good_radial_distances = dict()
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
                        if detdesc not in good_radial_distances:
                            good_radial_distances[detdesc] = f['good_radial_distances'][:]
                        else:
                            good_radial_distances[detdesc] = np.vstack((good_radial_distances[detdesc], f['good_radial_distances'][:]))

if len(detdesclist) > 0:
    sns.set()

    detdesclist = sorted(list(set(detdesclist)))

    df = pd.DataFrame()
    for detdesc in detdesclist:
        df = df.append(pd.DataFrame({'Change in radial distance': good_radial_distances[detdesc][:, 0], 'Change in ray angle (degrees)': good_radial_distances[detdesc][:, 1] * 180 / np.pi, 'Descriptor': '{}'.format(detdesc[1]), 'FOV': detdesc[2]}))
    # g1 = sns.catplot(x='Change in radial distance', y="Detector+Descriptor", hue='FOV', palette=sns.color_palette('muted', n_colors=len(fovs)), kind='violin', data=df, orient='h', inner="quartile", dodge=True, cut=0)
    # g2 = sns.catplot(x='Change in ray angle (degrees)', y="Detector+Descriptor", hue='FOV', palette=sns.color_palette('muted', n_colors=len(fovs)), kind='violin', data=df, orient='h', inner="quartile", dodge=True, cut=0)
    g1 = sns.catplot(y='Change in radial distance', x="Descriptor", hue='FOV', palette=sns.color_palette('muted', n_colors=len(fovs)), kind='violin', data=df, orient='w', inner="quartile", dodge=True, cut=0, aspect=4, height=3)
    g2 = sns.catplot(y='Change in ray angle (degrees)', x="Descriptor", hue='FOV', palette=sns.color_palette('muted', n_colors=len(fovs)), kind='violin', data=df, orient='w', inner="quartile", dodge=True, cut=0, aspect=4, height=3)

    g1.fig.subplots_adjust(right=0.91)
    g1.savefig('matchdist.png'.format(motion))
    g2.fig.subplots_adjust(right=0.91)
    g2.savefig('matchdistray.png'.format(motion))

    plt.show()

