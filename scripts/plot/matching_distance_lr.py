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

lr_list = [250, 195, 160]

detdesclist = []
good_radial_distances = dict()
good_radial_distances_lr = dict()
bad_radial_distances = dict()
bad_radial_distances_lr = dict()
lrexists = dict()
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
                        good_sampled = f['good_radial_distances'][:]
                        # good_sampled = good_sampled[np.random.choice(good_sampled.shape[0], 1000, replace=False)]
                        bad_sampled = f['bad_radial_distances'][:]
                        # bad_sampled = bad_sampled[np.random.choice(bad_sampled.shape[0], 1000, replace=False)]
                        if detdesc not in good_radial_distances:
                            good_radial_distances[detdesc] = good_sampled
                        else:
                            good_radial_distances[detdesc] = np.vstack((good_radial_distances[detdesc], good_sampled))
                        if detdesc not in bad_radial_distances:
                            bad_radial_distances[detdesc] = bad_sampled
                        else:
                            bad_radial_distances[detdesc] = np.vstack((bad_radial_distances[detdesc], bad_sampled))
                elif filename.split('.')[1] == fovstr and filename.endswith('.matching.hdf5') and "+LR" in filename:
                    results_file = os.path.join(bag_dir, filename)
                    with h5py.File(results_file, 'r') as f:
                        attrs = dict(f['attributes'].attrs.items())
                        detdesc = (attrs['detector_type'], attrs['descriptor_type'], int(fovstr))
                        detdescset.add(attrs['detector_type'] + '+' + attrs['descriptor_type'])
                        detdesclist.append(detdesc)
                        good_sampled = f['good_radial_distances'][:]
                        # good_sampled = good_sampled[np.random.choice(good_sampled.shape[0], 1000, replace=False)]
                        bad_sampled = f['bad_radial_distances'][:]
                        # bad_sampled = bad_sampled[np.random.choice(bad_sampled.shape[0], 1000, replace=False)]
                        if detdesc not in good_radial_distances_lr:
                            good_radial_distances_lr[detdesc] = good_sampled
                        else:
                            good_radial_distances_lr[detdesc] = np.vstack((good_radial_distances_lr[detdesc], good_sampled))
                        if detdesc not in bad_radial_distances_lr:
                            bad_radial_distances_lr[detdesc] = bad_sampled
                        else:
                            bad_radial_distances_lr[detdesc] = np.vstack((bad_radial_distances_lr[detdesc], bad_sampled))
                        lrexists[detdesc] = True

if len(detdesclist) > 0:
    sns.set()

    detdesclist = sorted(list(set(detdesclist)))

    df = pd.DataFrame()
    for detdesc in detdesclist:
        if detdesc in lrexists:
            if int(detdesc[2]) not in lr_list:
                continue
            matches = np.hstack((good_radial_distances[detdesc][:, 2], bad_radial_distances[detdesc][:, 2]))
            matches /= matches.max()
            raddist = (np.minimum(np.hstack((good_radial_distances[detdesc][:, 0], bad_radial_distances[detdesc][:, 0])), 0.499999) / 0.05).astype(int)
            df = df.append(pd.DataFrame({'Change in radial distance': ['{}-{}'.format(r * 0.05, (r + 1) * 0.05) for r in raddist], 'Descriptor': '{} | FOV {}'.format(detdesc[1], detdesc[2]), 'Normalized descriptor distance': matches, 'Match': ['Good' for i in range(len(good_radial_distances[detdesc]))] + ['Bad' for i in range(len(bad_radial_distances[detdesc]))], 'FOV': detdesc[2], 'LR': 'Without'}))

            matches = np.hstack((good_radial_distances_lr[detdesc][:, 2], bad_radial_distances_lr[detdesc][:, 2]))
            matches /= matches.max()
            raddist = (np.minimum(np.hstack((good_radial_distances_lr[detdesc][:, 0], bad_radial_distances_lr[detdesc][:, 0])), 0.499999) / 0.05).astype(int)
            df = df.append(pd.DataFrame({'Change in radial distance': ['{}-{}'.format(r * 0.05, (r + 1) * 0.05) for r in raddist], 'Descriptor': '{} | FOV {}'.format(detdesc[1], detdesc[2]), 'Normalized descriptor distance': matches, 'Match': ['Good' for i in range(len(good_radial_distances_lr[detdesc]))] + ['Bad' for i in range(len(bad_radial_distances_lr[detdesc]))], 'FOV': detdesc[2], 'LR': 'With'}))

    g1 = sns.catplot(y='Change in radial distance', x="Normalized descriptor distance", hue='Match', palette='Set2', row='LR', col='Descriptor', col_order=['BRISK | FOV 160', 'BRISK | FOV 195', 'BRISK | FOV 250', 'ORB | FOV 160', 'ORB | FOV 195', 'ORB | FOV 250'], kind='violin', data=df, orient='h', split=True, inner="quart", margin_titles=True, aspect=0.8, height=2.5)
    [plt.setp(ax.texts, text="") for ax in g1.axes.flat]
    g1.set_titles(row_template='{row_name} LR', col_template='{col_name}')
    g1.facet_axis(1, 0).set_ylabel('')
    g1.facet_axis(0, 0).set_ylabel('Change in radial distance                                      ')
    for i in range(6):
        if i != 3:
            g1.facet_axis(1, i).set_xlabel('')
        else:
            g1.facet_axis(1, i).set_xlabel('Normalized descriptor distance                              ')
    g1.set(xlim=(0,1), xticks=[0, 0.25, 0.5, 0.75, 1])
    g1.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])

    g1.fig.subplots_adjust(right=0.9, hspace=0.13, wspace=0.13)
    g1.savefig('descriptordist_lr.png')

    plt.show()


