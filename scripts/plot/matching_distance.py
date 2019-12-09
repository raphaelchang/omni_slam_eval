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
bad_radial_distances = dict()
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

if len(detdesclist) > 0:
    sns.set()

    detdesclist = sorted(list(set(detdesclist)))

    df = pd.DataFrame()
    for detdesc in detdesclist:
        matches = np.hstack((good_radial_distances[detdesc][:, 2], bad_radial_distances[detdesc][:, 2]))
        matches /= matches.max()
        raddist = (np.minimum(np.hstack((good_radial_distances[detdesc][:, 0], bad_radial_distances[detdesc][:, 0])), 0.499999) / 0.05).astype(int)
        # angle = (np.hstack((good_radial_distances[detdesc][:, 1], bad_radial_distances[detdesc][:, 1])) / (np.pi / 12)).astype(int)
        angle = np.round(np.hstack((good_radial_distances[detdesc][:, 1], bad_radial_distances[detdesc][:, 1])) / (np.pi / 36)) * np.pi / 36 * 180 / np.pi
        df = df.append(pd.DataFrame({'Change in radial distance': ['{}-{}'.format(r * 0.05, (r + 1) * 0.05) for r in raddist], 'Change in ray angle (degrees)': angle, 'Detector+Descriptor': '{}+{}'.format(detdesc[0], detdesc[1]), 'Normalized descriptor distance': matches, 'Match': ['Good' for i in range(len(good_radial_distances[detdesc]))] + ['Bad' for i in range(len(bad_radial_distances[detdesc]))], 'FOV': detdesc[2]}))
    g1 = sns.catplot(y='Change in radial distance', x="Normalized descriptor distance", hue='Match', palette='Set2', col='FOV', row='Detector+Descriptor', kind='violin', data=df, orient='h', split=True, inner="quart", margin_titles=True, aspect=1.1, height=2.2)
    g2 = sns.relplot(x='Change in ray angle (degrees)', y="Normalized descriptor distance", hue='Match', palette=sns.color_palette('Set2', n_colors=2), row='Detector+Descriptor', col='FOV', kind='line', data=df, ci='sd', aspect=1, height=2, facet_kws={'margin_titles': True})
    [plt.setp(ax.texts, text="") for ax in g1.axes.flat]
    g1.set_titles(row_template='{row_name}', col_template='FOV={col_name}')
    for i in range(len(detdescset)):
        if i != len(detdescset) / 2:
            g1.facet_axis(i, 0).set_ylabel('')
    for i in range(len(fovs)):
        if i != len(fovs) / 2:
            g1.facet_axis(len(detdescset) - 1, i).set_xlabel('')
        g1.set(xlim=(0,1), xticks=[0, 0.25, 0.5, 0.75, 1])
        g1.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])

    [plt.setp(ax.texts, text="") for ax in g2.axes.flat]
    g2.set_titles(row_template='{row_name}', col_template='FOV={col_name}')
    for i in range(len(detdescset)):
        if i != len(detdescset) / 2:
            g2.facet_axis(i, 0).set_ylabel('')
    for i in range(len(fovs)):
        if i != len(fovs) / 2:
            g2.facet_axis(len(detdescset) - 1, i).set_xlabel('')
        g2.set(xlim=(0,120), xticks=[0, 30, 60, 90, 120])

    g1.fig.subplots_adjust(right=0.9, hspace=0.13, wspace=0.13)
    g1.savefig('descriptordist.png'.format(motion))
    g2.fig.subplots_adjust(right=0.88, hspace=0.2, wspace=0.2)
    g2.savefig('descriptordistray.png'.format(motion))

    plt.show()
