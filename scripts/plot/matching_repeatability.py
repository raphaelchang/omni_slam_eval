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

motion_map = {'yaw': 'Yaw',  'strafe_side': 'Strafe', 'strafe_back': 'Strafe'}
style_map = {'yaw': False, 'strafe_side': 'Sideways', 'strafe_back': 'Backwards'}
unit_map = {'yaw': 'degrees', 'strafe_side': 'meters', 'strafe_back': 'meters'}
scale_map = {'yaw': 1, 'strafe_side': 0.2, 'strafe_back': 0.5}

fovs = []
for yaml in os.listdir(args.results_path):
    if not os.path.isdir(os.path.join(args.results_path, yaml)) and yaml.endswith('.yaml'):
        fov = os.path.splitext(os.path.basename(yaml))[0]
        fovs.append(fov)
fovs.sort(key=int)

rep = dict()
framediff = dict()
detdesclist = []
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
                        detdesc = (attrs['detector_type'], int(fovstr), motion)
                        detdesclist.append(detdesc)
                        stats = f['match_stats'][:]
                        df = pd.DataFrame(stats)
                        statsavg = df.groupby(0).mean().to_records()
                        statsavg = statsavg.view(np.float64).reshape(len(statsavg), -1)
                        framediff[detdesc], _, _, _, rep[detdesc] = statsavg.T

if len(detdesclist) > 0:
    sns.set()

    detdesclist = sorted(list(set(detdesclist)))

    df = pd.DataFrame()
    for detdesc in detdesclist:
        for i in range(int(framediff[detdesc].max())):
            if i not in framediff[detdesc] and detdesc[2] == 'yaw' and i > int(detdesc[1]) and i < 360 - int(detdesc[1]) and int(detdesc[1]) < 180:
                df = df.append(pd.DataFrame({'Detector': detdesc[0], 'FOV': int(detdesc[1]), 'Repeatability': [0], 'Baseline ({})'.format(unit_map[detdesc[2]]): i * scale_map[detdesc[2]], 'Motion': motion_map[detdesc[2]], 'Direction': style_map[detdesc[2]]}))
                continue
        if style_map[detdesc[2]] is not False:
            df = df.append(pd.DataFrame({'Detector': detdesc[0], 'FOV': int(detdesc[1]), 'Repeatability': rep[detdesc], 'Baseline ({})'.format(unit_map[detdesc[2]]): framediff[detdesc] * scale_map[detdesc[2]], 'Motion': motion_map[detdesc[2]], 'Direction': style_map[detdesc[2]]}))
        else:
            df = df.append(pd.DataFrame({'Detector': detdesc[0], 'FOV': int(detdesc[1]), 'Repeatability': rep[detdesc], 'Baseline ({})'.format(unit_map[detdesc[2]]): framediff[detdesc] * scale_map[detdesc[2]], 'Motion': motion_map[detdesc[2]]}))

    for motion, _ in motion_map.iteritems():
        if style_map[motion] is not False:
            g = sns.relplot(y='Repeatability', x='Baseline ({})'.format(unit_map[motion]), hue='FOV', row='Detector', kind='line', data=df.loc[df['Motion'] == motion_map[motion]], estimator=None, legend='full', palette=sns.color_palette('muted', n_colors=df.FOV.unique().shape[0]), aspect=3, height=1.8, style='Direction')
            g.fig.subplots_adjust(hspace=0.25, right=0.77)
        else:
            g = sns.relplot(y='Repeatability', x='Baseline ({})'.format(unit_map[motion]), hue='FOV', row='Detector', kind='line', data=df.loc[df['Motion'] == motion_map[motion]], estimator=None, legend='full', palette=sns.color_palette('muted', n_colors=df.FOV.unique().shape[0]), aspect=3, height=1.8)
            g.fig.subplots_adjust(hspace=0.25, right=0.84)
        [plt.setp(ax.texts, text="") for ax in g.axes.flat]
        g.set_titles(row_template='{row_name}', col_template='FOV={col_name}')
        g.savefig('repeatability_{}.png'.format(motion))

    plt.show()

