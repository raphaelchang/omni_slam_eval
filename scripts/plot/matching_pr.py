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
motion_frame_map = {'yaw': [1, 2, 5, 10, 20, 30, 45, 60, 90, 120, 180], 'strafe_side': [1, 2, 5, 10, 25, 50, 100], 'strafe_back': [1, 2, 5, 10, 20, 50]}
unit_map = {'yaw': 'degrees', 'strafe_side': 'meters', 'strafe_back': 'meters'}
scale_map = {'yaw': 1, 'strafe_side': 0.2, 'strafe_back': 0.5}

fovs = []
for yaml in os.listdir(args.results_path):
    if not os.path.isdir(os.path.join(args.results_path, yaml)) and yaml.endswith('.yaml'):
        fov = os.path.splitext(os.path.basename(yaml))[0]
        fovs.append(fov)
fovs.sort(key=int)

precrec = dict()
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
                        detdesc = (attrs['detector_type'], attrs['descriptor_type'], int(fovstr), motion)
                        detdesclist.append(detdesc)
                        precrec[detdesc] = f['precision_recall_curves'][:]

if len(detdesclist) > 0:
    sns.set()

    detdesclist = sorted(list(set(detdesclist)))

    baseline_set = dict()
    df_pr = pd.DataFrame()
    for detdesc in detdesclist:
        frames = motion_frame_map[detdesc[3]]
        pr = precrec[detdesc][precrec[detdesc][:, 0] == 0]
        if motion_map[detdesc[3]] not in baseline_set:
            baseline_set[motion_map[detdesc[3]]] = [i * scale_map[detdesc[3]] for i in motion_frame_map[detdesc[3]]]
        else:
            baseline_set[motion_map[detdesc[3]]] = list(set().union(baseline_set[motion_map[detdesc[3]]], [i * scale_map[detdesc[3]] for i in motion_frame_map[detdesc[3]]]))
        for i in frames:
            if detdesc[3] == 'yaw' and i >= int(detdesc[2]) and int(detdesc[2]) < 180:
                continue
            y, x = np.absolute(pr[pr[:, 1] == i][:, 2:].T)
            if style_map[detdesc[3]] is not False:
                df_pr = df_pr.append(pd.DataFrame({'Detector+Descriptor': '{}+{}'.format(detdesc[0], detdesc[1]), 'FOV': detdesc[2], 'Precision': y, 'Recall': x, 'Baseline ({})'.format(unit_map[detdesc[3]]): i * scale_map[detdesc[3]], 'Motion': motion_map[detdesc[3]], 'Direction': style_map[detdesc[3]]}))
            else:
                df_pr = df_pr.append(pd.DataFrame({'Detector+Descriptor': '{}+{}'.format(detdesc[0], detdesc[1]), 'FOV': detdesc[2], 'Precision': y, 'Recall': x, 'Baseline ({})'.format(unit_map[detdesc[3]]): i * scale_map[detdesc[3]], 'Motion': motion_map[detdesc[3]]}))

    df_auc = pd.DataFrame()
    for detdesc in detdesclist:
        pr = precrec[detdesc][precrec[detdesc][:, 0] == 0]
        for i in range(1, int(pr[:, 1].max(axis=0))):
            if detdesc[3] == 'yaw' and i > int(detdesc[2]) and i < 360 - int(detdesc[2]) and int(detdesc[2]) < 180:
                df_auc = df_auc.append(pd.DataFrame({'Detector+Descriptor': ['{}+{}'.format(detdesc[0], detdesc[1])], 'FOV': [int(detdesc[2])], 'AUC': [0], 'Baseline ({})'.format(unit_map[detdesc[3]]): [i * scale_map[detdesc[3]]], 'Motion': [motion_map[detdesc[3]]]}))
                continue
            y, x = np.absolute(pr[pr[:, 1] == i][:, 2:].T)
            auc = np.trapz(np.flip(y), np.flip(x))
            if style_map[detdesc[3]] is not False:
                df_auc = df_auc.append(pd.DataFrame({'Detector+Descriptor': ['{}+{}'.format(detdesc[0], detdesc[1])], 'FOV': [int(detdesc[2])], 'AUC': [auc], 'Baseline ({})'.format(unit_map[detdesc[3]]): [i * scale_map[detdesc[3]]], 'Motion': [motion_map[detdesc[3]]], 'Direction': style_map[detdesc[3]]}))
            else:
                df_auc = df_auc.append(pd.DataFrame({'Detector+Descriptor': ['{}+{}'.format(detdesc[0], detdesc[1])], 'FOV': [int(detdesc[2])], 'AUC': [auc], 'Baseline ({})'.format(unit_map[detdesc[3]]): [i * scale_map[detdesc[3]]], 'Motion': [motion_map[detdesc[3]]]}))

    for motion, _ in motion_map.iteritems():
        # palette = sns.cubehelix_palette(rot=-0.4, n_colors=len(baseline_set[motion_map[motion]]))
        palette = sns.color_palette("mako_r", len(baseline_set[motion_map[motion]]))
        if style_map[motion] is not False:
            g1 = sns.relplot(y='Precision', x='Recall', hue='Baseline ({})'.format(unit_map[motion]), col='FOV', row='Detector+Descriptor', kind='line', data=df_pr.loc[df_pr['Motion'] == motion_map[motion]], estimator=None, facet_kws={'margin_titles': True}, legend='full', palette=palette, aspect=1.25, height=1.8, style='Direction')
            g2 = sns.relplot(y='AUC', x='Baseline ({})'.format(unit_map[motion]), hue='FOV', row='Detector+Descriptor', kind='line', data=df_auc.loc[df_auc['Motion'] == motion_map[motion]], estimator=None, legend='full', palette=sns.color_palette('muted', n_colors=df_auc.FOV.unique().shape[0]), aspect=3, height=1.8, style='Direction')
            g2.fig.subplots_adjust(hspace=0.25, right=0.77)
        else:
            g1 = sns.relplot(y='Precision', x='Recall', hue='Baseline ({})'.format(unit_map[motion]), col='FOV', row='Detector+Descriptor', kind='line', data=df_pr.loc[df_pr['Motion'] == motion_map[motion]], estimator=None, facet_kws={'margin_titles': True}, legend='full', palette=palette, aspect=1.25, height=1.8)
            g2 = sns.relplot(y='AUC', x='Baseline ({})'.format(unit_map[motion]), hue='FOV', row='Detector+Descriptor', kind='line', data=df_auc.loc[df_auc['Motion'] == motion_map[motion]], estimator=None, legend='full', palette=sns.color_palette('muted', n_colors=df_auc.FOV.unique().shape[0]), aspect=3, height=1.8)
            g2.fig.subplots_adjust(hspace=0.25, right=0.84)
        g1.fig.subplots_adjust(right=0.825, hspace=0.2, wspace=0.2)
        [plt.setp(ax.texts, text="") for ax in g1.axes.flat]
        g1.set_titles(row_template='{row_name}', col_template='FOV={col_name}')
        # g1.savefig('pr_{}.png'.format(motion))
        g2.savefig('auc_{}.png'.format(motion))

    plt.show()
