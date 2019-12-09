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

motion_map = {'yaw': 'Yaw',  'strafe_side': 'Sideways Translate', 'strafe_back': 'Backward Translate'}
motion_frame_map = {'yaw': [1, 2, 5, 10, 20, 30, 45, 60, 90, 120, 180], 'strafe_side': [1, 2, 5, 10, 25, 50, 100], 'strafe_back': [1, 2, 5, 10, 20, 50]}
unit_map = {'yaw': 'degrees', 'strafe_side': 'meters', 'strafe_back': 'meters'}
scale_map = {'yaw': 1, 'strafe_side': 0.2, 'strafe_back': 0.5}
lr_list = [250, 195, 160]
motion_order = ['Yaw', 'Sideways Translate', 'Backward Translate']

fovs = []
for yaml in os.listdir(args.results_path):
    if not os.path.isdir(os.path.join(args.results_path, yaml)) and yaml.endswith('.yaml'):
        fov = os.path.splitext(os.path.basename(yaml))[0]
        fovs.append(fov)
fovs.sort(key=int)

precrec = dict()
precreclr = dict()
lrexists = dict()
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
                elif filename.split('.')[1] == fovstr and filename.endswith('.matching.hdf5') and "+LR" in filename:
                    results_file = os.path.join(bag_dir, filename)
                    with h5py.File(results_file, 'r') as f:
                        attrs = dict(f['attributes'].attrs.items())
                        detdesc = (attrs['detector_type'], attrs['descriptor_type'], int(fovstr), motion)
                        precreclr[detdesc] = f['precision_recall_curves'][:]
                        lrexists[detdesc] = True

if len(detdesclist) > 0:
    sns.set()

    detdesclist = sorted(list(set(detdesclist)))

    baseline_set = dict()

    df_auc = pd.DataFrame()
    for detdesc in detdesclist:
        if detdesc in lrexists:
            pr = precrec[detdesc][precrec[detdesc][:, 0] == 0]
            for i in range(1, int(pr[:, 1].max(axis=0))):
                if detdesc[3] == 'yaw' and i > int(detdesc[2]) and i < 360 - int(detdesc[2]) and int(detdesc[2]) < 180:
                    df_auc = df_auc.append(pd.DataFrame({'Descriptor': ['{}'.format(detdesc[1])], 'FOV': [int(detdesc[2])], 'AUC': [0], 'Baseline': [i * scale_map[detdesc[3]]], 'Motion': [motion_map[detdesc[3]]], 'LR': 'Without'}))
                    continue
                y, x = np.absolute(pr[pr[:, 1] == i][:, 2:].T)
                auc = np.trapz(np.flip(y), np.flip(x))
                df_auc = df_auc.append(pd.DataFrame({'Descriptor': ['{}'.format(detdesc[1])], 'FOV': [int(detdesc[2])], 'AUC': [auc], 'Baseline': [i * scale_map[detdesc[3]]], 'Motion': [motion_map[detdesc[3]]], 'LR': 'Without'}))
            if int(detdesc[2]) not in lr_list:
                continue
            pr = precreclr[detdesc][precreclr[detdesc][:, 0] == 0]
            for i in range(1, int(pr[:, 1].max(axis=0))):
                if detdesc[3] == 'yaw' and i > int(detdesc[2]) and i < 360 - int(detdesc[2]) and int(detdesc[2]) < 180:
                    df_auc = df_auc.append(pd.DataFrame({'Descriptor': ['{}'.format(detdesc[1])], 'FOV': [int(detdesc[2])], 'AUC': [0], 'Baseline': [i * scale_map[detdesc[3]]], 'Motion': [motion_map[detdesc[3]]], 'LR': 'With'}))
                    continue
                y, x = np.absolute(pr[pr[:, 1] == i][:, 2:].T)
                auc = np.trapz(np.flip(y), np.flip(x))
                df_auc = df_auc.append(pd.DataFrame({'Descriptor': ['{}'.format(detdesc[1])], 'FOV': [int(detdesc[2])], 'AUC': [auc], 'Baseline': [i * scale_map[detdesc[3]]], 'Motion': [motion_map[detdesc[3]]], 'LR': 'With'}))

    g2 = sns.relplot(y='AUC', x='Baseline', hue='FOV', row='Motion', col='Descriptor', kind='line', data=df_auc, estimator=None, legend='full', palette=sns.color_palette('muted', n_colors=df_auc.FOV.unique().shape[0]), aspect=2.3, height=2.2, style='LR', facet_kws={'sharex': False, 'margin_titles': True}, dashes=['', (5, 5)], style_order=['With', 'Without'], row_order=motion_order)
    i = 0
    for motion, unit in unit_map.iteritems():
        g2.facet_axis(i, 0).set_xlabel('Baseline ({})'.format(unit))
        g2.facet_axis(i, 1).set_xlabel('Baseline ({})'.format(unit))
        i += 1
    [plt.setp(ax.texts, text="") for ax in g2.axes.flat]
    g2.set_titles(row_template='{row_name}', col_template='{col_name}')
    g2.fig.subplots_adjust(hspace=0.42, right=0.865)
    g2.savefig('auc_lr.png')

    plt.show()

