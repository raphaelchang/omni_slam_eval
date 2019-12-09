import h5py
import numpy as np
import pandas as pd
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker

parser = argparse.ArgumentParser(description='Plot matching evaluation results')
parser.add_argument('results_path',  help='matching results file, source bag file, or working directory')
args = parser.parse_args()

with h5py.File(args.results_path, 'r') as f:
    stats = f['match_stats'][:]
    radial_overlaps_errors = f['radial_overlaps_errors'][:]
    good_radial_distances = f['good_radial_distances'][:]
    bad_radial_distances = f['bad_radial_distances'][:]
    roc = f['roc_curves'][:]
    pr = f['precision_recall_curves'][:]
    precrec = f['precision_recall_curves'][:]
    attrs = dict(f['attributes'].attrs.items())

df = pd.DataFrame(stats)
stats = df.groupby(0).mean().to_records()
stats = stats.view(np.float64).reshape(len(stats), -1)
framediff, nmatch, prec, rec, rep = stats.T

df_pr = pd.DataFrame()
for i in [1, 5, 10]:
    y, x = np.absolute(pr[pr[:, 1] == i][:, 2:].T)
    df_pr = df_pr.append(pd.DataFrame({'Precision': y, 'Recall': x, 'Hue': i}))

df_roc = pd.DataFrame()
for i in [1, 3, 5]:
    y, x = np.absolute(roc[roc[:, 1] == i][:, 2:].T)
    df_roc = df_roc.append(pd.DataFrame({'True positive rate': y, 'False positive rate': x, 'Hue': i}))

sns.set()
g1 = sns.relplot(data=df_pr, x='Recall', y='Precision', hue='Hue', legend=None, estimator=None, kind='line', aspect=1.5, height=3.5, palette=sns.color_palette("muted", n_colors=3, desat=0.5))
g2 = sns.relplot(data=df_roc, x='False positive rate', y='True positive rate', hue='Hue', legend=None, estimator=None, kind='line', aspect=1.5, height=3.5, palette=sns.color_palette('Set1', n_colors=3, desat=0.5))
for ax in g2.axes.flatten():
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useOffset=True)
g1.savefig('pr_sample.png')
g2.savefig('roc_sample.png')
plt.show()
