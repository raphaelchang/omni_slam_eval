import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import argparse
import os

parser = argparse.ArgumentParser(description='Plot matching evaluation results')
parser.add_argument('results_file',  help='matching results file, source bag file, or directory')
args = parser.parse_args()

if not os.path.isdir(args.results_file) and args.results_file.endswith('.hdf5'):
    with h5py.File(args.results_file, 'r') as f:
        stats = f['match_stats'][:]
        radial_overlaps_errors = f['radial_overlaps_errors'][:]
        good_radial_distances = f['good_radial_distances'][:]
        bad_radial_distances = f['bad_radial_distances'][:]
        roc = f['roc_curves'][:]
        precrec = f['precision_recall_curves'][:]
        attrs = dict(f['attributes'].attrs.items())

    df = pd.DataFrame(stats)
    stats = df.groupby(0).mean().to_records()
    stats = stats.view(np.float64).reshape(len(stats), -1)
    framediff, nmatch, prec, rec = stats.T

    fig = plt.figure()
    fig.suptitle('Matching - detector={}, descriptor={}, chi={}, alpha={}, fx={}, fy={}, cx={}, cy={}'.format(attrs["detector_type"], attrs["descriptor_type"], attrs["chi"][0], attrs["alpha"][0], attrs["fx"][0], attrs["fy"][0], attrs["cx"][0], attrs["cy"][0]))
    ax1 = fig.add_subplot(3, 2, 1)
    ax2 = fig.add_subplot(3, 2, 2)
    ax3 = fig.add_subplot(3, 2, 3, projection='3d')
    ax4 = fig.add_subplot(3, 2, 4)
    ax5 = fig.add_subplot(3, 2, 5)
    ax6 = fig.add_subplot(3, 2, 6)
    cm = plt.get_cmap('gist_ncar')
    numdesc = 1

    ax1.set_prop_cycle(color=[cm(1. * i / numdesc) for i in range(numdesc)])
    color = next(ax1._get_lines.prop_cycler)['color']
    ax1.plot(framediff * attrs["rate"][0], prec, color=color)
    ax1.plot(framediff * attrs["rate"][0], rec, linestyle='dashed', color=color)
    ax1.set_xlabel('Frame difference')
    ax1.set_ylabel('Precision / recall')
    ax1.set_title('Precision-recall over frame difference')

    ax2.set_prop_cycle(color=[cm(1. * i / numdesc) for i in range(numdesc)])
    color = next(ax2._get_lines.prop_cycler)['color']
    ax2.plot(framediff * attrs["rate"][0], nmatch, color=color)
    ax2.set_xlabel('Frame difference')
    ax2.set_ylabel('Number of matches')
    ax2.set_title('Number of matches over frame difference')

    ax3.set_prop_cycle(color=[cm(1. * i / numdesc) for i in range(numdesc)])
    ax3.set_title('ROC curve')
    ax3.set_xlabel('False positive rate')
    ax3.set_ylabel('True positive rate')
    ax3.set_zlabel('Frame difference')
    roc = roc[roc[:, 0] == 0]
    maxz = int(roc[:, 1].max(axis=0))
    maxx = 0
    maxy = 0
    color = next(ax3._get_lines.prop_cycler)['color']
    for i in range(1, maxz):
        y, x = np.absolute(roc[roc[:, 1] == i][:, 2:].T)
        maxx = max(maxx, max(x))
        maxy = max(maxy, max(y))
        ax3.plot(x, y, i, color=color)
    ax3.set_xlim([0, maxx])
    ax3.set_ylim([0, maxy])
    ax3.set_zlim([0, maxz])
    ax3.view_init(elev=100, azim=270)

    ax4.set_title('Distribution of matches over changes in radial distance')
    ax4.set_ylabel('Number of matches')
    ax4.set_xlabel('Delta radial distance')
    ax4.hist(good_radial_distances[:, 0].ravel(), color='c')

    df = pd.DataFrame({'Delta radial distance': ['{}-{}'.format(r * 0.05, (r + 1) * 0.05) for r in (np.minimum(np.hstack((good_radial_distances[:, 0], bad_radial_distances[:, 0])), 0.499999) / 0.05).astype(int)], 'Descriptor distance': np.hstack((good_radial_distances[:, 1], bad_radial_distances[:, 1])), 'Match': ['Good' for i in range(len(good_radial_distances))] + ['Bad' for i in range(len(bad_radial_distances))]})
    sns.violinplot(x="Delta radial distance", y="Descriptor distance", hue="Match", data=df, split=True, ax=ax5, palette="Set2", inner="quart")
    handles, labels = ax5.get_legend_handles_labels()
    ax5.legend(handles=handles[0:], labels=labels[0:], fontsize='small')
    ax5.set_title('Distribution of descriptor distances over changes in radial distance')

    binned_overlaps = [[0] for i in range(10)]
    for r, o, _ in radial_overlaps_errors:
        binned_overlaps[int(min(r, 0.499999) / 0.05)].append(o)
    ax6.set_title('Match IOU distribution for various radial distances')
    ax6.set_ylabel('IOU')
    ax6.set_xlabel('Radial distance')
    ax6.set_xticks(np.arange(1, 11))
    ax6.set_xticklabels(['{}-{}'.format(i * 0.05, (i + 1) * 0.05) for i in range(10)])
    ax6.violinplot(binned_overlaps)
    ax6.plot([i for i in range(1, 11)], [np.array(binned_overlaps[i]).mean() for i in range(10)], '-o', markersize=4, c='black')

    plt.show()

elif not os.path.isdir(args.results_file) and args.results_file.endswith('.bag'):
    radial_overlaps_errors = dict()
    good_radial_distances = dict()
    bad_radial_distances = dict()
    roc = dict()
    precrec = dict()
    framediff = dict()
    nmatch = dict()
    prec = dict()
    rec = dict()
    detdesclist = []
    for filename in os.listdir(os.path.dirname(args.results_file)):
        if filename.startswith(os.path.basename(args.results_file)) and filename.endswith('.matching.hdf5'):
            with h5py.File(os.path.join(os.path.dirname(args.results_file), filename), 'r') as f:
                attrs = dict(f['attributes'].attrs.items())
                detdesc = (attrs['detector_type'], attrs['descriptor_type'])
                detdesclist.append(detdesc)
                stats = f['match_stats'][:]
                radial_overlaps_errors[detdesc] = f['radial_overlaps_errors'][:]
                good_radial_distances[detdesc] = f['good_radial_distances'][:]
                bad_radial_distances[detdesc] = f['bad_radial_distances'][:]
                roc[detdesc] = f['roc_curves'][:]
                precrec[detdesc] = f['precision_recall_curves'][:]
                df = pd.DataFrame(stats)
                statsavg = df.groupby(0).mean().to_records()
                statsavg = statsavg.view(np.float64).reshape(len(statsavg), -1)
                framediff[detdesc], nmatch[detdesc], prec[detdesc], rec[detdesc] = statsavg.T
    if len(detdesclist) > 0:
        fig = plt.figure()
        fig.suptitle('Matching - chi={}, alpha={}, fx={}, fy={}, cx={}, cy={}'.format(attrs["chi"][0], attrs["alpha"][0], attrs["fx"][0], attrs["fy"][0], attrs["cx"][0], attrs["cy"][0]))
        ax1 = fig.add_subplot(3, 2, 1)
        ax2 = fig.add_subplot(3, 2, 2)
        ax3 = fig.add_subplot(3, 2, 3)
        ax4 = fig.add_subplot(3, 2, 4)
        ax5 = fig.add_subplot(3, 2, 5)
        ax6 = fig.add_subplot(3, 2, 6)

        cm = plt.get_cmap('gist_ncar')
        ax1.set_prop_cycle(color=[cm(1. * i / len(detdesclist)) for i in range(len(detdesclist))])
        ax2.set_prop_cycle(color=[cm(1. * i / len(detdesclist)) for i in range(len(detdesclist))])
        ax3.set_prop_cycle(color=[cm(1. * i / len(detdesclist)) for i in range(len(detdesclist))])

        handles = []
        for detdesc in detdesclist:
            color = next(ax1._get_lines.prop_cycler)['color']
            h1, = ax1.plot(framediff[detdesc] * attrs["rate"][0], prec[detdesc], color=color)
            h2, = ax1.plot(framediff[detdesc] * attrs["rate"][0], rec[detdesc], linestyle='dashed', color=color)
            handles.append((h1, h2))

        l1 = ax1.legend(handles, ['{}+{}'.format(det, desc) for det, desc in detdesclist], loc=1, title='Detector+Descriptor', fontsize='small')
        l2 = ax1.legend([h1, h2], ['Precision', 'Recall'], loc=4, fontsize='small')
        l2.legendHandles[0].set_color('black')
        l2.legendHandles[1].set_color('black')
        ax1.add_artist(l1)
        ax1.set_xlabel('Frame difference')
        ax1.set_ylabel('Precision / recall')
        ax1.set_title('Precision-recall over frame difference')

        handles = []
        for detdesc in detdesclist:
            color = next(ax2._get_lines.prop_cycler)['color']
            h, = ax2.plot(framediff[detdesc] * attrs["rate"][0], nmatch[detdesc], color=color)
            handles.append(h)

        l1 = ax2.legend(handles, ['{}+{}'.format(det, desc) for det, desc in detdesclist], loc=1, title='Detector+Descriptor', fontsize='small')
        ax2.add_artist(l1)
        ax2.set_xlabel('Frame difference')
        ax2.set_ylabel('Number of matches')
        ax2.set_title('Number of matches over frame difference')

        ax3.set_title('ROC curve')
        ax3.set_xlabel('False positive rate')
        ax3.set_ylabel('True positive rate')
        for detdesc in detdesclist:
            rocc = roc[detdesc][roc[detdesc][:, 0] == 0]
            color = next(ax3._get_lines.prop_cycler)['color']
            for i in range(1, int(rocc[:, 1].max(axis=0))):
                y, x = np.absolute(rocc[rocc[:, 1] == i][:, 2:].T)
                h, = ax3.plot(x, y, color=color)
            handles.append(h)
        l1 = ax3.legend(handles, ['{}+{}'.format(det, desc) for det, desc in detdesclist], loc=1, title='Detector+Descriptor', fontsize='small')
        ax3.add_artist(l1)

        plt.show()

    else:
        print "[ERROR] Bag filename had no associated results files"
else:
    print "[ERROR] Invalid path specified"

