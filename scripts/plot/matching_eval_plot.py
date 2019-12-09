import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import argparse
import os
import sklearn.metrics
import random

parser = argparse.ArgumentParser(description='Plot matching evaluation results')
parser.add_argument('results_path',  help='matching results file, source bag file, or working directory')
parser.add_argument('--detector_descriptor', help='detector+descriptor type for bag file')
parser.add_argument('--fov', help='fov for bag file')
args = parser.parse_args()

if not os.path.isdir(args.results_path) and args.results_path.endswith('.hdf5'):
    with h5py.File(args.results_path, 'r') as f:
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
    framediff, nmatch, prec, rec, rep = stats.T

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
    ax1.plot(framediff * attrs["rate"][0], prec, color=color, label='Precision')
    ax1.plot(framediff * attrs["rate"][0], rec, linestyle='dashed', color=color, label='Recall')
    ax1.set_xlabel('Frame difference')
    ax1.set_ylabel('Precision / recall')
    ax1.set_title('Precision-recall over frame difference')
    ax1.legend(loc=1, fontsize='small')

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
        if len(x) == 0 or len(y) == 0:
            continue
        maxx = max(maxx, max(x))
        maxy = max(maxy, max(y))
        ax3.plot(x, y, i, color=color)
    ax3.set_xlim([0, maxx])
    ax3.set_ylim([0, maxy])
    ax3.set_zlim([0, maxz])
    ax3.view_init(elev=100, azim=270)

    # ax4.set_title('Distribution of matches over changes in radial distance')
    # ax4.set_ylabel('Number of matches')
    # ax4.set_xlabel('Delta radial distance')
    # ax4.hist(good_radial_distances[:, 0].ravel(), color='c')

    # match_dict = [[] for i in range(10)]
    # for row in good_radial_distances:
        # r = int(min(row[0], 0.499999) / 0.05)
        # match_dict[r].append((row[2], 0))
    # for row in bad_radial_distances:
        # r = int(min(row[0], 0.499999) / 0.05)
        # match_dict[r].append((row[2], 1))
    # si_bins = [0] * 10
    # for r in range(len(match_dict)):
        # match_dict[r] = sorted(match_dict[r], key=lambda x: x[0])
        # if len(match_dict[r]) <= 1:
            # si_bins[r] = 1
            # continue
        # good_nn = 0
        # for i in range(len(match_dict[r])):
            # if i == 0:
                # if match_dict[r][i][1] == match_dict[r][i + 1]:
                    # good_nn += 1
            # elif i == len(match_dict[r]) - 1:
                # if match_dict[r][i][1] == match_dict[r][i - 1]:
                    # good_nn += 1
            # else:
                # if match_dict[r][i][0] - match_dict[r][i - 1][0] > match_dict[r][i + 1][0] - match_dict[r][i][0]:
                    # if match_dict[r][i + 1][1] == match_dict[r][i][1]:
                        # good_nn += 1
                # else:
                    # if match_dict[r][i][1] == match_dict[r][i - 1][1]:
                        # good_nn += 1
        # si_bins[r] = good_nn / float(len(match_dict[r]))
    # ax4.plot([i for i in range(0, len(match_dict))], [si_bins[i] for i in range(0, len(match_dict))])

    ax4.set_title('Match separability over changes in radial distance')
    ax4.set_ylabel('Silhouette coefficient')
    ax4.set_xlabel('Delta radial distance')
    num_bins = 50
    num_samples = 5000
    sr = [0 for i in range(num_bins)]
    X_good = [np.array([]) for i in range(num_bins)]
    labels_good = [np.array([]) for i in range(num_bins)]
    X_bad = [np.array([]) for i in range(num_bins)]
    labels_bad = [np.array([]) for i in range(num_bins)]
    valid_good = [False for i in range(num_bins)]
    valid_bad = [False for i in range(num_bins)]
    valid = [False for i in range(num_bins)]
    # for row in good_radial_distances:
        # r = int(min(row[0], 0.499999) / (0.5 / num_bins))
        # X_good[r] = np.append(X_good[r], row[2])
        # labels_good[r] = np.append(labels_good[r], 0)
        # valid_good[r] = True
    # for row in bad_radial_distances:
        # r = int(min(row[0], 0.499999) / (0.5 / num_bins))
        # X_bad[r] = np.append(X_bad[r], row[2])
        # labels_bad[r] = np.append(labels_bad[r], 1)
        # valid_bad[r] = True
    for row in good_radial_distances:
        if np.isnan(row[1]):
            continue
        r = int(min(row[1], np.pi) / (np.pi / num_bins))
        X_good[r] = np.append(X_good[r], row[2])
        labels_good[r] = np.append(labels_good[r], 0)
        valid_good[r] = True
    for row in good_radial_distances:
        if np.isnan(row[1]):
            continue
        r = int(min(row[1], np.pi) / (np.pi / num_bins))
        X_bad[r] = np.append(X_bad[r], row[2])
        labels_bad[r] = np.append(labels_bad[r], 1)
        valid_bad[r] = True
    for i in range(num_bins):
        valid[i] = valid_good[i] and valid_bad[i]
        if not valid[i]:
            continue
        idx_good = np.arange(len(X_good[i]))
        idx_bad = np.arange(len(X_bad[i]))
        if len(X_good[i]) > num_samples:
            idx_good = np.random.choice(np.arange(len(X_good[i])), num_samples, replace=False)
        if len(X_bad[i]) > num_samples:
            idx_bad = np.random.choice(np.arange(len(X_bad[i])), num_samples, replace=False)
        sr[i] = sklearn.metrics.silhouette_score(np.concatenate((X_good[i][idx_good], X_bad[i][idx_bad])).reshape(-1, 1), np.concatenate((labels_good[i][idx_good], labels_bad[i][idx_bad])), metric = 'l1')
    # ax4.plot([i * 0.5 / num_bins + 0.5 / num_bins / 2 for i in range(0, len(sr)) if valid[i]], [sr[i] for i in range(0, len(sr)) if valid[i]])
    ax4.plot([i * np.pi / num_bins + np.pi / num_bins / 2 for i in range(0, len(sr)) if valid[i]], [sr[i] for i in range(0, len(sr)) if valid[i]])

    df = pd.DataFrame({'Delta radial distance': ['{}-{}'.format(r * 0.05, (r + 1) * 0.05) for r in (np.minimum(np.hstack((good_radial_distances[:, 0], bad_radial_distances[:, 0])), 0.499999) / 0.05).astype(int)], 'Descriptor distance': np.hstack((good_radial_distances[:, 2], bad_radial_distances[:, 2])), 'Match': ['Good' for i in range(len(good_radial_distances))] + ['Bad' for i in range(len(bad_radial_distances))]})
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

elif not os.path.isdir(args.results_path) and args.results_path.endswith('.bag'):
    radial_overlaps_errors = dict()
    good_radial_distances = dict()
    bad_radial_distances = dict()
    roc = dict()
    precrec = dict()
    framediff = dict()
    nmatch = dict()
    prec = dict()
    rec = dict()
    rep = dict()
    detdesclist = []
    for filename in os.listdir(os.path.dirname(args.results_path)):
        bagname = os.path.splitext(os.path.basename(args.results_path))[0]
        if args.fov is not None:
            bagname += '.' + args.fov
        if filename.startswith(bagname) and filename.endswith('.matching.hdf5'):
            with h5py.File(os.path.join(os.path.dirname(args.results_path), filename), 'r') as f:
                if args.fov is None:
                    fov = int(filename[len(bagname):].split('.')[1])
                else:
                    fov = int(args.fov)
                attrs = dict(f['attributes'].attrs.items())
                if args.detector_descriptor is not None:
                    if attrs['detector_type'] + '_' + attrs['descriptor_type'] != args.detector_descriptor:
                        continue
                detdesc = (attrs['detector_type'], attrs['descriptor_type'], fov)
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
                framediff[detdesc], nmatch[detdesc], prec[detdesc], rec[detdesc], rep[detdesc] = statsavg.T

    if len(detdesclist) > 0:
        fig = plt.figure()
        sns.set()

        detdesclist = sorted(detdesclist)
        legendlist = []
        if args.fov is not None:
            legendlist = ['{}+{}'.format(det, desc) for det, desc, fov in detdesclist]
            legendtitle = 'Detector+Descriptor'
            fig.suptitle('Matching - Statistics for various detector+descriptors for FOV {}'.format(args.fov))
        elif args.detector_descriptor is not None:
            legendlist = [int(fov) for det, desc, fov in detdesclist]
            legendtitle = 'FOV'
            fig.suptitle('Matching - Statistics for {}+{} for various FOVs'.format(args.detector_descriptor.split('_')[0], args.detector_descriptor.split('_')[1]))
        else:
            legendlist = ['{}+{} {}'.format(det, desc, fov) for det, desc, fov in detdesclist]
            legendtitle = 'Detector+Descriptor+FOV'
            fig.suptitle('Matching - Statistics for various detector+descriptors and FOVs')

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
        ax4.set_prop_cycle(color=[cm(1. * i / len(detdesclist)) for i in range(len(detdesclist))])
        ax6.set_prop_cycle(color=[cm(1. * i / len(detdesclist)) for i in range(len(detdesclist))])

        handles = []
        for detdesc in detdesclist:
            color = next(ax1._get_lines.prop_cycler)['color']
            h1, = ax1.plot(framediff[detdesc] * attrs["rate"][0], prec[detdesc], color=color)
            h2, = ax1.plot(framediff[detdesc] * attrs["rate"][0], rec[detdesc], linestyle='dashed', color=color)
            handles.append((h1, h2))

        l1 = ax1.legend(handles, legendlist, loc=1, title=legendtitle, fontsize='small')
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

        l1 = ax2.legend(handles, legendlist, loc=1, title=legendtitle, fontsize='small')
        ax2.add_artist(l1)
        ax2.set_xlabel('Frame difference')
        ax2.set_ylabel('Number of matches')
        ax2.set_title('Number of matches over frame difference')

        ax3.set_title('ROC curve')
        ax3.set_xlabel('False positive rate')
        ax3.set_ylabel('True positive rate')
        handles = []
        for detdesc in detdesclist:
            rocc = roc[detdesc][roc[detdesc][:, 0] == 0]
            color = next(ax3._get_lines.prop_cycler)['color']
            for i in range(1, int(rocc[:, 1].max(axis=0))):
                y, x = np.absolute(rocc[rocc[:, 1] == i][:, 2:].T)
                h, = ax3.plot(x, y, color=color)
            handles.append(h)
        l1 = ax3.legend(handles, legendlist, loc=1, title=legendtitle, fontsize='small')
        ax3.add_artist(l1)

        df = pd.DataFrame()
        for d, detdesc in enumerate(detdesclist):
            df = df.append(pd.DataFrame({legendtitle: [legendlist[d] for i in range(len(good_radial_distances[detdesc]))], 'Delta radial distance': good_radial_distances[detdesc][:, 0]}))
        sns.violinplot(x=legendtitle, y='Delta radial distance', data=df, ax=ax4, palette="muted")
        ax4.grid(which='major', linestyle='--', axis='y', linewidth=0.5)
        ax4.set_axisbelow(True)
        ax4.set_title('Distribution of matches over changes in radial distance')

        df = pd.DataFrame()
        for d, detdesc in enumerate(detdesclist):
            matches = np.hstack((good_radial_distances[detdesc][:, 2], bad_radial_distances[detdesc][:, 2]))
            matches /= matches.max()
            df = df.append(pd.DataFrame({legendtitle: [legendlist[d] for i in range(len(matches))], 'Normalized descriptor distance': matches, 'Match': ['Good' for i in range(len(good_radial_distances[detdesc]))] + ['Bad' for i in range(len(bad_radial_distances[detdesc]))]}))
        sns.violinplot(x=legendtitle, y="Normalized descriptor distance", hue="Match", data=df, split=True, ax=ax5, palette="Set2", inner="quart")
        handles, labels = ax5.get_legend_handles_labels()
        ax5.legend(handles=handles[0:], labels=labels[0:], fontsize='small')
        ax5.set_title('Distribution of good and bad matches over descriptor distances')

        if args.fov is not None or args.detector_descriptor is not None:
            ax6.set_title('Descriptor separability over changes in radial distance')
            ax6.set_ylabel('Silhouette coefficient')
            ax6.set_xlabel('Delta radial distance')
            handles = []
            for detdesc in detdesclist:
                num_bins = 50
                num_samples = 1000
                sr = [0 for i in range(num_bins)]
                X_good = [np.array([]) for i in range(num_bins)]
                labels_good = [np.array([]) for i in range(num_bins)]
                X_bad = [np.array([]) for i in range(num_bins)]
                labels_bad = [np.array([]) for i in range(num_bins)]
                valid_good = [False for i in range(num_bins)]
                valid_bad = [False for i in range(num_bins)]
                valid = [False for i in range(num_bins)]
                for row in good_radial_distances[detdesc]:
                    r = int(min(row[0], 0.499999) / (0.5 / num_bins))
                    X_good[r] = np.append(X_good[r], row[1])
                    labels_good[r] = np.append(labels_good[r], 0)
                    valid_good[r] = True
                for row in bad_radial_distances[detdesc]:
                    r = int(min(row[0], 0.499999) / (0.5 / num_bins))
                    X_bad[r] = np.append(X_bad[r], row[1])
                    labels_bad[r] = np.append(labels_bad[r], 1)
                    valid_bad[r] = True
                for i in range(num_bins):
                    valid[i] = valid_good[i] and valid_bad[i]
                    if not valid[i]:
                        continue
                    idx_good = np.arange(len(X_good[i]))
                    idx_bad = np.arange(len(X_bad[i]))
                    if len(X_good[i]) > num_samples:
                        idx_good = np.random.choice(np.arange(len(X_good[i])), num_samples, replace=False)
                    if len(X_bad[i]) > num_samples:
                        idx_bad = np.random.choice(np.arange(len(X_bad[i])), num_samples, replace=False)
                    sr[i] = sklearn.metrics.silhouette_score(np.concatenate((X_good[i][idx_good], X_bad[i][idx_bad])).reshape(-1, 1), np.concatenate((labels_good[i][idx_good], labels_bad[i][idx_bad])), metric = 'l1')
                    # sr[i] = sklearn.metrics.davies_bouldin_score(X[i].reshape(-1, 1), labels[i])
                color = next(ax6._get_lines.prop_cycler)['color']
                h, = ax6.plot([i * 0.5 / num_bins + 0.5 / num_bins / 2 for i in range(0, len(sr)) if valid[i]], [sr[i] for i in range(0, len(sr)) if valid[i]], color=color)
                handles.append(h)
            l1 = ax6.legend(handles, legendlist, loc=1, title=legendtitle, fontsize='small')
            ax6.add_artist(l1)
        else:
            ax6.set_title('Descriptor separability over FOV')
            df = pd.DataFrame()
            for detdesc in detdesclist:
                num_samples = 1000
                labels_good = []
                labels_bad = []
                for row in good_radial_distances[detdesc]:
                    labels_good.append(0)
                for row in bad_radial_distances[detdesc]:
                    labels_bad.append(1)
                labels_good = np.array(labels_good)
                labels_bad = np.array(labels_bad)
                idx_good = np.arange(len(good_radial_distances[detdesc]))
                idx_bad = np.arange(len(bad_radial_distances[detdesc]))
                if len(idx_good) > num_samples:
                    idx_good = np.random.choice(np.arange(len(idx_good)), num_samples, replace=False)
                if len(idx_bad) > num_samples:
                    idx_bad = np.random.choice(np.arange(len(idx_bad)), num_samples, replace=False)
                s = sklearn.metrics.silhouette_score(np.concatenate((good_radial_distances[detdesc][idx_good, 2], bad_radial_distances[detdesc][idx_bad, 2])).reshape(-1, 1), np.concatenate((labels_good[idx_good], labels_bad[idx_bad])), metric = 'l1')
                df = df.append(pd.DataFrame({'Detector+Descriptor': ['{}+{}'.format(detdesc[0], detdesc[1])], 'Silhouette coefficient': [s], 'FOV': [detdesc[2]]}))
            sns.catplot(x='FOV', y='Silhouette coefficient', hue='Detector+Descriptor', data=df, ax=ax6, palette="muted", kind='bar')

        df_pr = pd.DataFrame()
        frames = [1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180]
        for detdesc in detdesclist:
            pr = precrec[detdesc][precrec[detdesc][:, 0] == 0]
            for i in frames:
                # if i >= int(detdesc[2]) and int(detdesc[2]) < 180:
                    # continue
                y, x = np.absolute(pr[pr[:, 1] == i][:, 2:].T)
                df_pr = df_pr.append(pd.DataFrame({'Detector+Descriptor': '{}+{}'.format(detdesc[0], detdesc[1]), 'FOV': detdesc[2], 'Precision': y, 'Recall': x, 'Baseline': i}))
        sns.relplot(y='Precision', x='Recall', hue='Baseline', col='FOV', row='Detector+Descriptor', kind='line', data=df_pr, estimator=None, facet_kws={'margin_titles': True}, legend='full', palette=sns.cubehelix_palette(rot=-0.4, n_colors=len(frames)))

        df_auc = pd.DataFrame()
        for detdesc in detdesclist:
            pr = precrec[detdesc][precrec[detdesc][:, 0] == 0]
            for i in range(1, int(pr[:, 1].max(axis=0))):
                # if i > int(detdesc[2]) and i < 360 - int(detdesc[2]) and int(detdesc[2]) < 180:
                    # df_auc = df_auc.append(pd.DataFrame({'Detector+Descriptor': ['{}+{}'.format(detdesc[0], detdesc[1])], 'FOV': [int(detdesc[2])], 'AUC': [0], 'Baseline': [i]}))
                    # continue
                y, x = np.absolute(pr[pr[:, 1] == i][:, 2:].T)
                auc = np.trapz(np.flip(y), np.flip(x))
                df_auc = df_auc.append(pd.DataFrame({'Detector+Descriptor': ['{}+{}'.format(detdesc[0], detdesc[1])], 'FOV': [int(detdesc[2])], 'AUC': [auc], 'Baseline': [i]}))
        ax_auc = sns.relplot(y='AUC', x='Baseline', hue='FOV', col='Detector+Descriptor', col_wrap=3, kind='line', data=df_auc, estimator=None, legend='full', palette=sns.color_palette('muted', n_colors=df_auc.FOV.unique().shape[0]))

        plt.show()

    else:
        print "[ERROR] Bag filename had no associated results files"

else:
    print "[ERROR] Invalid path specified"

