import h5py
import argparse
import rosbag
from geometry_msgs.msg import PoseStamped
import os
import numpy as np

parser = argparse.ArgumentParser(description='hdf5 odometry results file to bag')
parser.add_argument('results_path', help='odometry results file or bag file')
args = parser.parse_args()

if not os.path.isdir(args.results_path) and args.results_path.endswith('.hdf5'):
    with h5py.File(args.results_path, 'r') as f:
        estimated = f['estimated_poses'][:]
        ground_truth = f['ground_truth_poses'][:]
        attrs = dict(f['attributes'].attrs.items())

    dist = np.linalg.norm(estimated[0, :] - estimated[-1, :])
    print dist
