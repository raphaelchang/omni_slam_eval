import h5py
import argparse
import rosbag
from geometry_msgs.msg import PoseStamped

parser = argparse.ArgumentParser(description='hdf5 odometry results file to bag')
parser.add_argument('results_path', help='odometry results file')
parser.add_argument('output_file', help='output bag file')
args = parser.parse_args()

with h5py.File(args.results_path, 'r') as f:
    estimated = f['estimated_poses'][:]
    ground_truth = f['ground_truth_poses'][:]
    attrs = dict(f['attributes'].attrs.items())

bag = rosbag.Bag(args.output_file, 'w')

seq = 0
for estrow, gndrow in zip(estimated, ground_truth):
    estp = PoseStamped()
    estp.header.seq = seq
    estp.header.stamp.secs = seq
    estp.pose.position.x = estrow[0]
    estp.pose.position.y = estrow[1]
    estp.pose.position.z = estrow[2]
    estp.pose.orientation.x = estrow[3]
    estp.pose.orientation.y = estrow[4]
    estp.pose.orientation.z = estrow[5]
    estp.pose.orientation.w = estrow[6]
    gndp = PoseStamped()
    gndp.header.seq = seq
    gndp.header.stamp.secs = seq
    gndp.pose.position.x = gndrow[0]
    gndp.pose.position.y = gndrow[1]
    gndp.pose.position.z = gndrow[2]
    gndp.pose.orientation.x = gndrow[3]
    gndp.pose.orientation.y = gndrow[4]
    gndp.pose.orientation.z = gndrow[5]
    gndp.pose.orientation.w = gndrow[6]
    bag.write('/omni_slam/odometry', estp)
    bag.write('/omni_slam/odometry_truth', gndp)
    seq += 1

bag.close()
