import h5py
import argparse
import rosbag
from geometry_msgs.msg import PoseStamped
import os

parser = argparse.ArgumentParser(description='hdf5 odometry results file to bag')
parser.add_argument('results_path', help='odometry results file or bag file')
parser.add_argument('--output_file', help='output bag file')
parser.add_argument('--topic_name', help='pose topic')
args = parser.parse_args()

if not os.path.isdir(args.results_path) and args.results_path.endswith('.hdf5'):
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
        bag.write(args.topic_name, estp)
        bag.write('Tracking module', gndp)
        seq += 1

    bag.close()

elif not os.path.isdir(args.results_path) and args.results_path.endswith('.bag'):
    bag = rosbag.Bag(args.results_path + '.evo.bag', 'w')

    bagname = os.path.splitext(os.path.basename(args.results_path))[0]
    gnd_recorded = False
    for filename in os.listdir(os.path.dirname(args.results_path)):
        if filename.startswith(bagname) and (filename.endswith('.odometry.hdf5') or filename.endswith('.slam.hdf5')):
            fovstr = filename.split('.')[1]
            with h5py.File(os.path.join(os.path.dirname(args.results_path), filename), 'r') as f:
                attrs = dict(f['attributes'].attrs.items())
                # estimated = f['bundle_adjusted_poses'][:]
                estimated = f['estimated_poses'][:]
                ground_truth = f['ground_truth_poses'][:]
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
                    bag.write(fovstr, estp)
                    if not gnd_recorded:
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
                        bag.write('ground_truth', gndp)
                    seq += 1
                gnd_recorded = True

    bag.close()

else:
    print "[ERROR] Invalid path specified"
