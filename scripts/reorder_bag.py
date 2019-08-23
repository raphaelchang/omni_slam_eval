import rospy
import rosbag
import argparse
import shutil
import os

def reorder_bag(bagfile):
    orig = os.path.splitext(bagfile)[0] + ".orig.bag"
    shutil.move(bagfile, orig)
    with rosbag.Bag(bagfile, 'w') as outbag:
        for topic, msg, t in rosbag.Bag(orig).read_messages():
            if msg._has_header:
                outbag.write(topic, msg, msg.header.stamp)
            else:
                outbag.write(topic, msg, t)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reorder a bagfile based on header timestamps.')
    parser.add_argument('bagfile', nargs=1, help='input bag file')
    args = parser.parse_args()
    reorder_bag(args.bagfile)

