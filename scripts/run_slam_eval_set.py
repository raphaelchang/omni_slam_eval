import roslaunch
import os
from parse import parse
import sys
import argparse

parser = argparse.ArgumentParser(description='Run SLAM evaluation set')
parser.add_argument('working_dir', help='working directory')
parser.add_argument('--motion', type=str, help='motion type for motion set evaluation')
args = parser.parse_args()

parent = roslaunch.parent.ROSLaunchParent("", [], is_core=True)
parent.start()

if os.path.isdir(args.working_dir):
    if args.motion is None:
        print ''
        print '==========================================='
        print 'Full motion+FOV dataset SLAM evaluation'
        print '==========================================='
    else:
        print ''
        print '==========================================='
        print '{} motion dataset SLAM evaluation'.format(args.motion)
        print '==========================================='
    fovs = []
    for yaml in os.listdir(args.working_dir):
        if not os.path.isdir(os.path.join(args.working_dir, yaml)) and yaml.endswith('.yaml'):
            fov = os.path.splitext(os.path.basename(yaml))[0]
            fovs.append(fov)
    fovs.sort(key=int)
    for motion in os.listdir(args.working_dir):
        if os.path.isdir(os.path.join(args.working_dir, motion)):
            if args.motion is not None and motion != args.motion:
                continue
            bag_dir = os.path.join(args.working_dir, motion)
            for fov in fovs:
                if args.motion is None:
                    printstr = "Motion type {}, FOV {}".format(motion, fov)
                else:
                    printstr = "FOV {}".format(fov)
                print ''
                print '-' * len(printstr)
                print printstr
                print '-' * len(printstr)
                print ''
                fov_file = os.path.join(args.working_dir, fov + '.yaml')
                for filename in os.listdir(bag_dir):
                    if filename.endswith('.bag') and not filename.endswith('.orig.bag'):
                        bag_file = os.path.abspath(os.path.join(bag_dir, filename))
                        sys.argv = ['roslaunch', 'omni_slam_eval', 'slam_eval.launch', 'bag_file:={}'.format(bag_file), 'camera_file:={}'.format(fov_file)]
                        reload(roslaunch)
                        roslaunch.main()
    print ''
    print '==================='
    print 'Evaluation complete'
    print '==================='

else:
    print '[ERROR] Invalid path specified'

parent.shutdown()


