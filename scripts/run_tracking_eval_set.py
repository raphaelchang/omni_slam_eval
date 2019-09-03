import roslaunch
import os
from parse import parse
import sys
import argparse

parser = argparse.ArgumentParser(description='Run tracking evaluation set')
parser.add_argument('working_dir', help='working directory')
parser.add_argument("--rate", type=int, help='frame rate multiplier')
args = parser.parse_args()

parent = roslaunch.parent.ROSLaunchParent("", [], is_core=True)
parent.start()

if os.path.isdir(args.working_dir):
    print ''
    print '==========================================='
    print 'Full motion+FOV dataset tracking evaluation'
    print '==========================================='
    for motion in os.listdir(args.working_dir):
        if os.path.isdir(os.path.join(args.working_dir, motion)):
            bag_dir = os.path.join(args.working_dir, motion)
            for fov in os.listdir(bag_dir):
                if os.path.isdir(os.path.join(bag_dir, fov)):
                    chi, alpha, fx, fy, cx, cy = parse('chi{:f}_alpha{:f}_fx{:f}_fy{:f}_cx{:f}_cy{:f}', fov)
                    printstr = "Motion type {}, chi={}, alpha={}, fx={}, fy={}, cx={}, cy={}".format(motion, chi, alpha, fx, fy, cx, cy)
                    print ''
                    print '-' * len(printstr)
                    print printstr
                    print '-' * len(printstr)
                    print ''
                    fov_dir = os.path.join(bag_dir, fov)
                    for filename in os.listdir(fov_dir):
                        if filename.endswith('.bag') and not filename.endswith('.orig.bag'):
                            bag_file = os.path.abspath(os.path.join(fov_dir, filename))
                            sys.argv = ['roslaunch', 'omni_slam_eval', 'tracking_eval.launch', 'bag_file:={}'.format(bag_file), 'results_file:={}.tracking.hdf5'.format(bag_file), 'camera_params:={{fx: {}, fy: {}, cx: {}, cy: {}, chi: {}, alpha: {}}}'.format(fx, fy, cx, cy, chi, alpha), 'rate:={}'.format(args.rate)]
                            reload(roslaunch)
                            roslaunch.main()
    print ''
    print '==================='
    print 'Evaluation complete'
    print '==================='

else:
    print '[ERROR] Invalid path specified'

parent.shutdown()
