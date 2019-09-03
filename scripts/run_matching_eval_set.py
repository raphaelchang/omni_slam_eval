import roslaunch
import os
from parse import parse
import sys
import argparse

parser = argparse.ArgumentParser(description='Run matching evaluation set')
parser.add_argument('working_dir', help='working directory for full motion+fov set evaluation or bag file for single detector+descriptor set evaluation')
parser.add_argument("--rate", type=int, help='frame rate multiplier')
args = parser.parse_args()

d_list = [('SIFT','SIFT'), ('SURF','SURF'), ('ORB','ORB'), ('BRISK','BRISK'), ('AKAZE', 'AKAZE'), ('KAZE', 'KAZE'), ('SIFT','FREAK'), ('SIFT','DAISY'), ('SIFT','LUCID'), ('SIFT','LATCH'), ('SIFT','VGG'), ('SIFT','BOOST')]
det_param_map = dict()
det_param_map['SIFT'] = '{nfeatures: 2000}'
det_param_map['SURF'] = '{hessianThreshold: 500}'
det_param_map['ORB'] = '{nfeatures: 100}'
det_param_map['BRISK'] = '{thresh: 30}'
det_param_map['AGAST'] = '{threshold: 10}'
det_param_map['AKAZE'] = '{threshold: 0.001}'
det_param_map['KAZE'] = '{threshold: 0.001}'

parent = roslaunch.parent.ROSLaunchParent("", [], is_core=True)
parent.start()

if os.path.isdir(args.working_dir):
    print ''
    print '==========================================='
    print 'Full motion+FOV dataset matching evaluation'
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
                    fov_dir = os.path.join(bag_dir, fov)
                    for filename in os.listdir(fov_dir):
                        if filename.endswith('.bag') and not filename.endswith('.orig.bag'):
                            bag_file = os.path.abspath(os.path.join(fov_dir, filename))
                            for det, desc in d_list:
                                printstr = "Detector+Descriptor {}+{}".format(det, desc)
                                print ''
                                print '-' * len(printstr)
                                print printstr
                                print '-' * len(printstr)
                                print ''
                                sys.argv = ['roslaunch', 'omni_slam_eval', 'matching_eval.launch', 'bag_file:={}'.format(bag_file), 'results_file:={}.{}_{}.matching.hdf5'.format(bag_file, det, desc), 'camera_params:={{fx: {}, fy: {}, cx: {}, cy: {}, chi: {}, alpha: {}}}'.format(fx, fy, cx, cy, chi, alpha), 'detector_type:={}'.format(det), 'descriptor_type:={}'.format(desc), 'detector_params:={}'.format(det_param_map[det]), 'rate:={}'.format(args.rate)]
                                reload(roslaunch)
                                roslaunch.main()
else:
    print ''
    print '=================================================='
    print 'Single run detector+descriptor matching evaluation'
    print '=================================================='
    bag_dir = os.path.abspath(args.working_dir)
    par_dir = os.path.basename(os.path.dirname(bag_dir))
    parsed = parse('chi{:f}_alpha{:f}_fx{:f}_fy{:f}_cx{:f}_cy{:f}', par_dir)
    if parsed is not None:
        chi, alpha, fx, fy, cx, cy = parsed
    for det, desc in d_list:
        printstr = "Detector+Descriptor {}+{}".format(det, desc)
        print ''
        print '-' * len(printstr)
        print printstr
        print '-' * len(printstr)
        print ''
        sys.argv = ['roslaunch', 'omni_slam_eval', 'matching_eval.launch', 'bag_file:={}'.format(bag_dir), 'results_file:={}.{}_{}.matching.hdf5'.format(bag_dir, det, desc), 'detector_type:={}'.format(det), 'descriptor_type:={}'.format(desc), 'detector_params:={}'.format(det_param_map[det]), 'rate:={}'.format(args.rate)]
        if parsed is not None:
            sys.argv.append('camera_params:={{fx: {}, fy: {}, cx: {}, cy: {}, chi: {}, alpha: {}}}'.format(fx, fy, cx, cy, chi, alpha))
        reload(roslaunch)
        roslaunch.main()

print ''
print '==================='
print 'Evaluation complete'
print '==================='
parent.shutdown()
