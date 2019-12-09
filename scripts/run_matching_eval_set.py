import roslaunch
import os
from parse import parse
import sys
import argparse

parser = argparse.ArgumentParser(description='Run matching evaluation set')
parser.add_argument('working_dir', help='working directory for full motion+fov set evaluation or bag file for single detector+descriptor set evaluation')
parser.add_argument('--motion', type=str, help='motion type for motion set evaluation')
parser.add_argument('--camera_file', type=str, help='camera calibration file for single evaluation')
parser.add_argument("--rate", type=int, help='frame rate multiplier')
args = parser.parse_args()

d_list = [('SIFT','SIFT'), ('SURF','SURF'), ('ORB','ORB'), ('BRISK','BRISK'), ('ORB','ORB+LR'), ('BRISK','BRISK+LR'), ('AKAZE', 'AKAZE'), ('KAZE', 'KAZE'), ('SURF','FREAK'), ('SIFT','DAISY'), ('SIFT','LATCH'), ('SIFT','BOOST')]
det_param_map = dict()
det_param_map['SIFT'] = '{nfeatures: 5000}'
det_param_map['SURF'] = '{hessianThreshold: 1500}'
det_param_map['ORB'] = '{nfeatures: 50}'
det_param_map['BRISK'] = '{thresh: 40}'
det_param_map['AGAST'] = '{threshold: 25}'
det_param_map['AKAZE'] = '{threshold: 0.001}'
det_param_map['KAZE'] = '{threshold: 0.001}'

parent = roslaunch.parent.ROSLaunchParent("", [], is_core=True)
parent.start()

if os.path.isdir(args.working_dir):
    if args.motion is None:
        print ''
        print '==========================================='
        print 'Full motion+FOV dataset matching evaluation'
        print '==========================================='
    else:
        print ''
        print '==========================================='
        print '{} motion dataset matching evaluation'.format(args.motion)
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
                fov_file = os.path.join(args.working_dir, fov + '.yaml')
                for filename in os.listdir(bag_dir):
                    if filename.endswith('.bag') and not filename.endswith('.orig.bag'):
                        bag_file = os.path.abspath(os.path.join(bag_dir, filename))
                        for det, desc in d_list:
                            printstr = "Detector+Descriptor {}+{}".format(det, desc)
                            print ''
                            print '-' * len(printstr)
                            print printstr
                            print '-' * len(printstr)
                            print ''
                            if desc.endswith('+LR'):
                                desc = desc.split('+')[0]
                                sys.argv = ['roslaunch', 'omni_slam_eval', 'matching_eval.launch', 'bag_file:={}'.format(bag_file), 'camera_file:={}'.format(fov_file), 'detector_type:={}'.format(det), 'descriptor_type:={}'.format(desc), 'detector_params:={}'.format(det_param_map[det]), 'rate:={}'.format(args.rate), 'local_unwarping:=true']
                            else:
                                sys.argv = ['roslaunch', 'omni_slam_eval', 'matching_eval.launch', 'bag_file:={}'.format(bag_file), 'camera_file:={}'.format(fov_file), 'detector_type:={}'.format(det), 'descriptor_type:={}'.format(desc), 'detector_params:={}'.format(det_param_map[det]), 'rate:={}'.format(args.rate)]
                            reload(roslaunch)
                            roslaunch.main()
else:
    print ''
    print '=================================================='
    print 'Single run detector+descriptor matching evaluation'
    print '=================================================='
    bag_dir = os.path.abspath(args.working_dir)
    for det, desc in d_list:
        printstr = "Detector+Descriptor {}+{}".format(det, desc)
        print ''
        print '-' * len(printstr)
        print printstr
        print '-' * len(printstr)
        print ''
        sys.argv = ['roslaunch', 'omni_slam_eval', 'matching_eval.launch', 'bag_file:={}'.format(bag_dir), 'detector_type:={}'.format(det), 'descriptor_type:={}'.format(desc), 'detector_params:={}'.format(det_param_map[det]), 'rate:={}'.format(args.rate)]
        if args.camera_file is not None:
            cam_file = os.path.abspath(args.camera_file)
            sys.argv.append('camera_file:={}'.format(cam_file))
        reload(roslaunch)
        roslaunch.main()

print ''
print '==================='
print 'Evaluation complete'
print '==================='
parent.shutdown()
