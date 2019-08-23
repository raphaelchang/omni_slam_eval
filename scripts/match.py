import rospy
import message_filters
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import geometry_msgs
import tf.transformations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import seaborn as sns
import argparse
import rosbag
import reorder_bag
from double_sphere import DoubleSphereModel
import os
from collections import Counter
from parse import parse

parser = argparse.ArgumentParser(description='Feature Matching Evaluation')
parser.add_argument('--bag', help='Path to bag file')
parser.add_argument('--bag_dir', help='Path to bag database')
parser.add_argument('--image_topic', help='Fisheye image topic', default="/unity_ros/Sphere/FisheyeCamera/image_raw")
parser.add_argument('--depth_image_topic', help='Fisheye depth image topic', default="/unity_ros/Sphere/FisheyeDepthCamera/image_raw")
parser.add_argument('--pose_topic', help='Camera pose topic', default="/unity_ros/Sphere/TrueState/pose")
parser.add_argument('--detector', help='Feature detector', default='')
parser.add_argument('--descriptor', help='Feature descriptor', default='')
parser.add_argument('--focal_length', help='DS focal length', type=float)
parser.add_argument('--chi', help='DS chi', type=float)
parser.add_argument('--alpha', help='DS alpha', type=float)
parser.add_argument('--motion', help='Motion type to evaluate')
parser.add_argument('--rate', help='Rate multiplier', type=int, default=1)
args = parser.parse_args()

class Matcher:
    def __init__(self, image_topic, depth_image_topic, pose_topic, detector, descriptor, focal_length, chi, alpha, realtime=False):
        self.camera_model = DoubleSphereModel(focal_length, chi, alpha)

        if detector == 'FAST':
            self.detector = cv2.FastFeatureDetector_create(threshold=50)
        elif detector == 'SIFT':
            self.detector = cv2.xfeatures2d.SIFT_create(nfeatures = 5000)
        elif detector == 'SURF':
            self.detector = cv2.xfeatures2d.SURF_create(hessianThreshold=500)
        elif detector == 'ORB':
            self.detector = cv2.ORB_create(nfeatures = 5000)
        elif detector == 'BRISK':
            self.detector = cv2.BRISK_create(thresh=50)
        elif detector == 'STAR':
            self.detector = cv2.xfeatures2d.StarDetector_create()
        elif detector == 'AKAZE':
            self.detector = cv2.AKAZE_create(threshold=0.001)
        elif detector == 'KAZE':
            self.detector = cv2.KAZE_create(threshold=0.001)
        elif detector == 'AGAST':
            self.detector = cv2.AgastFeatureDetector_create(threshold=65)
        else:
            print 'Invalid feature detector specified'

        self.rgb = False
        self.vgg = False
        if descriptor == 'SIFT':
            self.descriptor = cv2.xfeatures2d.SIFT_create(nfeatures = 5000)
        elif descriptor == 'SURF':
            self.descriptor = cv2.xfeatures2d.SURF_create()
        elif descriptor == 'ORB':
            self.descriptor = cv2.ORB_create(nfeatures = 5000)
        elif descriptor == 'BRISK':
            self.descriptor = cv2.BRISK_create(thresh=50, octaves=3)
        elif descriptor == 'AKAZE':
            self.descriptor = cv2.AKAZE_create()
        elif descriptor == 'KAZE':
            self.descriptor = cv2.KAZE_create()
        elif descriptor == 'FREAK':
            self.descriptor = cv2.xfeatures2d.FREAK_create()
        elif descriptor == 'DAISY':
            self.descriptor = cv2.xfeatures2d.DAISY_create()
        elif descriptor == 'LUCID':
            self.descriptor = cv2.xfeatures2d.LUCID_create()
            self.rgb = True
        elif descriptor == 'LATCH':
            self.descriptor = cv2.xfeatures2d.LATCH_create()
        elif descriptor == 'VGG':
            self.descriptor = cv2.xfeatures2d.VGG_create()
            self.vgg = True
        elif descriptor == 'BOOST':
            scale = 1.5
            if detector == 'KAZE' or detector == 'SURF':
                scale = 6.25
            if detector == 'SIFT':
                scale = 6.75
            elif detector == 'AKAZE' or detector == 'AGAST' or detector == 'FAST' or detector == 'BRISK':
                scale = 5.0
            elif detector == 'ORB':
                scale = 0.75
            self.descriptor = cv2.xfeatures2d.BoostDesc_create(scale_factor=scale)

        np.random.seed(3)
        if descriptor == '' and (detector == 'SIFT' or detector == 'SURF' or detector == 'KAZE') or descriptor == 'SIFT' or descriptor == 'SURF' or descriptor == 'KAZE' or descriptor == 'DAISY' or descriptor == 'LUCID' or descriptor == 'VGG':
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        elif descriptor == '' and (detector == 'ORB' or detector == 'BRISK' or detector == 'AKAZE') or descriptor == 'ORB' or descriptor == 'BRISK' or descriptor == 'AKAZE' or descriptor == 'FREAK' or descriptor == 'LATCH' or descriptor == 'BOOST':
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.color = np.random.randint(0,200,(32768,3))
        self.frame_inx = 0
        self.num_matches = []
        self.recall = []
        self.precision = []
        self.t = tf.TransformListener()
        self.old_tf = geometry_msgs.msg.TransformStamped()
        self.image_pub = rospy.Publisher("/matcher/matches", Image, queue_size=2)
        self.bridge = CvBridge()
        if realtime:
            self.image_sub = message_filters.Subscriber(image_topic, Image)
            self.depth_sub = message_filters.Subscriber(depth_image_topic, Image)
            self.pose_sub = message_filters.Subscriber(pose_topic, PoseStamped)
            self.sync = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub, self.pose_sub], queue_size=3, slop=0.1)
            self.sync.registerCallback(self.on_new_image)

    def on_new_image(self, img, depth, pose):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
            cv_depth_image = self.bridge.imgmsg_to_cv2(depth, "passthrough")
        except CvBridgeError as e:
            print(e)

        frame_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        kp = self.detector.detect(frame_gray, None)
        if len(kp) == 0:
            return
        if self.descriptor is None:
            kp, des = self.detector.compute(frame_gray, kp)
        else:
            if self.rgb:
                kp, des = self.descriptor.compute(cv_image, kp)
            elif self.vgg:
                des = self.descriptor.compute(frame_gray, kp)
            else:
                kp, des = self.descriptor.compute(frame_gray, kp)
        if self.frame_inx == 0:
            rays = np.asarray([[self.camera_model.unproj((np.array(p.pt) / np.array([frame_gray.shape[::-1]]))[0]), self.camera_model.unproj((np.array((p.pt[0], p.pt[1] - p.size / 2.)) / np.array([frame_gray.shape[::-1]]))[0]), self.camera_model.unproj((np.array((p.pt[0] + p.size / 2., p.pt[1])) / np.array([frame_gray.shape[::-1]]))[0]), self.camera_model.unproj((np.array((p.pt[0], p.pt[1] + p.size / 2.)) / np.array([frame_gray.shape[::-1]]))[0]), self.camera_model.unproj((np.array((p.pt[0] - p.size / 2., p.pt[1])) / np.array([frame_gray.shape[::-1]]))[0])] for p in kp])
            depths = np.asarray([[cv_depth_image[int(p.pt[1]), int(p.pt[0])] / 65535. * 500] for p in kp])
            rays = rays * 10000 #depths[:, np.newaxis]
            self.kp0 = kp
            print len(self.kp0)
            self.des0 = des
            self.rays = rays
            self.old_tf = geometry_msgs.msg.TransformStamped()
            self.old_tf.header.frame_id = "map"
            self.old_tf.child_frame_id = "old"
            self.old_tf.header.stamp = pose.header.stamp
            self.old_tf.transform.translation = geometry_msgs.msg.Vector3(pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)
            self.old_tf.transform.rotation = pose.pose.orientation
            self.t.setTransform(self.old_tf)
            self.frame_inx += 1
            return

        m = geometry_msgs.msg.TransformStamped()
        m.header.frame_id = "map"
        m.header.stamp = pose.header.stamp
        m.child_frame_id = "cur"
        m.transform.translation = geometry_msgs.msg.Vector3(pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)
        m.transform.rotation = pose.pose.orientation
        self.t.setTransform(m)
        self.old_tf.header.stamp = pose.header.stamp
        self.t.setTransform(self.old_tf)

        matches = self.matcher.match(des, self.des0)
        matches = sorted(matches, key = lambda x:x.distance)

        p0_tf = []
        for i, p0 in enumerate(self.kp0):
            ray = self.rays[i][0]
            ray_pose = geometry_msgs.msg.PoseStamped()
            ray_pose.pose.position.x = np.ravel(ray)[0]
            ray_pose.pose.position.y = np.ravel(ray)[1]
            ray_pose.pose.position.z = np.ravel(ray)[2]
            ray_pose.header.stamp = pose.header.stamp
            ray_pose.header.frame_id = "old"
            self.t.waitForTransform("cur", "old", pose.header.stamp, rospy.Duration(1))
            new_pose = self.t.transformPose("cur", ray_pose)
            new_ray = [new_pose.pose.position.x, new_pose.pose.position.y, new_pose.pose.position.z]
            proj = self.camera_model.proj(np.array(new_ray))
            if type(proj) == int and proj == -1:
                p0_tf.append(None)
                continue
            x,y = np.ravel(np.multiply(proj, np.mat(frame_gray.shape[::-1])).astype(int)).ravel()
            # furthest_b = 0
            # for j in range(1, 5):
                # b_ray = self.rays[i][j]
                # b_ray_pose = geometry_msgs.msg.PoseStamped()
                # b_ray_pose.pose.position.x = np.ravel(b_ray)[0]
                # b_ray_pose.pose.position.y = np.ravel(b_ray)[1]
                # b_ray_pose.pose.position.z = np.ravel(b_ray)[2]
                # b_ray_pose.header.stamp = pose.header.stamp
                # b_ray_pose.header.frame_id = "old"
                # self.t.waitForTransform("cur", "old", pose.header.stamp, rospy.Duration(1))
                # new_b_pose = self.t.transformPose("cur", b_ray_pose)
                # new_b_ray = [new_b_pose.pose.position.x, new_b_pose.pose.position.y, new_b_pose.pose.position.z]
                # b_proj = self.camera_model.proj(np.array(new_b_ray))
                # if type(b_proj) == int and b_proj == -1:
                    # continue
                # xb,yb = np.ravel(np.multiply(b_proj, np.mat(frame_gray.shape[::-1])).astype(int)).ravel()
                # d = math.sqrt((x-xb)**2+(y-yb)**2)
                # if d > furthest_b:
                    # furthest_b = d
            p0.pt = (x,y)
            p0.size = p0.size #furthest_b * 2
            p0_tf.append(p0)

        # Count correspondences
        corr = 0
        for p1 in kp:
            closest = 99999.
            closest_p0 = None
            for p0 in p0_tf:
                if p0 is None:
                    continue
                x0,y0 = p0.pt
                x1,y1 = p1.pt
                err = math.sqrt((x0-x1)**2+(y0-y1)**2)
                if err < closest:
                    closest = err
                    closest_p0 = p0
            if closest_p0 is None:
                continue
            overlap = cv2.KeyPoint_overlap(closest_p0, p1)
            if overlap > 0.2:
                corr += 1

        mask = np.zeros_like(cv_image)
        num_matches = 0
        for m in matches:
            i0 = m.trainIdx
            i1 = m.queryIdx
            a,b = kp[i1].pt
            c,d = self.kp0[i0].pt
            if p0_tf[i0] is None:
                continue
            e,f = p0_tf[i0].pt
            error = math.sqrt((a-e)**2+(b-f)**2)
            overlap = cv2.KeyPoint_overlap(p0_tf[i0], kp[i1])
            if not np.isnan(error) and overlap > 0.2:
                color = np.array([0, 255 * (overlap), 255 * (1-overlap)])
                mask = cv2.line(mask, (int(a),int(b)), (int(e),int(f)), color, 2)
                cv_image = cv2.circle(cv_image, (int(a),int(b)), int(kp[i1].size / 2.), color, 1)
                cv_image = cv2.circle(cv_image, (int(e),int(f)), int(p0_tf[i0].size / 2.), color, 1)
                num_matches += 1

        img = cv2.add(cv_image, mask)
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
        except CvBridgeError as e:
            print(e)

        self.num_matches.append(num_matches)
        if corr > 0:
            self.recall.append(float(num_matches) / corr)
        else:
            self.recall.append(0)
        if len(matches) > 0:
            self.precision.append(float(num_matches) / len(matches))
        else:
            self.precision.append(0)
        self.frame_inx += 1

    def reset(self):
        self.frame_inx = 0
        self.t.clear()

    def get_stats(self):
        return self.num_matches, self.precision, self.recall

    def plot(self):
        fig = plt.figure()
        fig.suptitle('Feature Matching - Motion type {}, chi={}, alpha={}, focal_length={}, detector={}'.format(args.motion, args.chi, args.alpha, args.focal_length, args.detector))
        ax1 = fig.add_subplot(3, 2, 1)
        ax2 = fig.add_subplot(3, 2, 2)
        ax3 = fig.add_subplot(3, 2, 3)
        ax4 = fig.add_subplot(3, 2, 4)
        ax5 = fig.add_subplot(3, 2, 5)
        ax6 = fig.add_subplot(3, 2, 6)
        # binned_errors = [[0] for i in range(10)]
        # for r, e, _, _, _, _ in self.pixel_errors:
            # # binned_errors[int(min(r, 0.499999) / 0.05)].append(np.log10(e))
            # binned_errors[int(min(r, 0.499999) / 0.05)].append(e)
        # ax1.set_title('Pixel error distribution for various radial distances')
        # ax1.set_ylabel('Pixel error')
        # # ax1.set_yscale('log')
        # ax1.set_xlabel('Radial distance')
        # ax1.set_xticks(np.arange(1, 11))
        # ax1.set_xticklabels(['{}-{}'.format(i * 0.05, (i + 1) * 0.05) for i in range(10)])
        # ax1.set_ylim([0, 20])
        # ax1.violinplot(binned_errors)
        # ax1.plot([i for i in range(1, 11)], [np.array(binned_errors[i]).mean() for i in range(10)], '-o', markersize=4, c='black')
        # # ax1.set_ylim([1e-2, 10 ** 3.5])
        # # axt1 = ax1.twinx()
        # # axt1.violinplot(binned_errors)
        # # axt1.set_yscale('linear')
        # # axt1.set_xlim(0, 9)
        # # axt1.set_ylim([-2, 3.5])
        # # axt1.get_yaxis().set_visible(False)
        # # axt1.plot([i for i in range(1, 11)], [np.array(binned_errors[i]).mean() for i in range(10)], '-o', markersize=4, c='black')
        # ax2.set_title('Pixel errors over frames')
        # # ax2.set_yscale('log')
        # # ax2.set_ylim([1e-2, 10 ** 3.5])
        # ax2.set_ylim([0, 20])
        # ax2.set_ylabel('Pixel error')
        # ax2.set_xlabel('Frame')
        # for i in range(0, self.last_track_inx):
            # err = [(tl, e) for _, e, _, _, inx, tl in self.pixel_errors if inx == i and tl > 0]
            # rs = np.asarray([r for r, _, _, _, inx, finx in self.pixel_errors if inx == i and tl > 0])
            # colors = cm.rainbow(rs * 2)
            # if len(err) > 0:
                # ax2.scatter(*zip(*err), s=2, color=colors.reshape(-1, 4))
                # for j in range(0, len(err) - 1):
                    # # ax2.plot([err[j][0], err[j+1][0]], [err[j][1], err[j+1][1]], color=np.squeeze(colors[j]))
                    # ax2.plot([err[j][0], err[j+1][0]], [err[j][1], err[j+1][1]], color=np.squeeze(colors[j]))
        # tl_err = [(tl, e) for _, e, _, _, _, tl in self.pixel_errors if tl > 0]
        # lim = int(math.ceil(max(tl_err, key=lambda item:item[0])[0]))
        # binned_errors = [[] for i in range(lim + 1)]
        # for _, e, _, _, _, tl in self.pixel_errors:
            # # binned_errors[int(tl)].append(np.log10(e))
            # binned_errors[int(tl)].append(e)
        # ax2.plot([i for i in range(0, lim)], [np.array(binned_errors[i]).mean() for i in range(0, lim)], '-o', markersize=4, c='black')
        # # ax2.violinplot(binned_errors, positions=np.array([i + 0.5 for i in range(0, lim)]))
        # # axt2 = ax2.twinx()
        # # axt2.violinplot(binned_errors, positions=np.array([i + 0.5 for i in range(0, lim)]))
        # # axt2.set_yscale('linear')
        # # axt2.set_ylim([-2, 3.5])
        # # axt2.get_yaxis().set_visible(False)
        # # axt2.set_zorder(1)
        # # axt2.plot([i + 0.5 for i in range(0, lim)], [np.array(binned_errors[i]).mean() for i in range(0, lim)], '-o', markersize=4, c='black')
        # klt_err = [(r, e) for r, _, _, e, _, _ in self.pixel_errors]
        # ax3.set_title('KLT match error over radial distance')
        # ax3.set_ylabel('KLT error')
        # ax3.set_xlabel('Radial distance')
        # ax3.scatter(*zip(*klt_err), alpha=0.2, edgecolor='', s=10)
        # z = np.polyfit(*zip(*klt_err), deg=5)
        # p = np.poly1d(z)
        # tlx = np.asarray(sorted([r for r, _, _, _, _, _ in self.pixel_errors]))
        # ax3.plot(tlx, p(tlx), linewidth=2, color='black')
        # # ax4.set_title('Number of tracks over track lengths')
        # # ax4.set_xlabel('Track length')
        # # ax4.set_ylabel('Count')
        # # # sns.distplot([tl for _, _, tl, _, _ in self.pixel_errors], kde=False, ax=ax4, norm_hist=False)
        # # ax4.hist([tl for _, _, tl, _, _, _ in self.pixel_errors], bins=lim * 4, color='palegreen')
        # ax4.set_title('Number of failures at various radial distances')
        # ax4.set_xlabel('Radial distance')
        # ax4.set_ylabel('Count')
        # ax4.hist([r for r in self.track_failures], bins=[0.0125 * a for a in range(0, 41)], color='palegreen')
        # ax5.set_title('Feature detection distribution over radial distance')
        # ax5.set_xlabel('Radial distance')
        # sns.distplot(self.feature_locs, ax=ax5, norm_hist=True)
        # ax6.set_title('Number of tracks over frames')
        # ax6.set_xlabel('Frame')
        # ax6.set_ylabel('Count')
        # max_finx = max([finx - 2 for _, _, _, _, _, finx in self.pixel_errors])
        # counter = Counter(p[5] - 2 for p in self.pixel_errors)
        # ax6.plot(range(max_finx), [counter[finx] / float(self.orig_track_map[finx + 2]) for finx in range(0, max_finx)], color='blue', linewidth=2)
        # plt.show()
        pass

def process_bag(bagfile, matcher):
    reorder_bag.reorder_bag(bagfile)
    bag = rosbag.Bag(bagfile)
    image_msg = None
    depth_msg = None
    pose_msg = None
    run_next = 0
    matcher.reset()
    skip = 0
    for topic, msg, t in bag.read_messages():
        if topic == args.depth_image_topic:
            depth_msg = msg
            run_next += 1
        elif topic == args.image_topic:
            image_msg = msg
            run_next += 1
        elif run_next >= 2 and topic == args.pose_topic:
            pose_msg = msg
            run_next = 0
            if skip == 0:
                matcher.on_new_image(image_msg, depth_msg, pose_msg)
            skip += 1
            if skip >= args.rate:
                skip = 0

if __name__ == '__main__':
    rospy.init_node('klt_matcher', anonymous=True)
    if args.bag:
        print 'Processing bag...'
        old_sim_time = rospy.get_param('use_sim_time', False)
        rospy.set_param('use_sim_time', True)
        matcher = Matcher(args.image_topic, args.depth_image_topic, args.pose_topic, args.detector, args.descriptor, args.focal_length, args.chi, args.alpha, realtime=False)
        process_bag(args.bag, matcher)
        rospy.set_param('use_sim_time', old_sim_time)
        print 'Generating plots...'
        matcher.plot()
    elif args.bag_dir:
        print 'Processing bags...'
        old_sim_time = rospy.get_param('use_sim_time', False)
        rospy.set_param('use_sim_time', True)
        if args.motion is None and args.chi is None and args.alpha is None and args.focal_length is None:
            if args.detector == '' and args.descriptor == '':
                d_list = [('SIFT','SIFT'), ('SURF','SURF'), ('ORB','ORB'), ('BRISK','BRISK'), ('AKAZE', 'AKAZE'), ('KAZE', 'KAZE'), ('SIFT','FREAK'), ('SIFT','DAISY'), ('SIFT','LUCID'), ('SIFT','LATCH'), ('SIFT','VGG'), ('SIFT','BOOST')]
            if args.descriptor == '':
                descriptor_list = ['SIFT', 'SURF', 'ORB', 'BRISK', 'FREAK', 'DAISY', 'LUCID', 'LATCH', 'VGG', 'BOOST']
                if args.detector == 'AKAZE' or args.detector == 'KAZE':
                    descriptor_list.append('AKAZE')
                    if args.detector == 'KAZE':
                        descriptor_list.append('KAZE')
                if args.detector == 'SIFT':
                    descriptor_list.remove('ORB')
            else:
                descriptor_list = [args.descriptor]
            fig = plt.figure()
            if args.detector == '':
                fig.suptitle('Comparing Feature Descriptors for Various FOVs and Motions'.format(args.detector))
            else:
                fig.suptitle('Comparing Feature Descriptors for Various FOVs and Motions Using {} Detector'.format(args.detector))
            motion_count = 0
            fov_dict = dict()
            last_fov_num = 0
            for motion in os.listdir(args.bag_dir):
                if os.path.isdir(os.path.join(args.bag_dir, motion)):
                    bag_dir = os.path.join(args.bag_dir, motion)
                    motion_count += 1
                    for fov in os.listdir(bag_dir):
                        if os.path.isdir(os.path.join(bag_dir, fov)):
                            if fov not in fov_dict.keys():
                                fov_dict[fov] = last_fov_num
                                last_fov_num += 1
            num_rows = motion_count
            num_cols = last_fov_num
            motion_inx = 0
            for motion in os.listdir(args.bag_dir):
                if os.path.isdir(os.path.join(args.bag_dir, motion)):
                    bag_dir = os.path.join(args.bag_dir, motion)
                    for fov in os.listdir(bag_dir):
                        if os.path.isdir(os.path.join(bag_dir, fov)):
                            chi, alpha, fl = parse('chi{:f}_alpha{:f}_fl{:f}', fov)
                            print "Motion type {}, chi={}, alpha={}, focal_length={}".format(motion, chi, alpha, fl)
                            ax = fig.add_subplot(num_rows, num_cols, motion_inx * num_cols + fov_dict[fov] + 1)
                            cm = plt.get_cmap('nipy_spectral')
                            ax.set_prop_cycle(color=[cm(1. * i / len(descriptor_list)) for i in range(len(descriptor_list))])
                            if args.detector == '':
                                handles = []
                                for det, desc in d_list:
                                    print "Detector+Descriptor {}".format(det + '+' + desc)
                                    matcher = Matcher(args.image_topic, args.depth_image_topic, args.pose_topic, det, desc, fl, chi, alpha, realtime=False)
                                    for filename in os.listdir(os.path.join(bag_dir, fov)):
                                        if filename.endswith('.bag') and not filename.endswith('.orig.bag'):
                                            process_bag(os.path.join(bag_dir, fov, filename), matcher)
                                    nmatch, prec, rec = matcher.get_stats()
                                    color = next(ax._get_lines.prop_cycler)['color']
                                    h1, = ax.plot(np.arange(0, len(prec)) * args.rate, prec, color=color)
                                    h2, = ax.plot(np.arange(0, len(rec)) * args.rate, rec, linestyle='dashed', color=color)
                                    handles.append((h1, h2))
                                l1 = ax.legend(handles, ['{}+{}'.format(det, desc) for det, desc in d_list], loc=1, title='Detector+Descriptor', fontsize='small')
                                l2 = ax.legend([h1, h2], ['Precision', 'Recall'], loc=4, fontsize='small')
                                l2.legendHandles[0].set_color('black')
                                l2.legendHandles[1].set_color('black')
                                ax.add_artist(l1)
                                if fov_dict[fov] == 0:
                                    ax.set_ylabel(motion, size='large')
                                elif fov_dict[fov] == num_cols - 1:
                                    ax.set_ylabel('')
                                    ax.yaxis.set_label_position("right")
                            else:
                                for desc in descriptor_list:
                                    print "Descriptor {}".format(desc)
                                    matcher = Matcher(args.image_topic, args.depth_image_topic, args.pose_topic, args.detector, desc, fl, chi, alpha, realtime=False)
                                    for filename in os.listdir(os.path.join(bag_dir, fov)):
                                        if filename.endswith('.bag') and not filename.endswith('.orig.bag'):
                                            process_bag(os.path.join(bag_dir, fov, filename), matcher)
                                    nmatch, prec, rec = matcher.get_stats()
                                    ax.plot(np.arange(0, len(nmatch)) * args.rate, nmatch, label=desc)
                                ax.legend(loc='best', title='Detector+Descriptor', fontsize='small')
                                if fov_dict[fov] == 0:
                                    ax.set_ylabel(motion, size='large')
                                elif fov_dict[fov] == num_cols - 1:
                                    ax.set_ylabel('Number of matches')
                                    ax.yaxis.set_label_position("right")
                            if motion_inx == 0:
                                ax.set_title('FOV {} degrees'.format(int(round(matcher.camera_model.calc_fov() * 180 / math.pi))))
                            elif motion_inx == num_rows - 1:
                                ax.set_xlabel('Frame number')
                    motion_inx += 1
            rospy.set_param('use_sim_time', old_sim_time)
            print 'Generating plots...'
            plt.show()
        else:
            if args.chi is None:
                args.chi = 0.99
            if args.alpha is None:
                args.alpha = 0.666667
            if args.focal_length is None:
                args.focal_length = 0.288
            if args.motion is None:
                args.motion = 'yaw'
            bag_dir = os.path.join(args.bag_dir, args.motion, "chi{}_alpha{}_fl{}".format(args.chi, args.alpha, args.focal_length))
            matcher = Matcher(args.image_topic, args.depth_image_topic, args.pose_topic, args.detector, args.descriptor, args.focal_length, args.chi, args.alpha, realtime=False)
            for filename in os.listdir(bag_dir):
                if filename.endswith('.bag') and not filename.endswith('.orig.bag'):
                    process_bag(os.path.join(bag_dir, filename), matcher)
            rospy.set_param('use_sim_time', old_sim_time)
            print 'Generating plots...'
            matcher.plot()
    else:
        if args.chi is None:
            args.chi = 0.99
        if args.alpha is None:
            args.alpha = 0.666667
        if args.focal_length is None:
            args.focal_length = 0.288
        matcher = Matcher(args.image_topic, args.depth_image_topic, args.pose_topic, args.detector, args.descriptor, args.focal_length, args.chi, args.alpha, realtime=True)
        rospy.spin()
        print 'Generating plots...'
        matcher.plot()

