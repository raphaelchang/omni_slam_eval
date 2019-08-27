import rospy
import message_filters
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
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
from reconstruct import Reconstruct
from double_sphere import DoubleSphereModel
import os
from collections import Counter
from parse import parse
import time
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pcl2
import struct

parser = argparse.ArgumentParser(description='KLT Evaluation')
parser.add_argument('--bag', help='Path to bag file')
parser.add_argument('--bag_dir', help='Path to bag database')
parser.add_argument('--image_topic', help='Fisheye image topic', default="/unity_ros/Sphere/FisheyeCamera/image_raw")
parser.add_argument('--depth_image_topic', help='Fisheye depth image topic', default="/unity_ros/Sphere/FisheyeDepthCamera/image_raw")
parser.add_argument('--pose_topic', help='Camera pose topic', default="/unity_ros/Sphere/TrueState/pose")
parser.add_argument('--detector', help='Feature detector algorithm', default='GFTT')
parser.add_argument('--focal_length', help='DS focal length', type=float)
parser.add_argument('--chi', help='DS chi', type=float)
parser.add_argument('--alpha', help='DS alpha', type=float)
parser.add_argument('--motion', help='Motion type to evaluate')
parser.add_argument('--baseline', help='Use baseline KLT algorithm', action='store_true')
parser.add_argument('--incremental', help='Use incremental tracking', action='store_true')
parser.add_argument('--rate', help='Rate multiplier', type=int, default=1)
args = parser.parse_args()

class KLTTracker:
    def __init__(self, image_topic, depth_image_topic, pose_topic, detector, focal_length, chi, alpha, realtime=False):
        self.camera_model = DoubleSphereModel(focal_length, chi, alpha)

        self.half_window_theta = self.camera_model.calc_fov() / 36. # radians
        self.window_size = 129
        self.num_scales = 4
        self.min_num_pts_region = 10
        inc_flag = cv2.OPTFLOW_USE_INITIAL_FLOW
        if args.incremental:
            inc_flag = 0
        self.lk_params = dict(winSize = ((self.window_size - 1) / (2 ** self.num_scales), (self.window_size - 1) / (2 ** self.num_scales)),
                         maxLevel = self.num_scales,
                         criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01),
                         flags = inc_flag)# | cv2.OPTFLOW_LK_GET_MIN_EIGENVALS, minEigThreshold=0.5)

        if detector == 'GFTT':
            self.detector = cv2.GFTTDetector_create(maxCorners=2000, qualityLevel=0.005, minDistance=5, blockSize=5, useHarrisDetector=False)
        elif detector == 'FAST':
            self.detector = cv2.FastFeatureDetector_create(threshold=65)
        elif detector == 'SIFT':
            self.detector = cv2.xfeatures2d.SIFT_create(nfeatures = 2000)
        elif detector == 'SURF':
            self.detector = cv2.xfeatures2d.SURF_create()
        elif detector == 'ORB':
            self.detector = cv2.ORB_create(nfeatures = 2000)
        elif detector == 'BRISK':
            self.detector = cv2.BRISK_create(thresh=90, octaves=3)
        elif detector == 'STAR':
            self.detector = cv2.xfeatures2d.StarDetector_create()
        elif detector == 'AKAZE':
            self.detector = cv2.AKAZE_create()
        elif detector == 'AGAST':
            self.detector = cv2.AgastFeatureDetector_create(threshold=65)
        else:
            print 'Invalid feature detector specified'

        np.random.seed(10)
        self.color = np.random.randint(0,200,(32768,3))
        self.frame_inx = 0
        self.t = tf.TransformListener()
        self.old_tf = geometry_msgs.msg.TransformStamped()
        self.image_pub = rospy.Publisher("/klt/tracked", Image, queue_size=2)
        self.pc_pub = rospy.Publisher("/klt/reconstructed", PointCloud2, queue_size=2)
        self.bridge = CvBridge()
        if realtime:
            self.image_sub = message_filters.Subscriber(image_topic, Image)
            self.depth_sub = message_filters.Subscriber(depth_image_topic, Image)
            self.pose_sub = message_filters.Subscriber(pose_topic, PoseStamped)
            self.sync = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub, self.pose_sub], queue_size=3, slop=0.1)
            self.sync.registerCallback(self.on_new_image)
        self.pixel_errors = []
        self.feature_locs = []
        self.track_failures = []
        self.dead_track_lengths = []
        self.orig_track_map = dict()
        self.last_track_inx = 0
        self.orig_num_tracks = 0

        self.reconstruct = Reconstruct(self.camera_model)

    def detect_in_region(self, img, r_min, r_max, t_min, t_max):
        start_r = r_min * img.shape[0]
        end_r = r_max * img.shape[1]
        mask = np.zeros(img.shape, dtype=np.uint8)
        y,x = np.ogrid[0:img.shape[0], 0:img.shape[1]]
        cond1 = start_r ** 2 <= (x - img.shape[1] / 2 + 0.5) ** 2 + (y - img.shape[0] / 2 + 0.5) ** 2
        cond2 = (x - img.shape[1] / 2 + 0.5) ** 2 + (y - img.shape[0] / 2 + 0.5) ** 2 < end_r ** 2
        cond3 = t_min <= np.arctan2(y - img.shape[0] / 2 + 0.5, x - img.shape[1] / 2 + 0.5)
        cond4 = np.arctan2(y - img.shape[0] / 2 + 0.5, x - img.shape[1] / 2 + 0.5) < t_max
        cond = np.logical_and(cond1, cond2)
        cond = np.logical_and(cond, cond3)
        cond = np.logical_and(cond, cond4)
        mask[cond] = 255
        return self.detector.detect(img, mask)

    def on_new_image(self, img, depth, pose):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
            cv_depth_image = self.bridge.imgmsg_to_cv2(depth, "passthrough")
        except CvBridgeError as e:
            print(e)

        def detect():
            kp = []
            for r in np.linspace(0, 0.5, 5, endpoint=False):
                for t in np.linspace(-math.pi, math.pi, 8, endpoint=False):
                    kp += self.detect_in_region(self.old_img, r, r + 0.1, t, t + math.pi / 4)
            # kp = np.random.choice(kp, 300, replace=False)
            self.p0 = np.asarray([[[p.pt[0], p.pt[1]]] for p in kp]).astype(np.float32)
            self.p00 = np.copy(self.p0)
            self.rays = np.asarray([self.camera_model.unproj(((p[0]+0.5) / np.array([self.old_img.shape[::-1]]))[0]) for p in self.p0])
            # depths = np.asarray([[min([self.old_depth[int(p[0][1]), int(p[0][0])], self.old_depth[int(p[0][1])-1, int(p[0][0])], self.old_depth[int(p[0][1])+1, int(p[0][0])], self.old_depth[int(p[0][1]), int(p[0][0])-1], self.old_depth[int(p[0][1]), int(p[0][0])+1], self.old_depth[int(p[0][1])-1, int(p[0][0])-1], self.old_depth[int(p[0][1])-1, int(p[0][0])+1], self.old_depth[int(p[0][1])+1, int(p[0][0])+1], self.old_depth[int(p[0][1])+1, int(p[0][0])-1]]) / 65535. * 500] for p in self.p0])
            depths = np.asarray([[self.old_depth[int(p[0][1]), int(p[0][0])] / 65535. * 500] for p in self.p0])
            self.rays = self.rays * depths[:, np.newaxis]
            self.old_pose = self.prev_pose
            self.mask = np.zeros_like(cv_image)
            self.track_lengths = np.asarray([[[0.]] for p in self.p0])
            self.indices = np.asarray([[[i + self.last_track_inx]] for i in range(len(self.p0))])
            self.track_start_frame = np.asarray([[[self.frame_inx]] for p in self.p0])
            self.last_track_inx += len(self.p0)
            self.orig_num_tracks = len(self.p0)
            self.feature_locs += [np.sqrt(((p[0][0] / self.old_img.shape[1]) - 0.5) ** 2 + ((p[0][1] / self.old_img.shape[0]) - 0.5) ** 2) for p in self.p0]
            self.old_errors = np.zeros((len(self.p0), 1))

        if self.frame_inx == 0:
            self.frame_inx += 1
            return
        if self.frame_inx == 1:
            self.old_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            self.old_depth = cv_depth_image
            self.prev_pose = pose.pose
            self.old_tf = geometry_msgs.msg.TransformStamped()
            self.old_tf.header.frame_id = "map"
            self.old_tf.child_frame_id = "old"
            self.old_tf.header.stamp = pose.header.stamp
            self.old_tf.transform.translation = geometry_msgs.msg.Vector3(pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)
            self.old_tf.transform.rotation = pose.pose.orientation
            self.t.setTransform(self.old_tf)
            detect()
            self.frame_inx += 1
            return

        frame_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # if self.frame_inx % 80 == 0:
            # self.old_tf = geometry_msgs.msg.TransformStamped()
            # self.old_tf.header.frame_id = "map"
            # self.old_tf.child_frame_id = "old"
            # self.old_tf.header.stamp = pose.header.stamp
            # self.old_tf.transform.translation = geometry_msgs.msg.Vector3(self.prev_pose.position.x, self.prev_pose.position.y, self.prev_pose.position.z)
            # self.old_tf.transform.rotation = self.prev_pose.orientation
            # self.t.setTransform(self.old_tf)
            # detect()

        bins = dict()
        for kp in self.p00:
            x,y = np.ravel(kp)
            r_bin = min(int(math.sqrt((x / self.old_img.shape[1] - 0.5) ** 2 + (y / self.old_img.shape[0] - 0.5) ** 2) * 10), 4)
            t_bin = min(int((np.arctan2(y / self.old_img.shape[0] - 0.5, x / self.old_img.shape[1] - 0.5) + math.pi) / (math.pi / 4)), 7)
            if r_bin not in bins:
                bins[r_bin] = dict()
            if t_bin not in bins[r_bin]:
                bins[r_bin][t_bin] = 0
            bins[r_bin][t_bin] += 1
        for r_bin, t_bins in bins.iteritems():
            for t_bin, count in t_bins.iteritems():
                if count < self.min_num_pts_region:
                    kp = self.detect_in_region(self.old_img, r_bin / 10., r_bin / 10. + 0.1, t_bin * math.pi / 4 - math.pi, t_bin * math.pi / 4 - 3 * math.pi / 4)
                    if len(kp) == 0:
                        continue
                    p0_new = np.asarray([[[p.pt[0], p.pt[1]]] for p in kp]).astype(np.float32)
                    self.p0 = np.concatenate((self.p0, p0_new))
                    self.p00 = np.concatenate((self.p00, p0_new))
                    rays = np.asarray([self.camera_model.unproj(((p[0]+0.5) / np.array([self.old_img.shape[::-1]]))[0]) for p in p0_new])
                    # depths = np.asarray([[min([self.old_depth[int(p[0][1]), int(p[0][0])], self.old_depth[int(p[0][1])-1, int(p[0][0])], self.old_depth[int(p[0][1])+1, int(p[0][0])], self.old_depth[int(p[0][1]), int(p[0][0])-1], self.old_depth[int(p[0][1]), int(p[0][0])+1], self.old_depth[int(p[0][1])-1, int(p[0][0])-1], self.old_depth[int(p[0][1])-1, int(p[0][0])+1], self.old_depth[int(p[0][1])+1, int(p[0][0])+1], self.old_depth[int(p[0][1])+1, int(p[0][0])-1]]) / 65535. * 500] for p in p0_new])
                    depths = np.asarray([[self.old_depth[int(p[0][1]), int(p[0][0])] / 65535. * 500] for p in p0_new])
                    try:
                        t = self.t.getLatestCommonTime("old", "cur")
                        for ray in rays * depths[:, np.newaxis]:
                            ray_pose = geometry_msgs.msg.PoseStamped()
                            ray_pose.pose.position.x = np.ravel(ray)[0]
                            ray_pose.pose.position.y = np.ravel(ray)[1]
                            ray_pose.pose.position.z = np.ravel(ray)[2]
                            ray_pose.header.stamp = self.t.getLatestCommonTime("old", "cur")
                            ray_pose.header.frame_id = "cur"
                            # self.t.waitForTransform("old", "cur", pose.header.stamp, rospy.Duration(1))
                            new_pose = self.t.transformPose("old", ray_pose)
                            new_ray = np.array([[[new_pose.pose.position.x, new_pose.pose.position.y, new_pose.pose.position.z]]])
                            self.rays = np.concatenate((self.rays, new_ray))
                    except:
                        self.rays = np.concatenate((self.rays, rays * depths[:, np.newaxis]))
                    self.track_lengths = np.concatenate((self.track_lengths, np.asarray([[[0.]] for p in kp])))
                    self.indices = np.concatenate((self.indices, np.asarray([[[i + self.last_track_inx]] for i in range(len(kp))])))
                    self.track_start_frame = np.concatenate((self.track_start_frame, np.asarray([[[self.frame_inx]] for p in kp])))
                    self.last_track_inx += len(kp)
                    self.orig_num_tracks += len(kp)
                    self.old_errors = np.concatenate((self.old_errors, np.zeros((len(kp), 1))))

        if self.frame_inx not in self.orig_track_map.keys():
            self.orig_track_map[self.frame_inx] = 0
        self.orig_track_map[self.frame_inx] += self.orig_num_tracks

        if len(self.p0) == 0:
            return

        if args.baseline:
            if args.incremental:
                p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_img, frame_gray, self.p00, None, **self.lk_params)
                p2, st2, err2 = cv2.calcOpticalFlowPyrLK(frame_gray, self.old_img, p1, None, **self.lk_params)
                fb_err = np.linalg.norm(self.p00-p2, axis=2)
            else:
                p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_img, frame_gray, self.p0, np.copy(self.p00), **self.lk_params)
                st2 = np.copy(st)
                fb_err = np.zeros_like(np.linalg.norm(self.p00, axis=2))
        else:
            if args.incremental:
                p1, st, err = self.unwarp_and_track(self.old_img, frame_gray, self.p00)
                st2 = np.copy(st)
                fb_err = np.zeros_like(np.linalg.norm(self.p00, axis=2))
            else:
                p1, st, err = self.unwarp_and_track(self.old_img, frame_gray, self.p0, np.copy(self.p00))
                st2 = np.copy(st)
                fb_err = np.zeros_like(np.linalg.norm(self.p00, axis=2))

        m = geometry_msgs.msg.TransformStamped()
        m.header.frame_id = "map"
        m.header.stamp = pose.header.stamp
        m.child_frame_id = "cur"
        m.transform.translation = geometry_msgs.msg.Vector3(pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)
        m.transform.rotation = pose.pose.orientation
        self.t.setTransform(m)
        self.old_tf.header.stamp = pose.header.stamp
        self.t.setTransform(self.old_tf)

        for i, (s, ray) in enumerate(zip(st, self.rays)):
            if s == 1:
                continue
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
                continue
            x,y = np.ravel(proj)
            self.track_failures.append(np.sqrt((x-0.5)**2+(y-0.5)**2))
            self.dead_track_lengths.append(self.frame_inx - np.ravel(self.track_start_frame[i]))

        st2 = np.squeeze(st2)
        good_new = p1[st==1]
        good_old = self.p00[st==1]
        good_rays = self.rays[st==1]
        good_errors = err[st==1]
        good_old_errors = self.old_errors[st==1]
        fb_err = fb_err[st==1]
        fb_err[:] = 0
        if st2.ndim == 0:
            st2 = np.array([st2])
        st2[:] = 1
        st2 = np.vstack(st2)[st==1]
        self.indices = self.indices[st==1].reshape(-1, 1, 1)
        self.track_lengths = self.track_lengths[st==1].reshape(-1, 1, 1)
        self.track_start_frame = self.track_start_frame[st==1].reshape(-1, 1, 1)

        # quat_inv = [pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, -pose.pose.orientation.w]
        # quat_old = [self.old_pose.orientation.x, self.old_pose.orientation.y, self.old_pose.orientation.z, self.old_pose.orientation.w]
        # q1 = tf.transformations.quaternion_multiply(quat_old, quat_inv)
        errors = []
        errors_bad = []
        for i, (new,old,ray,klt_err,fb,old_err) in enumerate(zip(good_new, good_old, good_rays, good_errors, fb_err, good_old_errors)):
            # ray = tf.transformations.unit_vector(ray)
            # q2 = list(ray)
            # q2.append(0.0)
            # new_ray = tf.transformations.quaternion_multiply(
                # tf.transformations.quaternion_multiply(q1, q2),
                # tf.transformations.quaternion_conjugate(q1))[:-1]
            ray_pose = geometry_msgs.msg.PoseStamped()
            ray_pose.pose.position.x = np.ravel(ray)[0]
            ray_pose.pose.position.y = np.ravel(ray)[1]
            ray_pose.pose.position.z = np.ravel(ray)[2]
            ray_pose.header.stamp = pose.header.stamp
            ray_pose.header.frame_id = "old"
            self.t.waitForTransform("cur", "old", pose.header.stamp, rospy.Duration(1))
            new_pose = self.t.transformPose("cur", ray_pose)
            new_ray = [new_pose.pose.position.x, new_pose.pose.position.y, new_pose.pose.position.z]
            a,b = new.ravel()
            c,d = old.ravel()
            proj = self.camera_model.proj(np.array(new_ray))
            old_proj = self.camera_model.proj(np.ravel(ray))
            if type(proj) == int and proj == -1:
                errors_bad.append(1)
                self.dead_track_lengths.append(self.frame_inx - np.ravel(self.track_start_frame[i]))
                continue
            if type(old_proj) != int:
                self.track_lengths[i] += np.linalg.norm(proj - old_proj)
            e,f = np.ravel(np.multiply(proj, np.mat(self.old_img.shape[::-1])).astype(int)).ravel()
            x,y = np.ravel(proj)
            error = np.sqrt((a-e)**2+(b-f)**2)
            if fb >= 1 or st2[i] == 0 or klt_err >= 20 or error - old_err > 3:
                self.track_failures.append(np.sqrt((x-0.5)**2+(y-0.5)**2))
                if error - old_err > 3:
                    errors_bad.append(1)
                else:
                    errors_bad.append(0)
                self.dead_track_lengths.append(self.frame_inx - np.ravel(self.track_start_frame[i]))
                continue
            errors_bad.append(0)
            errors.append(error)
            if error < 100:
                self.pixel_errors.append((np.sqrt((x-0.5)**2+(y-0.5)**2), error, self.track_lengths[i][0][0], klt_err, self.indices[i], self.frame_inx, np.ravel(self.track_start_frame[i])))

            t, r = self.t.lookupTransform("cur", "map", pose.header.stamp)
            if int(b) >= 0 and int(b) < self.old_img.shape[0] and int(a) >= 0 and int(a) < self.old_img.shape[1]:
                color = cv_image[int(b), int(a)]
            else:
                color = [0, 0, 0]
            self.reconstruct.add_view_to_track(self.indices[i].ravel()[0], ((a+0.5) / self.old_img.shape[1], (b+0.5) / self.old_img.shape[0]), (t, r), color)

            if not np.isnan(error):
                # self.mask = cv2.line(self.mask, (a,b), (c,d), self.color[i].tolist(), 2)
                # cv_image = cv2.circle(cv_image, (a,b), 5, self.color[i].tolist(), -1)
                color = np.array([0, 255 * (1 - error / 10), 255 * (error / 10)])
                self.mask = cv2.line(self.mask, (int(a),int(b)), (int(c),int(d)), color, 1)
                cv_image = cv2.circle(cv_image, (int(a),int(b)), 1, color, -1)
                if e >= 0 and e < cv_image.shape[0] and f >= 0 and f < cv_image.shape[1]:
                    cv_image = cv2.circle(cv_image, (e,f), 3, self.color[i].tolist(), -1)

        img = cv2.add(cv_image, self.mask)
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
        except CvBridgeError as e:
            print(e)

        errors_bad = np.array(errors_bad)
        good_new = good_new[st2==1]
        good_old = good_old[st2==1]
        good_rays = good_rays[st2==1]
        good_errors = good_errors[st2==1]
        fb_err = fb_err[st2==1]
        good_errors = good_errors[fb_err<1]
        errors_bad = errors_bad[st2==1][fb_err<1][good_errors<20]
        good_new = good_new[fb_err<1][good_errors<20][errors_bad==0]
        good_old = good_old[fb_err<1][good_errors<20][errors_bad==0]
        good_rays = good_rays[fb_err<1][good_errors<20][errors_bad==0]
        self.track_lengths = self.track_lengths[st2==1][fb_err<1][good_errors<20][errors_bad==0].reshape(-1, 1, 1)
        self.indices = self.indices[st2==1][fb_err<1][good_errors<20][errors_bad==0].reshape(-1, 1, 1)
        self.track_start_frame = self.track_start_frame[st2==1][fb_err<1][good_errors<20][errors_bad==0].reshape(-1, 1, 1)
        self.p0 = self.p0[st==1][st2==1][fb_err<1][good_errors<20][errors_bad==0].reshape(-1,1,2)
        good_errors = good_errors[good_errors<20][errors_bad==0]

        if args.incremental:
            self.old_img = frame_gray.copy()
        self.old_depth = cv_depth_image.copy()
        self.p00 = good_new.reshape(-1,1,2)
        self.rays = good_rays.reshape(-1,1,3)
        self.prev_pose = pose.pose
        if len(errors) > 0:
            self.old_errors = np.vstack(np.array(errors))
        self.frame_inx += 1

    def unwarp_and_track(self, prev_frame, cur_frame, points, priors=None):
        p1 = np.zeros(points.shape)
        st = np.zeros((points.shape[0], 1))
        err = np.zeros((points.shape[0], 1))
        inx = 0
        for p in points:
            ray = np.ravel(self.camera_model.unproj((p[0] / np.array([prev_frame.shape[::-1]]))[0]))
            ray = tf.transformations.unit_vector(ray)
            if priors is not None:
                pr = priors[inx][0]
                ray_pr = np.ravel(self.camera_model.unproj((pr / np.array([cur_frame.shape[::-1]]))[0]))
                ray_pr = tf.transformations.unit_vector(ray_pr)
            zero_ray = np.array([1.0, 0.0, 0.0])
            # q_final = tf.transformations.quaternion_about_axis(math.acos(np.dot(zero_ray, ray)), tuple(tf.transformations.unit_vector(np.cross(zero_ray, ray))))
            q_final_yaw = tf.transformations.quaternion_about_axis(math.atan2(zero_ray[0] * ray[1] - zero_ray[1] * ray[0], zero_ray[0] * ray[0] + zero_ray[1] * ray[1]), [0.0, 0.0, 1.0])
            q_final_pitch = tf.transformations.quaternion_about_axis(math.atan2(np.linalg.norm(np.cross(np.array([ray[0], ray[1], 0]), ray)), ray[0] * ray[0] + ray[1] * ray[1]), tuple(tf.transformations.unit_vector(np.cross(np.array([ray[0], ray[1], 0]), ray))))
            if priors is not None:
                q_final_yaw_pr = tf.transformations.quaternion_about_axis(math.atan2(zero_ray[0] * ray_pr[1] - zero_ray[1] * ray_pr[0], zero_ray[0] * ray_pr[0] + zero_ray[1] * ray_pr[1]), [0.0, 0.0, 1.0])
                q_final_pitch_pr = tf.transformations.quaternion_about_axis(math.atan2(np.linalg.norm(np.cross(np.array([ray_pr[0], ray_pr[1], 0]), ray_pr)), ray_pr[0] * ray_pr[0] + ray_pr[1] * ray_pr[1]), tuple(tf.transformations.unit_vector(np.cross(np.array([ray_pr[0], ray_pr[1], 0]), ray_pr))))
            prev_patch = np.zeros((self.window_size, self.window_size, 1))
            cur_patch = np.zeros((3 * self.window_size, 3 * self.window_size, 1))
            row_num = 0
            for d1 in np.linspace(-math.tan(self.half_window_theta), math.tan(self.half_window_theta), self.window_size):
                col_num = 0
                for d2 in np.linspace(-math.tan(self.half_window_theta), math.tan(self.half_window_theta), self.window_size):
                    q1 = tf.transformations.quaternion_from_euler(0, math.atan2(d1, 1), -math.atan2(d2, 1), 'rxyz')
                    q2 = [1.0, 0.0, 0.0, 0.0]
                    delta_ray = tf.transformations.quaternion_multiply(
                        tf.transformations.quaternion_multiply(q1, q2),
                        tf.transformations.quaternion_conjugate(q1))
                    new_ray_temp = tf.transformations.quaternion_multiply(
                        tf.transformations.quaternion_multiply(q_final_yaw, delta_ray),
                        tf.transformations.quaternion_conjugate(q_final_yaw))
                    new_ray = tf.transformations.quaternion_multiply(
                        tf.transformations.quaternion_multiply(q_final_pitch, new_ray_temp),
                        tf.transformations.quaternion_conjugate(q_final_pitch))[:-1]
                    x, y = np.ravel(self.camera_model.proj(np.array(new_ray) * 1000)) * prev_frame.shape[::-1]
                    if np.isnan(x) or np.isnan(y):
                        prev_patch[row_num, col_num] = 0
                        continue
                    ix = int(x)
                    iy = int(y)
                    x0 = cv2.borderInterpolate(ix, prev_frame.shape[1], cv2.BORDER_REFLECT_101)
                    x1 = cv2.borderInterpolate(ix + 1, prev_frame.shape[1], cv2.BORDER_REFLECT_101)
                    y0 = cv2.borderInterpolate(iy, prev_frame.shape[0], cv2.BORDER_REFLECT_101)
                    y1 = cv2.borderInterpolate(iy + 1, prev_frame.shape[0], cv2.BORDER_REFLECT_101)
                    dx = x - ix
                    dy = y - iy
                    prev_patch[row_num, col_num] = (prev_frame[y0, x0] * (1. - dx) + prev_frame[y0, x1] * dx) * (1. - dy) + (prev_frame[y1, x0] * (1. - dx) + prev_frame[y1, x1] * dx) * dy
                    col_num += 1
                row_num += 1
            row_num = 0
            for d1 in np.linspace(-math.tan(3 * self.half_window_theta), math.tan(3 * self.half_window_theta), 3 * self.window_size):
                col_num = 0
                for d2 in np.linspace(-math.tan(3 * self.half_window_theta), math.tan(3 * self.half_window_theta), 3 * self.window_size):
                    q1 = tf.transformations.quaternion_from_euler(0, math.atan2(d1, 1), -math.atan2(d2, 1), 'rxyz')
                    q2 = [1.0, 0.0, 0.0, 0.0]
                    delta_ray = tf.transformations.quaternion_multiply(
                        tf.transformations.quaternion_multiply(q1, q2),
                        tf.transformations.quaternion_conjugate(q1))
                    if priors is not None:
                        new_ray_temp_pr = tf.transformations.quaternion_multiply(
                            tf.transformations.quaternion_multiply(q_final_yaw_pr, delta_ray),
                            tf.transformations.quaternion_conjugate(q_final_yaw_pr))
                        new_ray_pr = tf.transformations.quaternion_multiply(
                            tf.transformations.quaternion_multiply(q_final_pitch_pr, new_ray_temp_pr),
                            tf.transformations.quaternion_conjugate(q_final_pitch_pr))[:-1]
                    else:
                        new_ray_temp = tf.transformations.quaternion_multiply(
                            tf.transformations.quaternion_multiply(q_final_yaw, delta_ray),
                            tf.transformations.quaternion_conjugate(q_final_yaw))
                        new_ray = tf.transformations.quaternion_multiply(
                            tf.transformations.quaternion_multiply(q_final_pitch, new_ray_temp),
                            tf.transformations.quaternion_conjugate(q_final_pitch))[:-1]
                    if priors is not None:
                        xr, yr = np.ravel(self.camera_model.proj(np.array(new_ray_pr) * 1000)) * cur_frame.shape[::-1]
                        if np.isnan(xr) or np.isnan(yr):
                            cur_patch[row_num, col_num] = 0
                            continue
                    else:
                        x, y = np.ravel(self.camera_model.proj(np.array(new_ray) * 1000)) * prev_frame.shape[::-1]
                        if np.isnan(x) or np.isnan(y):
                            cur_patch[row_num, col_num] = 0
                            continue
                    if priors is not None:
                        ixr = int(xr)
                        iyr = int(yr)
                        x0r = cv2.borderInterpolate(ixr, cur_frame.shape[1], cv2.BORDER_REFLECT_101)
                        x1r = cv2.borderInterpolate(ixr + 1, cur_frame.shape[1], cv2.BORDER_REFLECT_101)
                        y0r = cv2.borderInterpolate(iyr, cur_frame.shape[0], cv2.BORDER_REFLECT_101)
                        y1r = cv2.borderInterpolate(iyr + 1, cur_frame.shape[0], cv2.BORDER_REFLECT_101)
                        dxr = xr - ixr
                        dyr = yr - iyr
                        cur_patch[row_num, col_num] = (cur_frame[y0r, x0r] * (1. - dxr) + cur_frame[y0r, x1r] * dxr) * (1. - dyr) + (cur_frame[y1r, x0r] * (1. - dxr) + cur_frame[y1r, x1r] * dxr) * dyr
                    else:
                        ix = int(x)
                        iy = int(y)
                        x0 = cv2.borderInterpolate(ix, prev_frame.shape[1], cv2.BORDER_REFLECT_101)
                        x1 = cv2.borderInterpolate(ix + 1, prev_frame.shape[1], cv2.BORDER_REFLECT_101)
                        y0 = cv2.borderInterpolate(iy, prev_frame.shape[0], cv2.BORDER_REFLECT_101)
                        y1 = cv2.borderInterpolate(iy + 1, prev_frame.shape[0], cv2.BORDER_REFLECT_101)
                        dx = x - ix
                        dy = y - iy
                        cur_patch[row_num, col_num] = (cur_frame[y0, x0] * (1. - dx) + cur_frame[y0, x1] * dx) * (1. - dy) + (cur_frame[y1, x0] * (1. - dx) + cur_frame[y1, x1] * dx) * dy
                    col_num += 1
                row_num += 1
            prev_patch = cv2.copyMakeBorder(prev_patch, top=self.window_size, bottom=self.window_size, left=self.window_size, right=self.window_size, borderType=cv2.BORDER_CONSTANT, value=[0])
            new_pt, st[inx], err[inx] = cv2.calcOpticalFlowPyrLK(np.uint8(prev_patch), np.uint8(cur_patch), np.asarray([[[3 * self.window_size / 2., 3 * self.window_size / 2.]]]).astype(np.float32), np.asarray([[[3 * self.window_size / 2., 3 * self.window_size / 2.]]]).astype(np.float32), **self.lk_params)
            # cv2.cornerSubPix(np.uint8(cur_patch), new_pt, (1, 1), (-1, -1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 0.003))
            xn, yn = np.ravel(new_pt)
            # cur_patch = cv2.circle(cur_patch, (int(xn),int(yn)), 3, (255, 0, 0), -1)
            # prev_patch = cv2.circle(prev_patch, (3*self.window_size/2, 3*self.window_size/2), 3, (255, 0, 0), -1)
            # cv2.imshow("patch_old", np.uint8(prev_patch))
            # cv2.imshow("patch_new", np.uint8(cur_patch))
            # cv2.waitKey(0)
            d1n = -3 * self.half_window_theta + (yn - 0.5) / (3 * self.window_size - 1.) * 6 * self.half_window_theta
            d2n = -3 * self.half_window_theta + (xn - 0.5) / (3 * self.window_size - 1.) * 6 * self.half_window_theta
            q1n = tf.transformations.quaternion_from_euler(0, d1n, -d2n, 'rxyz')
            q2n = [1.0, 0.0, 0.0, 0.0]
            delta_ray_n = tf.transformations.quaternion_multiply(
                tf.transformations.quaternion_multiply(q1n, q2n),
                tf.transformations.quaternion_conjugate(q1n))
            if priors is None:
                new_ray_n_temp = tf.transformations.quaternion_multiply(
                    tf.transformations.quaternion_multiply(q_final_yaw, delta_ray_n),
                    tf.transformations.quaternion_conjugate(q_final_yaw))
                new_ray_n = tf.transformations.quaternion_multiply(
                    tf.transformations.quaternion_multiply(q_final_pitch, new_ray_n_temp),
                    tf.transformations.quaternion_conjugate(q_final_pitch))[:-1]
            else:
                new_ray_n_temp = tf.transformations.quaternion_multiply(
                    tf.transformations.quaternion_multiply(q_final_yaw_pr, delta_ray_n),
                    tf.transformations.quaternion_conjugate(q_final_yaw_pr))
                new_ray_n = tf.transformations.quaternion_multiply(
                    tf.transformations.quaternion_multiply(q_final_pitch_pr, new_ray_n_temp),
                    tf.transformations.quaternion_conjugate(q_final_pitch_pr))[:-1]
            p1[inx] = np.asarray([[np.ravel(self.camera_model.proj(np.array(new_ray_n) * 1000)) * cur_frame.shape[::-1]]])
            inx += 1

        return p1, st, err

    def reset(self):
        self.frame_inx = 0
        self.t.clear()

    def main_plot(self, ax):
        ax.hist([[r for r, e, _, _, _, _ in self.pixel_errors if e <= 5], [r for r, e, _, _, _, _ in self.pixel_errors if 5 < e <= 20], [r for r, e, _, _, _, _ in self.pixel_errors if 20 < e <= 100], [r for r in self.track_failures]], bins=[i * 0.05 for i in range(11)], alpha=0.5, label=['<5', '5-20', '20-100', 'Failures'], stacked=False)
        ax.legend(loc='best', title='Pixel error', fontsize='x-small')

    def plot(self):
        points = self.reconstruct.triangulate_tracks()
        pcl_pts = []
        for xyz, bgr in points:
            pt = [xyz[0], xyz[1], xyz[2], struct.unpack('I', struct.pack('BBBB', int(bgr[0]), int(bgr[1]), int(bgr[2]), 255))[0]]
            pcl_pts.append(pt)
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('rgb', 12, PointField.UINT32, 1)]
        header = Header()
        header.frame_id = 'map'
        pc2 = pcl2.create_cloud(header, fields, pcl_pts)
        while not rospy.is_shutdown():
            pc2.header.stamp = rospy.Time.now()
            self.pc_pub.publish(pc2)
            rospy.sleep(0.1)

        fig = plt.figure()
        fig.suptitle('KLT Tracking - Motion type {}, chi={}, alpha={}, focal_length={}, detector={}'.format(args.motion, args.chi, args.alpha, args.focal_length, args.detector))
        ax1 = fig.add_subplot(4, 1, 1)
        ax2 = fig.add_subplot(4, 1, 2)
        ax3 = fig.add_subplot(4, 1, 3)
        # ax4 = fig.add_subplot(3, 2, 4, sharex=ax2)
        ax4 = fig.add_subplot(4, 1, 4)
        # ax5 = fig.add_subplot(3, 2, 5)
        # ax6 = fig.add_subplot(3, 2, 6)
        binned_errors = [[0] for i in range(10)]
        for r, e, _, _, _, _, _ in self.pixel_errors:
            # binned_errors[int(min(r, 0.499999) / 0.05)].append(np.log10(e))
            binned_errors[int(min(r, 0.499999) / 0.05)].append(e)
        ax1.set_title('Pixel error distribution for various radial distances')
        ax1.set_ylabel('Pixel error')
        # ax1.set_yscale('log')
        ax1.set_xlabel('Radial distance')
        ax1.set_xticks(np.arange(1, 11))
        ax1.set_xticklabels(['{}-{}'.format(i * 0.05, (i + 1) * 0.05) for i in range(10)])
        ax1.set_ylim([0, 20])
        ax1.violinplot(binned_errors)
        ax1.plot([i for i in range(1, 11)], [np.array(binned_errors[i]).mean() for i in range(10)], '-o', markersize=4, c='black')
        # ax1.set_ylim([1e-2, 10 ** 3.5])
        # axt1 = ax1.twinx()
        # axt1.violinplot(binned_errors)
        # axt1.set_yscale('linear')
        # axt1.set_xlim(0, 9)
        # axt1.set_ylim([-2, 3.5])
        # axt1.get_yaxis().set_visible(False)
        # axt1.plot([i for i in range(1, 11)], [np.array(binned_errors[i]).mean() for i in range(10)], '-o', markersize=4, c='black')
        ax2.set_title('Pixel errors over track lifetime')
        # ax2.set_yscale('log')
        # ax2.set_ylim([1e-2, 10 ** 3.5])
        ax2.set_ylim([0, 20])
        ax2.set_ylabel('Pixel error')
        ax2.set_xlabel('Frame')
        for i in range(0, self.last_track_inx):
            err = [(tl-sf, e) for _, e, _, _, inx, tl, sf in self.pixel_errors if inx == i and tl > 0]
            rs = np.asarray([r for r, _, _, _, inx, finx, _ in self.pixel_errors if inx == i and tl > 0])
            colors = cm.rainbow(rs * 2)
            if len(err) > 0:
                ax2.scatter(*zip(*err), s=2, color=colors.reshape(-1, 4))
                for j in range(0, len(err) - 1):
                    # ax2.plot([err[j][0], err[j+1][0]], [err[j][1], err[j+1][1]], color=np.squeeze(colors[j]))
                    ax2.plot([err[j][0], err[j+1][0]], [err[j][1], err[j+1][1]], color=np.squeeze(colors[j]))
        tl_err = [(tl-sf, e) for _, e, _, _, _, tl, sf in self.pixel_errors if tl > 0]
        lim = int(math.ceil(max(tl_err, key=lambda item:item[0])[0]))
        binned_errors = [[] for i in range(lim + 1)]
        for _, e, _, _, _, tl, sf in self.pixel_errors:
            # binned_errors[int(tl)].append(np.log10(e))
            binned_errors[int(tl-sf)].append(e)
        ax2.plot([i for i in range(0, lim)], [np.array(binned_errors[i]).mean() for i in range(0, lim)], '-o', markersize=4, c='black')
        # ax2.violinplot(binned_errors, positions=np.array([i + 0.5 for i in range(0, lim)]))
        # axt2 = ax2.twinx()
        # axt2.violinplot(binned_errors, positions=np.array([i + 0.5 for i in range(0, lim)]))
        # axt2.set_yscale('linear')
        # axt2.set_ylim([-2, 3.5])
        # axt2.get_yaxis().set_visible(False)
        # axt2.set_zorder(1)
        # axt2.plot([i + 0.5 for i in range(0, lim)], [np.array(binned_errors[i]).mean() for i in range(0, lim)], '-o', markersize=4, c='black')
        # klt_err = [(r, e) for r, _, _, e, _, _ in self.pixel_errors]
        # ax3.set_title('KLT match error over radial distance')
        # ax3.set_ylabel('KLT error')
        # ax3.set_xlabel('Radial distance')
        # ax3.scatter(*zip(*klt_err), alpha=0.2, edgecolor='', s=10)
        # z = np.polyfit(*zip(*klt_err), deg=5)
        # p = np.poly1d(z)
        # tlx = np.asarray(sorted([r for r, _, _, _, _, _ in self.pixel_errors]))
        # ax3.plot(tlx, p(tlx), linewidth=2, color='black')
        # ax4.set_title('Number of tracks over track lengths')
        # ax4.set_xlabel('Track length')
        # ax4.set_ylabel('Count')
        # # sns.distplot([tl for _, _, tl, _, _ in self.pixel_errors], kde=False, ax=ax4, norm_hist=False)
        # ax4.hist([tl for _, _, tl, _, _, _ in self.pixel_errors], bins=lim * 4, color='palegreen')
        # ax4.set_title('Number of failures at various radial distances')
        # ax4.set_xlabel('Radial distance')
        # ax4.set_ylabel('Count')
        # ax4.hist([r for r in self.track_failures], bins=[0.0125 * a for a in range(0, 41)], color='palegreen')
        # ax5.set_title('Feature detection distribution over radial distance')
        # ax5.set_xlabel('Radial distance')
        # sns.distplot(self.feature_locs, ax=ax5, norm_hist=True)
        ax3.set_title('Number of tracks over frames')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Count')
        max_finx = max([finx - 2 for _, _, _, _, _, finx, _ in self.pixel_errors])
        counter = Counter(p[5] - 2 for p in self.pixel_errors)
        # ax6.plot(range(max_finx), [counter[finx] / float(self.orig_track_map[finx + 2]) for finx in range(0, max_finx)], color='blue', linewidth=2)
        ax3.plot(range(max_finx), [counter[finx] for finx in range(0, max_finx)], color='blue', linewidth=2)
        for i in range(len(self.p00)):
            self.dead_track_lengths.append(self.frame_inx - np.ravel(self.track_start_frame[i]))
        tls = np.array(self.dead_track_lengths)
        ax4.set_title('Distribution of track lifetimes')
        ax4.set_xlabel('Lifetime (frames)')
        ax4.set_ylabel('Count')
        ax4.hist(tls, color='c', bins=max_finx / 10)
        ax4.axvline(tls.mean(), color='k', linestyle='dashed', linewidth=1)
        ax4.axvline(np.median(tls), color='k', linestyle='dotted', linewidth=1)
        # print "Average track lifetime: {} frames".format(float(sum(self.dead_track_lengths)) / len(self.dead_track_lengths))
        plt.show()

def process_bag(bagfile, tracker):
    reorder_bag.reorder_bag(bagfile)
    bag = rosbag.Bag(bagfile)
    image_msg = None
    depth_msg = None
    pose_msg = None
    run_next = 0
    tracker.reset()
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
                if image_msg is not None and depth_msg is not None and pose_msg is not None:
                    tracker.on_new_image(image_msg, depth_msg, pose_msg)
            skip += 1
            if skip >= args.rate:
                skip = 0

if __name__ == '__main__':
    rospy.init_node('klt_tracker', anonymous=True)
    if args.bag:
        print 'Processing bag...'
        old_sim_time = rospy.get_param('use_sim_time', False)
        rospy.set_param('use_sim_time', True)
        tracker = KLTTracker(args.image_topic, args.depth_image_topic, args.pose_topic, args.detector, args.focal_length, args.chi, args.alpha, realtime=False)
        process_bag(args.bag, tracker)
        rospy.set_param('use_sim_time', old_sim_time)
        print 'Generating plots...'
        tracker.plot()
    elif args.bag_dir:
        print 'Processing bags...'
        old_sim_time = rospy.get_param('use_sim_time', False)
        rospy.set_param('use_sim_time', True)
        if args.motion is None and args.chi is None and args.alpha is None and args.focal_length is None:
            fig = plt.figure()
            fig.suptitle('KLT Tracking Performance for Various FOVs and Motions')
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
                            tracker = KLTTracker(args.image_topic, args.depth_image_topic, args.pose_topic, args.detector, fl, chi, alpha, realtime=False)
                            for filename in os.listdir(os.path.join(bag_dir, fov)):
                                if filename.endswith('.bag') and not filename.endswith('.orig.bag'):
                                    process_bag(os.path.join(bag_dir, fov, filename), tracker)
                            ax = fig.add_subplot(num_rows, num_cols, motion_inx * num_cols + fov_dict[fov] + 1)
                            tracker.main_plot(ax)
                            if motion_inx == 0:
                                ax.set_title('FOV {} degrees'.format(int(round(tracker.camera_model.calc_fov() * 180 / math.pi))))
                            elif motion_inx == num_rows - 1:
                                ax.set_xlabel('Radial distance')
                            if fov_dict[fov] == 0:
                                ax.set_ylabel(motion, size='large')
                            elif fov_dict[fov] == num_cols - 1:
                                ax.set_ylabel('Number of tracks')
                                ax.yaxis.set_label_position("right")
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
            tracker = KLTTracker(args.image_topic, args.depth_image_topic, args.pose_topic, args.detector, args.focal_length, args.chi, args.alpha, realtime=False)
            for filename in os.listdir(bag_dir):
                if filename.endswith('.bag') and not filename.endswith('.orig.bag'):
                    process_bag(os.path.join(bag_dir, filename), tracker)
            rospy.set_param('use_sim_time', old_sim_time)
            print 'Generating plots...'
            tracker.plot()
    else:
        if args.chi is None:
            args.chi = 0.99
        if args.alpha is None:
            args.alpha = 0.666667
        if args.focal_length is None:
            args.focal_length = 0.288
        tracker = KLTTracker(args.image_topic, args.depth_image_topic, args.pose_topic, args.detector, args.focal_length, args.chi, args.alpha, realtime=True)
        rospy.spin()
        print 'Generating plots...'
        tracker.plot()

