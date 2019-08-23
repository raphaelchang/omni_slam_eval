import numpy as np
import tf.transformations

class Track:
    def __init__(self, features, poses, colors):
        self.features = features
        self.poses = poses
        self.colors = colors

    def triangulate(self):
        cost = np.zeros((4, 4))
        for ray, pose in zip(self.features, self.poses):
            A = pose - np.dot(np.dot(ray, np.transpose(ray)),  pose)
            cost += np.dot(np.transpose(A), A)
        w, v = np.linalg.eig(cost)
        idx = np.argsort(w)
        ev = v[:, idx[0]]
        ev /= ev[3]
        ev = ev[:3]
        return ev

    def get_average_reprojection_error(self, camera_model, point):
        avg_err = 0
        c = 0
        for ray, pose in zip(self.features, self.poses):
            point_h = np.vstack((np.transpose(np.mat(point)), np.mat([1])))
            point_tf = np.dot(pose, point_h)
            pix = camera_model.proj(point_tf)
            feature = camera_model.proj(ray)
            avg_err += np.linalg.norm(pix - feature)
            c += 1
        return avg_err / c

    def add_view(self, feature, pose, color):
        self.features.append(feature)
        self.poses.append(pose)
        self.colors.append(color)

    def get_color(self):
        if len(self.colors) > 10:
            return sum(self.colors[:10]) / 10.
        return sum(self.colors) / float(len(self.colors))

class Reconstruct:
    def __init__(self, camera_model):
        self.camera_model = camera_model
        self.tracks = dict()

    def add_view_to_track(self, track_id, feature, pose, color):
        if track_id not in self.tracks:
            self.tracks[track_id] = Track([], [], [])
        ray = self.camera_model.unproj(np.array(feature))
        if type(ray) == int and ray == -1:
            return
        t, r = pose
        tmat = tf.transformations.translation_matrix(t)
        rmat = tf.transformations.quaternion_matrix(r)
        pose_mat = np.dot(tmat, rmat)
        ray = np.transpose(ray) / np.linalg.norm(ray)
        self.tracks[track_id].add_view(ray, pose_mat[:3, :], np.array(color, dtype=np.float64))

    def triangulate_tracks(self):
        pts = []
        for track_id, track in self.tracks.iteritems():
            pt = track.triangulate()
            if not np.iscomplexobj(pt):
                pts.append((pt, track.get_color()))
        return pts
