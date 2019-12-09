import cv2
import math
from double_sphere import DoubleSphereModel
import numpy as np

image = cv2.imread('sample.png', 1)
camera_model = DoubleSphereModel(0.289, 0.3, 0.6666667)
# camera_model = DoubleSphereModel(0.877, 0, 0)
color = (0, 0, 255)
for z in np.arange(-90, 90, 10):
    prev_pt = None
    for i in np.arange(-180, 180, 0.1):
        ray = np.array([np.sin(i * np.pi / 180), np.cos(i * np.pi / 180) * np.cos(z * np.pi / 180), np.cos(i * np.pi / 180) * np.sin(z * np.pi / 180)])
        pt = camera_model.proj(ray)
        if prev_pt is not None and pt is not -1 and prev_pt is not -1:
            new_pt = pt * 1024
            new_prev_pt = prev_pt * 1024
            cv2.line(image, (int(new_prev_pt[0]), int(new_prev_pt[1])), (int(new_pt[0]), int(new_pt[1])), color, 2)
        prev_pt = pt
cv2.imshow("sample", image)
cv2.waitKey(0)
cv2.imwrite("epipolar.png", image)
