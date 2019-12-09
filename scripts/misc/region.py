import cv2
import math

image = cv2.imread('sample.png', 1)
color = (0, 0, 255)
for i in [0.1, 0.2, 0.3, 0.4, 0.5]:
    cv2.circle(image, (512, 512), int(i * 1024), color, 2)
for i in [0, math.pi / 4., math.pi / 2., 3 * math.pi / 4.]:
    cv2.line(image, (int(-512 * math.sin(i)) + 512, int(-512 * math.cos(i)) + 512), (int(512 * math.sin(i)) + 512, int(512 * math.cos(i)) + 512), color, 2)
cv2.imshow("sample", image)
cv2.waitKey(0)
