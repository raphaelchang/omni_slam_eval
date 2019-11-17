import matplotlib.pyplot as plt
from double_sphere import DoubleSphereModel
import numpy as np
import math

fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()

params = [(205.824, -0.055, 0.577), (287, 0, 0.647), (250.88, -0.179, 0.591), (348.16, -0.271, 0.555), (898.048, 0.0, 0.0)]

for f, c, a in params:
    camera_model = DoubleSphereModel(f / 1024., c, a)
    dt = []
    dt_vert = []
    last_ang = 0
    first = True
    for i in np.linspace(0.5, 1.0, 1024):
        ray = camera_model.unproj(np.array([i, 0.5]))
        ang = np.arctan2(ray[0, 0], ray[0, 1])
        if first:
            last_ang = ang
            first = False
            continue
        ray_vert = camera_model.unproj(np.array([i, 0.5 - 0.5 / 1024.]))
        ang_vert = np.arctan2(ray_vert[0, 2], np.sqrt(ray_vert[0, 0] ** 2 + ray_vert[0, 1] ** 2))
        dt.append(ang - last_ang)
        dt_vert.append(ang_vert)
        last_ang = ang

    ax.plot(np.arange(0, len(dt)), (np.array(dt) * 180 / math.pi), label='{} FOV'.format(camera_model.calc_fov() * 180 / math.pi))
    ax.plot(np.arange(0, len(dt_vert)), (np.array(dt_vert) * 180 / math.pi), label='{} FOV'.format(camera_model.calc_fov() * 180 / math.pi))
    ax2.plot(np.arange(0, len(dt_vert)), (np.array(dt) * 180 / math.pi) / (np.array(dt_vert) * 180 / math.pi), label='{} FOV'.format(camera_model.calc_fov() * 180 / math.pi))
    ax.set_xlabel('Distance from image center (pixels)')
    ax.set_ylabel('Angular resolution (degrees / pixel)')
    ax2.set_xlabel('Distance from image center (pixels)')
    ax2.set_ylabel('Local aspect ratio')

ax.legend(loc='best')
ax2.legend(loc='best')
plt.show()
