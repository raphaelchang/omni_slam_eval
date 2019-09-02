import matplotlib.pyplot as plt
from double_sphere import DoubleSphereModel
import numpy as np
import math

fig, ax = plt.subplots()

params = [(0.289, 0.3, 0.6666667), (0.5, 0, 0), (0.36, 0.66, 0.006), (0.213, -0.2, 0.59)]

for f, c, a in params:
    camera_model = DoubleSphereModel(f, c, a)
    dt = []
    last_ang = 0
    first = True
    for i in np.linspace(0.5, 1.0, 1024):
        ray = camera_model.unproj(np.array([i, 0.5]))
        ang = np.arctan2(ray[0, 0], ray[0, 1])
        if first:
            last_ang = ang
            first = False
            continue
        dt.append(ang - last_ang)
        last_ang = ang

    ax.plot(np.arange(0, len(dt)), np.array(dt) * 180 / math.pi, label='{} FOV'.format(camera_model.calc_fov() * 180 / math.pi))

plt.legend(loc='best')
plt.show()
