import matplotlib.pyplot as plt
from double_sphere import DoubleSphereModel
import numpy as np
import math
import seaborn as sns
import pandas as pd

fig, axs = plt.subplots(1, 2, figsize=(10, 3))
sns.set()

params = [(205.824, -0.055, 0.577, 250), (250.88, -0.179, 0.591, 195), (366.592, -0.005, 0.6453, 160), (348.16, -0.271, 0.555, 120), (491.52, -0.237, 0.55, 90), (898.048, 0.0, 0.0, 60)]

df = pd.DataFrame()
df_ratio = pd.DataFrame()
for f, c, a, fov in params:
    camera_model = DoubleSphereModel(f / 1024., c, a)
    last_ang = 0
    first = True
    p = 0
    for i in np.linspace(0.5, 1.0, 1024):
        ray = camera_model.unproj(np.array([i, 0.5]))
        ang = np.arctan2(ray[0, 0], ray[0, 1])
        if first:
            last_ang = ang
            first = False
            continue
        ray_vert = camera_model.unproj(np.array([i, 0.5 - 0.5 / 1024.]))
        ang_vert = np.arctan2(ray_vert[0, 2], np.sqrt(ray_vert[0, 0] ** 2 + ray_vert[0, 1] ** 2))
        df = df.append(pd.DataFrame({'Direction': ['Radial', 'Tangential'], 'Angular resolution (degrees / pixel)': [(ang - last_ang) * 180 / math.pi, ang_vert * 180 / math.pi], 'Distance from image center (pixels)': p, 'FOV': fov}))
        df_ratio = df_ratio.append(pd.DataFrame({'Distance from image center (pixels)': [p], 'Local aspect ratio': (ang - last_ang) / ang_vert, 'FOV': [fov]}))
        last_ang = ang
        p += 1

sns.lineplot(ax=axs[0], x='Distance from image center (pixels)', y='Angular resolution (degrees / pixel)', hue='FOV', style='Direction', data=df, legend='full', estimator=None, palette=sns.color_palette("deep", n_colors=len(params)))
sns.lineplot(ax=axs[1], x='Distance from image center (pixels)', y='Local aspect ratio', hue='FOV', data=df_ratio, legend='full', estimator=None, palette=sns.color_palette("deep", n_colors=len(params)))
axs[0].set_xlim([0, 1500])
axs[1].set_xlim([0, 1400])
plt.show()

