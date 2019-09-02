import numpy as np
import math

class DoubleSphereModel:
    def __init__(self, focal_length, chi, alpha):
        self.cx = 0.5
        self.cy = 0.5
        self.focal_length = focal_length
        self.chi = chi
        self.alpha = alpha
        self.fov = self.calc_fov()

    def unproj(self, uv):
        cx = self.cx
        cy = self.cy
        fx = self.focal_length
        fy = self.focal_length
        alpha = self.alpha
        chi = self.chi

        mx = (uv[0] - cx) / fx
        my = (uv[1] - cy) / fy

        r2 = mx * mx + my * my
        beta1 = 1. - (2. * alpha - 1) * r2
        if beta1 < 0:
            return -1
        mz = (1. - alpha * alpha * r2) / (alpha * math.sqrt(beta1) + 1. - alpha)

        beta2 = mz * mz + (1. - chi * chi) * r2

        if beta2 < 0:
            return -1

        fisheye_ray = (mz * chi + math.sqrt(mz * mz + (1. - chi * chi) * r2)) / float(mz * mz + r2) * np.mat([mx, mz, -my]) - np.mat([0, chi, 0])

        return fisheye_ray

    def proj(self, xyz):
        cx = self.cx
        cy = self.cy
        fx = self.focal_length
        fy = self.focal_length
        alpha = self.alpha
        chi = self.chi

        x = xyz[0]
        y = -xyz[2]
        z = xyz[1]

        d1 = math.sqrt(x * x + y * y + z * z)
        d2 = math.sqrt(x * x + y * y + (chi * d1 + z) ** 2)
        w1 = 0
        if alpha <= 0.5 and alpha > 0:
            w1 = alpha / (1 - alpha)
        elif alpha > 0.5:
            w1 = (1 - alpha) / alpha
        w2 = (w1 + chi) / math.sqrt(2 * w1 * chi + chi ** 2 + 1)
        # if z <= -w2 * d1:
            # return -1

        resx = fx * x / (alpha * d2 + (1 - alpha) * (chi * d1 + z)) + cx
        resy = fy * y / (alpha * d2 + (1 - alpha) * (chi * d1 + z)) + cy
        if resx > 2 * self.cx or resx < 0 or resy > 2 * self.cy or resy < 0:
            return -1
        res = np.array([resx, resy])
        if alpha > 0.5:
            theta = math.pi / 2 - self.fov / 2
            if z <= math.sin(theta) * d1:
                return -1
            # r2 = ((res[0, 0] - self.cx) / fx) ** 2 + ((res[0, 1] - self.cy) / fy) ** 2
            # if r2 > 1. / (2 * alpha - 1):
                # return -1
        return res

    def calc_fov(self):
        # if self.alpha <= 0.5:
            # mx = self.cx / self.focal_length
            # r2 = mx ** 2
            # mz = (1 - self.alpha ** 2 * r2) / (self.alpha * np.sqrt(1 - (2 * self.alpha - 1) * r2) + 1 - self.alpha)
            # beta = (mz * self.chi + np.sqrt(mz ** 2 + (1 - self.chi ** 2) / r2)) / (mz ** 2 + r2)
        # else:
            # mz = (1 - self.alpha ** 2 / (2 * self.alpha - 1)) / (1 - self.alpha)
            # mx = np.sqrt(1 / (2 * self.alpha - 1))
            # beta = (mz * self.chi + np.sqrt(mz ** 2 + (1 - self.chi ** 2) / (2 * self.alpha - 1))) / (mz ** 2 + (1 / (2 * self.alpha - 1)) ** 2)
        mx = self.cx / self.focal_length
        if self.alpha > 0.5:
            r2 = min(mx ** 2, 1 / (2 * self.alpha - 1))
        else:
            r2 = mx ** 2
        mz = (1 - self.alpha ** 2 * r2) / (self.alpha * math.sqrt(1 - (2 * self.alpha - 1) * r2) + 1 - self.alpha)
        beta = (mz * self.chi + math.sqrt(mz ** 2 + (1 - self.chi ** 2) * r2)) / (mz ** 2 + r2)
        return 2 * (math.pi / 2 - math.atan2(beta * mz - self.chi, beta * mx))
