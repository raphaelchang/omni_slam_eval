import math

cx = 0.5
focal_length = 0.6028125
alpha = 0.0
chi = 0

mx = cx / focal_length
r2 = mx ** 2
mz = (1 - alpha ** 2 * r2) / (alpha * math.sqrt(1 - (2 * alpha - 1) * r2) + 1 - alpha)
beta = (mz * chi + math.sqrt(mz ** 2 + (1 - chi ** 2) * r2)) / (mz ** 2 + r2)
print 2 * (math.pi / 2 - math.atan2(beta * mz - chi, beta * mx)) * 180/math.pi
