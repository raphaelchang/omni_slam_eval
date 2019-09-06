import math

fov=360
alpha=0.666667

d = (180. - fov) / 2. * math.pi / 180.
a = max(min(alpha, 1.), 0.5)
beta = math.atan((a - 1) * math.sqrt(1 / (2 * a - 1)))
chi = math.sqrt(abs((-4*a**2*math.sqrt(1/(2.*a - 1))*math.tan(d) + a**2 + 6*a*math.sqrt(1/(2.*a - 1))*math.tan(d) + 2*a*math.tan(d)**2 - 2*a - 2*math.sqrt(1/(2.*a - 1))*math.tan(d) - math.tan(d)**2 + 1)*math.cos(d)**2/(2.*a - 1)))

if d < beta:
    print 'chi = ' + str(chi)
else:
    print 'chi = ' + str(-chi)
print 'focal length = ' + str(0.5 / math.sqrt(1 / (2 * a - 1)))
