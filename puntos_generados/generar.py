import math
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, norm
from matplotlib.mlab import frange
import sys
import pickle

number_of_points=int(sys.argv[1])

def m(x):
    if -1 <= x < -0.5:
        return (x+2)**2/2
    if -0.5 <= x < 0:
        return x/2 + 0.875
    if 0 <= x < 0.5:
        return -5*((x-0.2)**2)+1.075
    if 0.5 <= x < 1:
        return x + 0.125

    raise "No pertenece al dominio"

def sigma(x):
    return 0.2-0.1*math.cos(2*math.pi*x)

def sim_y(x):
    return norm.rvs(loc=m(x), scale = sigma(x), size=1)[0]

r=frange(-0.99,0.99,0.001)
rv=truncnorm(-1,1)
#plt.plot(r,[rv.pdf(x) for x in r])

xs=rv.rvs(size=number_of_points)
ys=[]
points=[]
for x in xs:
    y=sim_y(x)
    print(x,y,sep=",")
    ys.append(y)
    points.append((x,y))

plt.scatter(xs,ys)
plt.plot(r,[m(x) for x in r])
plt.plot(r,[sigma(x) for x in r])
pickle.dump(points,open("puntos.pickle","wb"))

plt.show()