import numpy as np
# import matplotlib.pyplot as plt
#Int de 0 a 0.5 dx : int de 0 a 2x dy: fx(x) fy(y) = 1/2
import math
def generar(n):
    ret_x=[]
    ret_y=[]
    for i in range(n):
        arriba=math.random() < 0.5
        x=None
        y=None
        while True:
            x=math.random()
            y=math.random()
            if arriba:
                if x<0.5:
                    if y > 1-2*x:
                        break
                else: 
                    if y>2*x-1:
                        break
            else:
                if x<0.5:
                    if y<2*x:
                        break
                else:
                    if y<1-2*x:
                        break
        ret_x.append(x)
        ret_y.append(y)
    return (ret_x,ret_y)


