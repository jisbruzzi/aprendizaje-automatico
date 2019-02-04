"""
Genero funciones entre -1,1,-1,1; con una cantidad azarosa de polinomios de Lagrange encastradas.

Genero la varianza del ruido de la misma forma.

Corro el naive kernel variando la cantidad de datos, de esta forma consigo m_n

Estimo E|| m_n - m ||^2, y lo corro también randomizando los x (todas las veces que pueda, capaz una noche entera generando datos/situaciones), y también randomizando la función m original.
"""

import math
from random import randint, random, shuffle,uniform
import  scipy.interpolate as interpolate
import numpy as np


def generar_funcion_1d(polinomios,condiciones):
    def f(x):
        condiciones_bool=[]
        for c in condiciones:
            es_mayor=(c["desde"] <= x)
            es_menor=(x < c["hasta"])
            valores= es_menor * es_mayor
            condiciones_bool.append(valores)
        return np.piecewise(x,condiciones_bool,polinomios)
    return f



def generar_poli_1d(xi,yi,xf,yf):
    def generar_algun_poli_1d(xi,yi,xf,yf):
        print("----")
        grado=randint(2,5)
        xs=[xi]
        ys=[yi]
        for i in range(grado-2):
            xs.append(uniform(xi,xf))
            ys.append(uniform(-1,1))
        xs.append(xf)
        ys.append(yf)
        print(ys)
        return interpolate.lagrange(xs,ys)
    while True:
        p=generar_algun_poli_1d(xi,yi,xf,yf)
        minmax_locales=p.deriv().roots
        if all([abs(p(mm))<=1 for mm in minmax_locales]):
            return p


def generar_poli_partes_1d():
    cantidad_polinomios=randint(3,5)+1
    cortes=[-1]+[uniform(-1,1) for x in range(cantidad_polinomios-2)]+[1]
    polinomios=[]
    condiciones=[]
    x1=-1
    y1=uniform(-1,1)
    for c in range(len(cortes)-1):
        x2, y2 = cortes[c+1], uniform(-1,1)
        polinomios.append(generar_poli_1d(x1,y1,x2,y2))
        condiciones.append({'desde':x1,'hasta':x2})
        x1, y1 = x2, y2
    return generar_funcion_1d(polinomios,condiciones)

def sumar_ruido(m,sigma):
    def f(x):
        return norm.rvs(loc=m(x), scale = abs(sigma(x)), size=1)[0]
    def f_muchos(x):
        return np.asarray([f(y) for y in x])
    return f_muchos


import matplotlib.pyplot as plt
from matplotlib.mlab import frange
from scipy.stats import norm

r=frange(-0.99,0.99,0.001)
print("poli partes:")
poli_partes=generar_poli_partes_1d()
print("ruido:")
ruido=generar_poli_partes_1d()
y=poli_partes(r)
y_ruido=sumar_ruido(poli_partes,ruido)(r)
plt.plot(r,y)
plt.scatter(r,y_ruido)
plt.plot(r,ruido(r))
plt.show()



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

def sim_y(x):
    def sigma(x):
        return 0.2-0.1*math.cos(2*math.pi*x)
    return norm.rvs(loc=m(x), scale = sigma(x), size=1)[0]
