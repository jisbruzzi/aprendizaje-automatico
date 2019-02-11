import logging
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)

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
        grado=randint(2,5)
        xs=[xi]
        ys=[yi]
        for i in range(grado-2):
            xs.append(uniform(xi,xf))
            ys.append(uniform(-1,1))
        xs.append(xf)
        ys.append(yf)
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

"""
No usar esta!!!
"""
def generar_poli_partes(d=1):
    polis_partes=[generar_poli_partes_1d() for i in range(d)]
    def f(x):
        if x.shape[1]!=d:
            raise Exception("La matriz proporcionada tiene "+str(x.shape[1])+" columnas en vez de "+str(d))
        resultados=np.asarray([1 for i in range (x.shape[0])])
        for i in range(d):
            resultado_columna = np.asarray(polis_partes[i](x[:,i]))
            
            resultados=resultados*resultado_columna
        return resultados
    return f
import numbers


def suma_kernelizada(kernel,d,puntos,ys):
    def f(x):
        if isinstance(x, numbers.Number):
            x=np.asarray([[x]])
        if len(x.shape)==1:
            x=np.reshape(x,(-1,1))
        
        if x.shape[1]!=d:
            raise Exception("La matriz proporcionada tiene "+str(x.shape[1])+" columnas en vez de "+str(d))
        rets=[]
        
        for valor_x in x:
            distancias=np.linalg.norm(puntos-valor_x,axis=1)
            valores_kernel=kernel(distancias)
            factores=np.divide(valores_kernel,np.sum(valores_kernel))
            ret=np.sum(ys*factores)
            if math.isnan(ret):
                ret=0.0
            rets.append(ret)
            
                
        return rets
    return f

def generar_polinomio_kernel(kernel,d=1):
    
    cantidad_puntos=randint(5,8)*d
    puntos=np.random.uniform(-1,1,(cantidad_puntos,d))
    ys=np.random.uniform(-1,1,cantidad_puntos)
    min_y=min(ys)
    max_y=max(ys)
    alto=max_y-min_y
    ys=((ys-min_y)/alto)*2.0-1

    return suma_kernelizada(kernel,d,puntos,ys)


def kernel_polinomico(q):
    def f(x):
        return np.maximum(np.power(1-x,q),0)
    return f
def ancho_kernel(kernel,h):
    def f(x):
        return kernel(x/float(h))
    return f


def kernel_gausiano(x):
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    return gaussian(x,0,0.2)


def sumar_ruido(m,sigma):
    def f(x):
        m_x=m(x)
        s_x=sigma(x)
        return norm.rvs(loc=m_x, scale = np.abs(s_x), size=1)[0]
    def f_muchos(x):
        return np.asarray([f(np.reshape(y,(1,-1))) for y in x])
    return f_muchos


import matplotlib.pyplot as plt
from matplotlib.mlab import frange
from scipy.stats import norm


def generar_Xn_Yn(funcion_original,funcion_ruido,n,d):
    x_n=np.random.uniform(-1,1,(n,d))
    y_n=sumar_ruido(funcion_original,funcion_ruido)(x_n)
    return x_n, y_n

def generar_predictor(x_n,y_n,kernel,d):
    return suma_kernelizada(kernel,d,x_n,y_n)

def generar_predictor_a_partir_de_datos(funcion_original,funcion_ruido,n,d,kernel):
    x_n,y_n=generar_Xn_Yn(funcion_original,funcion_ruido,n,d)
    return generar_predictor(x_n,y_n,kernel,d)


import mcint
def estimar_error_L2(f1,f2,d,precision=1000):
    def f(x):

        r= np.power(np.asarray(f1(x))-np.asarray(f2(x)),2)
        return r
    def sampler():
        while True:
            u=np.random.uniform(-1,1,(1,d))
            yield u
    r=mcint.integrate(f,sampler(),measure=1.0,n=precision)
    return r

def kernel_naive(x):
    return np.array(abs(x)<=1,dtype=np.float64)

"""
Encuentra la media del error L2 promediando el obtenido a partir de estimar_error_L2 para q regresiones y qq estimaciones de la regresión para cada q (obtenida cada una a partir de )
"""
def encontrarE(n,d,q=100,qq=3,kernel_prediccion=ancho_kernel(kernel_naive,0.1)):
    logging.info("Voy a calcular E para n={}, d={}, q={}, qq={}".format(n,d,q,qq))
    suma_estimaciones_L2 = .0
    cantidad_estimaciones_L2 = 0
    suma_errores_estimaciones_L2=.0
    for m_s in range(q):
        m = generar_polinomio_kernel(kernel_gausiano,d)
        for m_n_s in range(qq):
            logging.info("Corro el experimento número {} de {}".format(m_s*qq+m_n_s,q*qq))
            sigma = generar_polinomio_kernel(kernel_gausiano,d)
            m_n = generar_predictor_a_partir_de_datos(m,sigma,n,d,kernel_prediccion)
            estimacion_L2, error=estimar_error_L2(m,m_n,d)
            suma_estimaciones_L2+=estimacion_L2
            if math.isnan(suma_errores_estimaciones_L2):
                suma_errores_estimaciones_L2=.0
            if math.isnan(suma_estimaciones_L2):
                suma_estimaciones_L2=.0
            suma_errores_estimaciones_L2+=error
            cantidad_estimaciones_L2+=1

    promedio_estimaciones_L2=float(suma_estimaciones_L2)/float(cantidad_estimaciones_L2)

    error_promedio_estimaciones_L2=float(suma_errores_estimaciones_L2)/float(cantidad_estimaciones_L2)

    return {
        "suma_estimaciones_L2":suma_estimaciones_L2,
        "suma_errores_estimaciones_L2":suma_errores_estimaciones_L2,
        "cantidad_estimaciones_L2":cantidad_estimaciones_L2,
        "promedio_estimaciones_L2":promedio_estimaciones_L2, 
        "error_promedio_estimaciones_L2":error_promedio_estimaciones_L2,
        "n":n,
        "d":d,
        "q":q,
        "qq":qq
    }

import pickle
from time import gmtime, strftime
hs=strftime("%d-%b-%Y-%H-%M-%S", gmtime())

Es=[]
for d in [1,2,3,10,30,100]:
    for i in range(100):
        Es.append(encontrarE(100*(i+1),d))
    with open("salida_experimento_{}------2.pickle".format(hs),"wb") as f:
        pickle.dump(Es,f)



"""


r=frange(-0.99,0.99,0.001)
print("poli partes:")
#poli_partes=generar_poli_partes_1d()
poli_partes=generar_polinomio_kernel(kernel_gausiano)
print("ruido:")
ruido=generar_poli_partes_1d()
y=poli_partes(np.asmatrix(r).T)
predictor=generar_predictor_a_partir_de_datos(poli_partes,ruido,1000,1,ancho_kernel(kernel_naive,0.2))
y_predicho=predictor(r)
plt.plot(r,y)
plt.scatter(r,y_predicho)
#plt.plot(r,ruido(r))
plt.show()

print("Empiezo a calcular la diferencia")
resultado,error=estimar_error_L2(predictor,poli_partes,1,precision=100000)
print(resultado,error)
"""



"""

print(p(np.asarray([
    [0.1],
    [0.1],
])))
"""

"""
# the function to be plotted

p=generar_polinomio_kernel(kernel_lineal,2)
def func(x,y):    
    
    # gives vertical color bars if x is horizontal axis
    return p(np.asarray([[x,y]]))[0]

import pylab

# define the grid over which the function should be plotted (xx and yy are matrices)
xx, yy = pylab.meshgrid(
    pylab.linspace(-1,1, 101),
    pylab.linspace(-1,1, 111))

# indexing of xx and yy (with the default value for the
# 'indexing' parameter of meshgrid(..) ) is as follows:
#
#   first index  (row index)    is y coordinate index
#   second index (column index) is x coordinate index
#
# as required by pcolor(..)

# fill a matrix with the function values
zz = pylab.zeros(xx.shape)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        zz[i,j] = func(xx[i,j], yy[i,j])

# plot the calculated function values
pylab.pcolor(xx,yy,zz)

# and a color bar to show the correspondence between function value and color
pylab.colorbar()

pylab.show() 
"""