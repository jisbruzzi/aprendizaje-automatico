import pickle
import matplotlib.pyplot as plt
import sys
import numpy as np

experimentos=pickle.load(open(sys.argv[1],"rb"))

"""
ns=[]
ds=[]
y_mins=[]
y_maxs=[]
y_posibles=[]
for experimento in experimentos:

    n=experimento["n"]
    d=experimento["d"]
    y_posible=experimento["promedio_estimaciones_L2"]
    y_min = y_posible - experimento["error_promedio_estimaciones_L2"]
    y_max = y_posible + experimento["error_promedio_estimaciones_L2"]

    if d!=1:
        continue

    ns.append(n)
    ds.append(d)
    y_posibles.append(y_posible)
    y_mins.append(y_min)
    y_maxs.append(y_max)

plt.plot(ns,y_posibles)
plt.plot(np.asarray(ns),np.power(np.asarray(ns),-2.0/3.0)/1.5)

plt.show()

"""

from scipy.optimize import minimize

def c_para_coordenadas(k,n,e):
    return e*(n**k)

def obtener_menor_c_para_k(k,ns,es):
    return max([c_para_coordenadas(k,n,e) for n,e in zip(ns,es)])

def error_para_c_k(c,k,ns,es):
    return sum([ ( c*(n**(-k)) - e )**2 for n,e in zip(ns,es) ])

def buscar_errores_para_ks(ns,es,ks):
    errores=[]
    cs=[]
    for k in ks:
        c = obtener_menor_c_para_k(k,ns,es)
        error = error_para_c_k(c,k,ns,es)
        cs.append(c)
        errores.append(error)
    return cs,errores

def hacer_funcional_busqueda_k(ns,es):
    def f(k):
        c=obtener_menor_c_para_k(k,ns,es)
        return error_para_c_k(c,k,ns,es)
    return f
"""
ks=np.linspace(0,2,100)
cs,errores = buscar_errores_para_ks(ns,y_posibles,ks)

plt.scatter(ks,np.log(errores))
plt.show()

"""

ns,errores_l2=zip(*[ (experimento["n"],experimento["promedio_estimaciones_L2"]) for experimento in experimentos if experimento["d"]==3])
funcional=hacer_funcional_busqueda_k(ns,errores_l2)
optimo=minimize(funcional,1,tol=0.000001)
print(optimo)

