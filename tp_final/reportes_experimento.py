import pickle
import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy.optimize import minimize

experimentos=pickle.load(open(sys.argv[1],"rb"))

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

def n_e_para_dimension(d):
    return zip(*[ 
        (
            experimento["n"],
            experimento["promedio_estimaciones_L2"]
        )
        for experimento in experimentos if experimento["d"]==d
    ])

def calcular_k_c_cideal_para_d(d):
    ns,errores_l2=n_e_para_dimension(d)
    funcional=hacer_funcional_busqueda_k(ns,errores_l2)
    optimo=minimize(funcional,1,tol=0.000001)
    k=optimo.x
    c=obtener_menor_c_para_k(k,ns,errores_l2)
    k_ideal=2.0/(d+2.0)
    c_ideal=obtener_menor_c_para_k(k_ideal,ns,errores_l2)
    def funcion_ideal(x):
        return c_ideal*(x**(-k_ideal))
    def funcion_real(x):
        return c*(x**(-k))
    return np.asscalar(k), np.asscalar(c), funcional(k), c_ideal, funcion_ideal, funcion_real

ds=[1,2,3,4,6,8,10,30]
ks,cs,es,c_ideales,funciones_ideales,funciones_reales = zip(*[
    calcular_k_c_cideal_para_d(d) for d in ds
])
print(ds)
print("----")
print(2.0/(np.asarray(ds)+2.0))
print(ks)
print("----")
print(cs)
print("----")
print(es)

def plot(d):
    n,e=n_e_para_dimension(d)
    plt.plot(n,e,label="d={}".format(d))

for d in ds:
    plot(d)
plt.legend()
plt.ylabel("Error estimado")
plt.xlabel("n")
plt.title("Promedio del error estimado variando n y d")
plt.savefig("figuras/resultados-grales.png")
plt.clf()

def grafico_para(d,f_i,f_r):
    n,e=n_e_para_dimension(d)
    plt.plot(n,e,label="error real")
    plt.plot(n,f_i(np.asarray(n)),label="cota con k teórico")
    plt.plot(n,f_r(np.asarray(n)),label="menor cota con un k distinto del teórico")
    plt.title("Promedio del error experimental y cotas teóricas para d={}".format(d))
    plt.xlabel("n")
    plt.ylabel("Promedio del error estimado")
    plt.legend()
    plt.savefig("figuras/cotas-error-d={}.png".format(d))
    plt.clf()

for d, f_i, f_r in zip(ds,funciones_ideales,funciones_reales):
    grafico_para(d,f_i,f_r)



plt.title("k variando d")
plt.plot(ds,ks,label="k óptimo")
plt.plot(ds,2/(np.asarray(ds)+2),label="k teórico")
plt.ylabel("k")
plt.xlabel("n")
plt.legend()
plt.savefig("figuras/k-variando-d.png")
plt.clf()