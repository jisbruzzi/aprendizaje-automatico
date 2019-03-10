import pickle
import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy.optimize import minimize

from pathlib import Path
archivos=Path(sys.argv[1]).glob("*.pickle")
contenido = map(lambda f:pickle.load(f.open("rb")) ,archivos)
experimentos=sum(contenido,[])

ds = sorted(list(set(map(lambda x:x["d"], experimentos))))

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
def mostrar_busqueda_k_para_d(d,k_elegido):
    ns,errores_l2=n_e_para_dimension(d)
    funcional=hacer_funcional_busqueda_k(ns,errores_l2)
    xs=np.arange(-1,1,0.001)
    ys=[funcional(x) for x in xs]
    plt.title("busqueda de k para d={}. K elegido:{}".format(d,k_elegido))
    plt.xlabel("k")
    plt.yscale("log")
    plt.ylabel("diferencia cuadrada")
    plt.plot(xs,ys)
    plt.savefig( (Path(sys.argv[1])/"figuras"/"busqueda-de-k-d={}.png".format(d)).open("wb") )
    plt.clf()

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
    return np.asscalar(k), np.asscalar(c), funcional(k), c_ideal, funcion_ideal, funcion_real,k[0]



ks,cs,es,c_ideales,funciones_ideales,funciones_reales,k_reales = zip(*[
    calcular_k_c_cideal_para_d(d) for d in ds
])
for d,k in zip(ds,k_reales):
    mostrar_busqueda_k_para_d(d,k)

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

if not (Path(sys.argv[1])/"figuras").exists:
    (Path(sys.argv[1])/"figuras").mkdir()

plt.savefig( (Path(sys.argv[1])/"figuras"/"resultados-grales.png").open("wb") )
plt.clf()

def grafico_para(d,f_i,f_r,k):
    n,e=n_e_para_dimension(d)
    plt.plot(n,e,label="error real")
    plt.plot(n,f_i(np.asarray(n)),label="cota con k=d/(d+2)={}".format(str(float(d)/(float(d)+2))))
    plt.plot(n,f_r(np.asarray(n)),label="menor cota con k={}".format(str(k)))
    plt.title("Promedio del error experimental y cotas teóricas para d={}".format(d))
    plt.xlabel("n")
    plt.ylabel("Promedio del error estimado")
    plt.legend()
    plt.savefig(  (Path(sys.argv[1])/"figuras"/("cotas-error-d={}.png".format(d))).open("wb") )
    plt.clf()

for d, f_i, f_r,k in zip(ds,funciones_ideales,funciones_reales,k_reales):
    grafico_para(d,f_i,f_r,k)



plt.title("k variando d")
plt.plot(ds,ks,label="k óptimo")
plt.plot(ds,2/(np.asarray(ds)+2),label="k teórico")
plt.ylabel("k")
plt.xlabel("n")
plt.legend()
plt.savefig( (Path(sys.argv[1])/"figuras"/"k-variando-d.png").open("wb") )
plt.clf()