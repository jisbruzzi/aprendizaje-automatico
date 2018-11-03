import pickle
import math
from heapq import nsmallest
import matplotlib.pyplot as plt
import numpy as np
from math import floor
puntos = pickle.load(open("../puntos_generados/puntos.pickle","rb"))
puntos=puntos[0:200]
def generar_m_knn(k,puntos):
    def calcular(x):
        print("----------------------")
        print(x)
        def distancia_x(z_prima):
            df = abs(float(z_prima[0])-float(x))
            return df
        puntos_ordenados=nsmallest(k,puntos,key=distancia_x)
        print(puntos_ordenados)
        s=0
        for p in puntos_ordenados:
            s+=p[1]
        ret = float(s)/float(len(puntos_ordenados))
        print(s)
        print(ret)
        return ret
    return calcular

def generar_nadaraya_simple(h,puntos):
    def k(x):
        return 1 if abs(x)<= 1 else 0
    def calcular(x):
        prod=sum([k(float(x-p[0])/float(h))*p[1] for p in puntos])
        ks=sum([k(float(x-p[0])/float(h)) for p in puntos])

        if ks==0 and prod==0:#si solo ks da 0, que se rompa
            return 0

        return float(prod)/float(ks)
    return calcular

def generar_particionado(cantidad_particiones,puntos):
    def calcular_particion(x):
        return floor((x+1.0)/2.0 * cantidad_particiones)

    sumas_partes={}
    cantidades_partes={}
    for p in puntos:
        particion=calcular_particion(p[0])
        sumas_partes[particion]=sumas_partes.get(particion,0)+p[1]
        cantidades_partes[particion]=cantidades_partes.get(particion,0)+1
    promedios_partes={}
    for k in sumas_partes:
        promedios_partes[k] = float(sumas_partes[k]) / float(cantidades_partes[k])
    
    def calcular(x):
        particion=calcular_particion(x)
        return promedios_partes.get(particion,0)

    return calcular




def dibujar(fun,nombre_imagen):
    xs=[p[0] for p in puntos]
    ys=[p[1] for p in puntos]
    plt.scatter(xs,ys,color='r')
    xs=np.linspace(-1,1,1000)
    ys=[fun(x) for x in xs]
    plt.plot(xs,ys)
    plt.title(nombre_imagen)
    plt.savefig(nombre_imagen+".png")
    plt.clf()

dibujar(generar_m_knn(5,puntos),"5nn")
dibujar(generar_m_knn(2,puntos),"2nn")
dibujar(generar_m_knn(10,puntos),"10nn")
dibujar(generar_nadaraya_simple(1,puntos),"nadaraya h=1")
dibujar(generar_nadaraya_simple(0.5,puntos),"nadaraya h=0.5")
dibujar(generar_nadaraya_simple(0.2,puntos),"nadaraya h=0.2")
dibujar(generar_nadaraya_simple(0.1,puntos),"nadaraya h=0.1")
dibujar(generar_nadaraya_simple(0.01,puntos),"nadaraya h=0.01")
dibujar(generar_particionado(14,puntos),"14-particionado")
dibujar(generar_particionado(28,puntos),"28-particionado")
dibujar(generar_particionado(56,puntos),"56-particionado")


