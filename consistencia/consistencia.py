import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
# import matplotlib.pyplot as plt
#Int de 0 a 0.5 dx : int de 0 a 2x dy: fx(x) fy(y) = 1/2
from random import random
import matplotlib.pyplot as plt
import math

def clasificar_rectangulos(cantidad_por_lado,xs,ys,clases):
    rectangulos_contados={}
    for x,y,clase in zip(xs,ys,clases):
        xr=math.floor(x*cantidad_por_lado)
        yr=math.floor(y*cantidad_por_lado)
        cantidades=rectangulos_contados.get((xr,yr),{
            "positivos":0,
            "negativos":0
        })

        if clase:
            cantidades["positivos"]+=1
        else:
            cantidades["negativos"]+=1
        
        rectangulos_contados[(xr,yr)]=cantidades

    rectangulos_clasificados={}
    for xr in range(cantidad_por_lado):
        for yr in range(cantidad_por_lado):
            conteo = rectangulos_contados.get((xr,yr),{
                "positivos":0,
                "negativos":0
            })
            ganador = 0
            if conteo["positivos"]>conteo["negativos"]:
                ganador=1
            if conteo["negativos"]>conteo["positivos"]:
                ganador=-1
            
            rectangulos_clasificados[(xr,yr)]=ganador
    return rectangulos_clasificados
def parches_iguales_a(cantidad_por_lado, rectangulos_clasificados, valor):
    parches=[]
    for xr in range(cantidad_por_lado):
        for yr in range(cantidad_por_lado):
            if rectangulos_clasificados[(xr,yr)]==valor:
                parches.append(Rectangle(
                    (float(xr)/cantidad_por_lado,float(yr)/cantidad_por_lado),
                    1.0/cantidad_por_lado,1.0/cantidad_por_lado
                ))
    return parches



def generar(n):
    ret_x=[]
    ret_y=[]
    ret_clase=[]
    for i in range(n):
        arriba=random() < 0.5

        x=None
        y=None
        while True:
            x=random()
            y=random()
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
                    if y<2-2*x:
                        break
        ret_x.append(x)
        ret_y.append(y)
        ret_clase.append(arriba)
    return (ret_x,ret_y,ret_clase)


def calcular_perdida(xs,ys,clases,rects,cantidad_por_lado):
    matches=0
    for x,y,clase in zip(xs,ys,clases):
        xr=math.floor(x*cantidad_por_lado)
        yr=math.floor(y*cantidad_por_lado)
        prediccion=rects[(xr,yr)]
        if prediccion ==-1 and clase == 0:
            matches+=1
        if prediccion == 1 and clase == 1:
            matches+=1
    return matches/len(xs)

def mostrar_instancia(xs,ys,clases,rects,cantidad_por_lado):    
    xs_np=np.array(xs)
    ys_np=np.array(ys)
    clases_np=np.array(clases)
    plt.scatter(xs_np[clases_np==1],ys_np[clases_np==1],color="#0000FF11")
    plt.scatter(xs_np[clases_np==0],ys_np[clases_np==0],color="#FF000011")
    parches_positivos=parches_iguales_a(cantidad_por_lado,rects,1)
    parches_negativos=parches_iguales_a(cantidad_por_lado,rects,-1)
    ax = plt.axes()
    ax.add_collection(PatchCollection(parches_positivos,color="#0000FF33"))
    ax.add_collection(PatchCollection(parches_negativos,color="#FF000033"))
    plt.ylim(0,1)
    plt.xlim(0,1)
    perdida=calcular_perdida(xs,ys,clases,rects,cantidad_por_lado)
    plt.title("n={}, L(g)={}, h_n={}".format(
        len(xs),
        perdida,
        1.0/cantidad_por_lado
    ))
    plt.show()


def calcular_cantidad_por_lado(c,n):
    return math.floor( 1.0 / (c * math.pow(n,-0.25)) )

def analizar_instancia(xs,ys,clases):
    cantidad_por_lado = calcular_cantidad_por_lado(0.5,len(clases))
    print(cantidad_por_lado)
    rects=clasificar_rectangulos(cantidad_por_lado,xs,ys,clases)
    mostrar_instancia(xs,ys,clases,rects,cantidad_por_lado)

def agregar_n(xs,ys,clases,n):
    xs_2,ys_2,clases_2=generar(n)
    return xs+xs_2, ys+ys_2, clases+clases_2


xs,ys,clases=generar(10**2)
analizar_instancia(xs,ys,clases)
xs,ys,clases=agregar_n(xs,ys,clases,10**3-10**2)
analizar_instancia(xs,ys,clases)
xs,ys,clases=agregar_n(xs,ys,clases,10**4-10**3)
analizar_instancia(xs,ys,clases)
xs,ys,clases=agregar_n(xs,ys,clases,10**5-10**4)
analizar_instancia(xs,ys,clases)
xs,ys,clases=agregar_n(xs,ys,clases,10**6-10**5)
analizar_instancia(xs,ys,clases)