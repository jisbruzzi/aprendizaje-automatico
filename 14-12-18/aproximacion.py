import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

datos=pickle.load(open("../puntos_generados/puntos.pickle","rb"))
#print(datos)
cantidad=200
a=np.array(datos)[:cantidad]
print(a.shape)
print("Primer argumento: h (tamaño de los intervalos). Segundo argumento: M (grado del polinomio).")

## CHEQUEAR QUE SEA EL ERROR CUADRÁTICOOOOOO
a+=np.array([1.0,0.0])
a.view('i8,i8').sort(order=['f0'], axis=0)
a+=np.array([-1.0,0.0])

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


def puntos_de_intervalo(a,minimo,maximo):
    return a[np.where((minimo<=a[:,0])*(a[:,0] < maximo))]

def aproximar_intervalo(a,minimo,maximo,grado):
    a=puntos_de_intervalo(a,minimo,maximo)
    x=np.array(a[:,0])
    y=np.array(a[:,1])
    p,residuals,_,_,_=np.polyfit(x,y,grado,full=True)
    def f(x):
        deg=p.shape[0]
        sp=[float(p[i])*(x**(deg-i-1)) for i in range(deg)]
        return sum(sp)
    x_predicho=np.arange(minimo,maximo,0.0025)
    y_predicho=f(x_predicho)
    y_predicho_posta=f(x)
    y_real=np.array([m(xx) for xx in x])
    return x, y, x_predicho, y_predicho, y_predicho_posta, y_real

def aproximar_intervalos(a,h,m):
    cantidad_intervalos=int(2.0/h)
    x_t=np.array([])
    y_t=np.array([])
    y_real_t=np.array([])
    x_predicho_t=np.array([])
    y_predicho_t=np.array([])
    y_predicho_posta_t=np.array([])
    for  i in range(cantidad_intervalos):
        fin=-1+(i+1)*h
        inicio=-1+i*h
        x, y, x_predicho, y_predicho, y_predicho_posta, y_real = aproximar_intervalo(a,inicio,fin,m)
        x_t=np.append(x_t,x)
        y_t=np.append(y_t,y)
        y_predicho_t=np.append(y_predicho_t,y_predicho)
        x_predicho_t=np.append(x_predicho_t,x_predicho)
        y_predicho_posta_t=np.append(y_predicho_posta_t,y_predicho_posta)
        y_real_t=np.append(y_real_t,y_real)
        plt.plot(x_predicho,y_predicho,color="g")
        plt.plot([fin,fin],[-1,2],color="#BBBBBB")
        plt.plot([inicio,inicio],[-1,2],color="#BBBBBB")
    diferencia=sum((y_real_t-y_predicho_posta_t)*(y_real_t-y_predicho_posta_t))/cantidad
    #plt.scatter(x_t,y_t)
    plt.ylim(y_t.min()-0.1,y_t.max()+0.1)
    plt.title("Predicción con h={}, M={}, error contra m={}".format(h,m,diferencia))
    
    plt.savefig("aproximar_intervalos_h={}_M={}.png".format(h,m))
    plt.clf()

#aproximar_intervalos(a,float(sys.argv[1]),int(sys.argv[2]))
aproximar_intervalos(a,0.1,1)
aproximar_intervalos(a,0.5,1)
aproximar_intervalos(a,0.5,2)
aproximar_intervalos(a,0.25,2)
