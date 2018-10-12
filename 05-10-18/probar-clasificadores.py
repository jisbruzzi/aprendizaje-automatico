from random import random
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

def evaluar_forma(forma,puntos):
    clase_lado={True:{},False:{}}
    
    for x, y, clase in zip(*puntos):
        lado=forma(x,y)
        clase_lado[clase][lado] = clase_lado[clase].get(lado,0) + 1
    clase_mayoritaria_de_lado={}
    bien_clasificados=0
    for lado in clase_lado[True].keys() & clase_lado[False].keys():
        clase_mayoritaria_de_lado[lado] = clase_lado[True].get(lado,0) > clase_lado[False].get(lado,0)
        bien_clasificados+=clase_lado[clase_mayoritaria_de_lado[lado]][lado]
    return float(bien_clasificados)/len(puntos[0]), clase_lado

def forma_O_E(x,y):
    return x<0.5
def forma_N_S(x,y):
    return y<0.5
def forma_NO_SE(x,y):
    return x<y
def forma_SO_NE(x,y):
    return y<1-x
def forma_cruz(x,y):
    fx = 1 if forma_O_E(x,y) else 0
    fy = 1 if forma_N_S(x,y) else 0
    return fx+fy*2
def forma_X(x,y):
    fx = 1 if forma_SO_NE(x,y) else 0
    fy = 1 if forma_NO_SE(x,y) else 0
    return fx+fy*2 


formas=[
    forma_O_E,
    forma_N_S,
    forma_NO_SE,
    forma_SO_NE,
    forma_cruz,
    forma_X
]
datos=generar(10000)
for i, f in enumerate(formas):
    ev,r = evaluar_forma(f,datos)
    print(i,ev,r)
    


