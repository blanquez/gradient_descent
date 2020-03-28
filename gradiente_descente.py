##############################################################
## Antonio José Blánquez Pérez
## 45926869D
## Aprendizaje Automático - 3 CSI
## Grupo 2 - Martes
## Práctica 1 - Ejercicio 1: Gradiente descendente
##############################################################

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from math import e,pi,sin,cos

#-------------------------------------------------------------------------------#
#------------- Ejercicio sobre la búsqueda iterativa de óptimos ----------------#
#-------------------------------------------------------------------------------#


#------------------------------Ejercicio 1 -------------------------------------#

# Fijamos la semilla
def E(w): 
	return (w[0]*(e**w[1]) - 2*w[1]*(e**-w[0]))**2
			 
# Derivada parcial de E respecto de u
def Eu(w):
	return 2*(e**(-2*w[0]))*(e**(w[0] + w[1])*w[0] - 2*w[1])*(e**(w[0] + w[1]) + 2*w[1])

# Derivada parcial de E respecto de v
def Ev(w):
	return 2*(e**(-2*w[0]))*(e**(w[0] + w[1])*w[0] - 2)*(e**(w[0] + w[1])*w[0] - 2*w[1])
	
# Gradiente de E
def gradE(w):
	return np.array([Eu(w), Ev(w)])

# Gradiente descendente para el ejercicio 1
def gd_ej1(w, lr, grad_fun, fun, epsilon, max_iters = 50):
	it = 0
	while (fun > epsilon and it < max_iters):
		w = w - lr * grad_fun
		it = it + 1
		fun = E(w)
		grad_fun = gradE(w)
		plt.plot(w[0],w[1],"o",c="red")
	return w, it

# Codigo principal para el ejercicio 1

print('\nGRADIENTE DESCENDENTE')
print('\nEjercicio 1\n\n')

w = np.array([1,1])
lrate = 0.1
fun = E(w)
grad_fun = gradE(w)
eps = np.float64(10**-14)

plt.plot(w[0], w[1], "o", c="black", label="Punto inicial")

w, num_ite = gd_ej1(w,lrate,grad_fun,fun,eps)

plt.plot(w[0], w[1], "o", c="pink", label = "Punto de parada")

print('Numero de iteraciones: ', num_ite)
print('\nCoordenadas obtenidas: (', w[0], ', ', w[1],')')

X = np.linspace(-0.4,1.03,100)
Y = np.linspace(-0.6,1.03,100)
Z = np.zeros((100,100))

for ix, x in enumerate(X):
	for iy, y in enumerate(Y):
		Z[iy,ix] = E([x,y])

plt.contourf(X,Y,Z)
plt.colorbar()
plt.legend(loc="lower right")
plt.show()

input("\n\n--- Pulsar tecla para continuar ---\n\n")

#------------------------------Ejercicio 2 -------------------------------------#

def f(w):   
	return (w[0]-2)**2 + 2*(w[1]+2)**2 + 2*sin(2*pi*w[0])*sin(2*pi*w[1])
	
# Derivada parcial de f respecto de x
def fx(w):
	return 2*(2*pi*cos(2*pi*w[0])*sin(2*pi*w[1])+w[0]-2)

# Derivada parcial de f respecto de y
def fy(w):
	return 4*(pi*sin(2*pi*w[0])*cos(2*pi*w[1])+w[1]+2)
	
# Gradiente de f
def gradf(w):
	return np.array([fx(w), fy(w)])
	
# a) Usar gradiente descendente para minimizar la función f, con punto inicial (1,1)
# tasa de aprendizaje 0.01 y max 50 iteraciones. Repetir con tasa de aprend. 0.1

# Gradiente descendente para el ejercicio 2 apartado a)
def gd_grafica(w, lr, grad_fun, fun, max_iters = 50):
	graf = np.array(fun)
	it = 0
	while (it < max_iters):
		w = w - lr * grad_fun
		fun = f(w)
		grad_fun = gradf(w)
		graf = np.append(graf,f(w))
		it = it + 1

	plt.plot(range(0,max_iters+1),graf,label=lr)
	plt.title("Learning rate")
	plt.xlabel('iteraciones')
	plt.ylabel('f(x,y)')
	plt.legend()
	plt.show()

# Codigo principal para el apartado a del ejercicio 2

print('Ejercicio 2\n\n')

w = np.array([1,-1])
lrate = 0.01
grad_fun = gradf(w)
fun = f(w)

print('\nGrafica con learning rate igual a ', lrate)
gd_grafica(w,lrate,grad_fun,fun)
input("\n\n--- Pulsar tecla para continuar ---\n\n")

w = np.array([1,-1])
lrate = 0.1
grad_fun = gradf(w)
fun = f(w)
print('\nGrafica con learning rate igual a ', lrate)
gd_grafica(w,lrate,grad_fun,fun)
input("\n\n--- Pulsar tecla para continuar ---\n\n")


# b) Obtener el valor minimo y los valores de (x,y) con los
# puntos de inicio siguientes:

# Gradiente descendente para el ejercicio 2 apartado b)
def gd_ej2(w, lr, grad_fun, max_iters = 50):
	it=0
	while (it < max_iters):
		w = w - lr * grad_fun
		it = it + 1
		grad_fun = gradf(w)
	return w

# Codigo principal para el apartado 2 del ejercicio 2

# Ejecucion para (2.1,-2.1)

l_rate=0.01

w = np.array([2.1,-2.1])
grad_fun = gradf(w)
w = gd_ej2(w,lrate,grad_fun)

print ('Punto de inicio: (2.1, -2.1)\n')
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n\n--- Pulsar tecla para continuar ---\n\n")

# Ejecucion para (3,-3)

w = np.array([3,-3])
grad_fun = gradf(w)
w = gd_ej2(w,lrate,grad_fun)

print ('Punto de inicio: (3.0, -3.0)\n')
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n\n--- Pulsar tecla para continuar ---\n\n")

# Ejecucion para (1.5,1.5)

w = np.array([1.5,1.5])
grad_fun = gradf(w)
w = gd_ej2(w,lrate,grad_fun)

print ('Punto de inicio: (1.5, 1.5)\n')
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n\n--- Pulsar tecla para continuar ---\n\n")

# Ejecucion para (1,-1)

w = np.array([1,-1])
grad_fun = gradf(w)
w = gd_ej2(w,lrate,grad_fun)

print ('Punto de inicio: (1.0, -1.0)\n')
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor mínimo: ',f(w))

input("\n\n--- Pulsar tecla para terminar ---\n\n")