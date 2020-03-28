##############################################################
## Antonio José Blánquez Pérez
## 45926869D
## Aprendizaje Automático - 3 CSI
## Grupo 2 - Martes
## Práctica 1 - Ejercicio 2: Regresión lineal
##############################################################

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------#
#---------------------- Ejercicio sobre regresión lineal -----------------------#
#-------------------------------------------------------------------------------#

#------------------------------Ejercicio 1 -------------------------------------#


# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []
	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(1)
			else:
				y.append(-1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y
	
# Funcion para calcular el error
def Err(x,y,w):
	return np.sum((x @ w - y)**2) / len(x)
	
# Gradiente Descendente Estocastico
def sgd(x, y, lr, max_iters, tam_minibatch):
	w = np.zeros(len(x[0]))
	indices = np.arange(len(x))
	
	for it in range(max_iters):
		np.random.shuffle(indices)
		m = []
		ym = []
		for i in range(tam_minibatch):
			m.append(x[indices[i]])
			ym.append(y[indices[i]])
		m = np.array(m)
		ym = np.array(ym)
		
		grad = []
		for i in range(len(w)):
			grad.append((2 / tam_minibatch) * np.sum((m @ w - ym) * m[:,i]))
		grad = np.array(grad)
		w = w - lr * grad

	return w
	
# Algoritmo pseudoinversa	
def pseudoinverse(x, y):
	return np.linalg.pinv(x) @ y
	
# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')

# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

print ('EJERCICIO SOBRE REGRESION LINEAL\n')
print ('Ejercicio 1\n')

# Gradiente descendente estocastico

w = sgd(x, y, 0.05, 100, 32)

x_pos = []
x_neg = []

h = x @ w
for i in range(len(x)):
	if np.sign(h[i]) == 1:
		x_pos.append(x[i])
	else:
		x_neg.append(x[i])

x_pos = np.array(x_pos)
x_neg = np.array(x_neg)

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x, y, w))
print ("Eout: ", Err(x_test, y_test, w))

plt.scatter(x_pos[:,1], x_pos[:,2], c="blue", alpha=0.8)
plt.scatter(x_neg[:,1], x_neg[:,2], c="yellow", alpha=0.8)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# Algoritmo Pseudoinversa

w = pseudoinverse(x,y)

x_pos = []
x_neg = []

h = x @ w

for i in range(len(x)):
	if np.sign(h[i]) == 1:
		x_pos.append(x[i])
	else:
		x_neg.append(x[i])

x_pos = np.array(x_pos)
x_neg = np.array(x_neg)

plt.scatter(x_pos[:,1], x_pos[:,2], c="blue", alpha=0.8)
plt.scatter(x_neg[:,1], x_neg[:,2], c="yellow", alpha=0.8)
plt.show()

print ('\nBondad del resultado para el algoritmo de la pseudoinversa:\n')
print ("Ein: ", Err(x, y, w))
print ("Eout: ", Err(x_test, y_test, w))
input("\n--- Pulsar tecla para continuar ---\n")


#------------------------------Ejercicio 2 -------------------------------------#

# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	datos = []
	for i in range(N):
		datos.append(np.random.uniform(-size,size,d))
	return np.array(datos)
	
# EXPERIMENTO	
# a) Muestra de entrenamiento N = 1000, cuadrado [-1,1]x[-1,1]

print ('Ejercicio 2\n')
print ('Muestra N = 1000, cuadrado [-1,1]x[-1,1]')

muestra = simula_unif(1000,2,1)

plt.scatter(muestra[:,0],muestra[:,1], c="blue", alpha=0.7)
plt.show()

# b.1) Asignacion de etiquetas

def f_sign(x1,x2):
	return np.sign((x1-0.2)**2 + x2**2-0.6)

y_muestra = f_sign(muestra[:,0],muestra[:,1])

# b.2) Introduccion de ruido

cambios = np.arange(len(muestra))
np.random.shuffle(cambios)

for i in range(int(len(muestra)*0.1)):
	if y_muestra[cambios[i]] == 1:
		y_muestra[cambios[i]] = -1
	else:
		y_muestra[cambios[i]] = 1

muestra_pos = []
muestra_neg = []

for i in range(len(muestra)):
	if y_muestra[i] == 1:
		muestra_pos.append(muestra[i])
	else:
		muestra_neg.append(muestra[i])

muestra_pos = np.array(muestra_pos)
muestra_neg = np.array(muestra_neg)

plt.scatter(muestra_pos[:,0],muestra_pos[:,1], c="blue", alpha=0.7)
plt.scatter(muestra_neg[:,0],muestra_neg[:,1], c="yellow", alpha=0.7)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# c1) Ajustar modelo de regresion lineal

muestra = np.append(np.ones((len(muestra),1)), muestra, axis=1)

w = sgd(muestra, y_muestra, 0.05, 100, 32)

h = muestra @ w

muestra_pos = []
muestra_neg = []

for i in range(len(muestra)):
	if np.sign(h[i]) == 1:
		muestra_pos.append(muestra[i])
	else:
		muestra_neg.append(muestra[i])

muestra_pos = np.array(muestra_pos)
muestra_neg = np.array(muestra_neg)

print ('Bondad del resultado para un modelo lineal:\n')
print ("Ein: ", Err(muestra, y_muestra, w))

plt.scatter(muestra_pos[:,1], muestra_pos[:,2], c="blue", alpha=0.7)
plt.scatter(muestra_neg[:,1], muestra_neg[:,2], c="yellow", alpha=0.7)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# c2) Ajustar modelo de regresion no lineal

muestra = np.insert(muestra,muestra.shape[1], (muestra[:,1] * muestra[:,2]) ,1)
muestra = np.insert(muestra,muestra.shape[1], (muestra[:,1] * muestra[:,1]) ,1)
muestra = np.insert(muestra,muestra.shape[1], (muestra[:,2] * muestra[:,2]) ,1)

w = sgd(muestra, y_muestra, 0.05, 100, 32)

h = muestra @ w

muestra_pos = []
muestra_neg = []

for i in range(len(muestra)):
	if np.sign(h[i]) == 1:
		muestra_pos.append(muestra[i])
	else:
		muestra_neg.append(muestra[i])

muestra_pos = np.array(muestra_pos)
muestra_neg = np.array(muestra_neg)

print ('Bondad del resultado para un modelo lineal:\n')
print ("Ein: ", Err(muestra, y_muestra, w))

plt.scatter(muestra_pos[:,1], muestra_pos[:,2], c="blue", alpha=0.7)
plt.scatter(muestra_neg[:,1], muestra_neg[:,2], c="yellow", alpha=0.7)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# d) Ejecutar el experimento 1000 veces

# Funcion para el experimento 1000 veces

def experimento(no_lineal):

	Ein = []
	Eout = []

	for i in range(1000):

		# Generacion de la muestra

		muestra = simula_unif(1000,2,1)
		y_muestra = f_sign(muestra[:,0],muestra[:,1])

		# Introduccion de ruido

		cambios = np.arange(len(muestra))
		np.random.shuffle(cambios)

		for i in range(int(len(muestra)*0.1)):
			if y_muestra[cambios[i]] == 1:
				y_muestra[cambios[i]] = -1
			else:
				y_muestra[cambios[i]] = 1

		# Ajustar modelo de regresion lineal

		muestra = np.append(np.ones((len(muestra),1)), muestra, axis=1)
		if no_lineal:
			muestra = np.insert(muestra,muestra.shape[1], (muestra[:,1] * muestra[:,2]) ,1)
			muestra = np.insert(muestra,muestra.shape[1], (muestra[:,1] * muestra[:,1]) ,1)
			muestra = np.insert(muestra,muestra.shape[1], (muestra[:,2] * muestra[:,2]) ,1)

		w = sgd(muestra, y_muestra, 0.05, 100, 32)

		Ein.append(Err(muestra, y_muestra, w))

		# Generacion de muestra para Eout y calculo de Eout

		muestra = simula_unif(1000,2,1)
		y_muestra = f_sign(muestra[:,0],muestra[:,1])

		muestra = np.append(np.ones((len(muestra),1)), muestra, axis=1)
		if no_lineal:
			muestra = np.insert(muestra,muestra.shape[1], (muestra[:,1]* muestra[:,2]) ,1)
			muestra = np.insert(muestra,muestra.shape[1], (muestra[:,1] * muestra[:,1]) ,1)
			muestra = np.insert(muestra,muestra.shape[1], (muestra[:,2] * muestra[:,2]) ,1)

		Eout.append(Err(muestra, y_muestra, w))
	
	Ein = np.array(Ein)
	Eout = np.array(Eout)

	return Ein.mean(), Eout.mean()

# Ejecucion del experimente 1000 veces con caracteristicas lineales

print("Ejecutando experimento usando caracteristicas lineales...")

Ein_media, Eout_media = experimento(False)

print ('Errores Ein y Eout medios tras 1000reps del experimento con caracteristicas lineales:\n')
print ("Ein media: ", Ein_media)
print ("Eout media: ", Eout_media)

input("\n--- Pulsar tecla para continuar ---\n")

# Ejecucion del experimente 1000 veces con caracteristicas no lineales

print("Ejecutando experimento usando caracteristicas no lineales...")

Ein_media, Eout_media = experimento(True)

print ('Errores Ein y Eout medios tras 1000reps del experimento con caracteristicas no lineales:\n')
print ("Ein media: ", Ein_media)
print ("Eout media: ", Eout_media)

input("\n--- Pulsar tecla para terminar ---\n")