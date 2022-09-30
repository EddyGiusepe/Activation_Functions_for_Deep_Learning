'''
Data Scientist.: Dr.Eddy Giusepe Chirinos Isidro
'''
import matplotlib.pyplot as plt
from numpy import size
#% matplotlib inline
from math import exp


print("###########################################")
print("Função de ativação Linear retificada (ReLU)")
print("###########################################")
 
# Definimos a nossa função
def rectified(x):
	return max(0.0, x)
 
# Nosso input de Dados
inputs = [x for x in range(-10, 10)]
# Calculamos os Outputs
outputs = [rectified(x) for x in inputs]
# Plotamos nossos Inputs vs Outputs
plt.plot(inputs, outputs)
plt.title("ReLU", c="red", size=20)
plt.grid(True)
plt.show()



print("###########################")
print("Função de ativação Sigmoide")
print("###########################")
 
# Definimos a nossa função
def sigmoid(x):
	return 1.0 / (1.0 + exp(-x))
 
# Inputs
inputs = [x for x in range(-10, 10)]
# Outputs
outputs = [sigmoid(x) for x in inputs]
# plotamos inputs vs outputs
plt.plot(inputs, outputs)
plt.title("Sigmoide", c="red", size=20)
plt.grid(True)
plt.show()



print("###########################")
print("Função de ativação Sigmoide")
print("###########################")

# Função de ativação tanh 
def tanh(x):
	return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 
# Inputs
inputs = [x for x in range(-10, 10)]
# Outputs
outputs = [tanh(x) for x in inputs]
# plotando inputs vs outputs
plt.plot(inputs, outputs)
plt.title("Tanh", c="red", size=20)
plt.grid(True)
plt.show()

