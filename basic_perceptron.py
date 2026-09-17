import numpy as np
listarecompensa = [10,0,0,-1,0,0,5,-1,-1]
def filtrarrecompensa(lista):
    somatotal = sum(lista)
    mediarecompensa = np.mean(lista)
    positivo = [positivo for positivo in lista if positivo > 0]
    return positivo,mediarecompensa,somatotal

entradas = [1.0, 2.0, -1.0]
pesos = [0.5, -0.5, 1.0]
vies = 0.2
target = 1
def leaky_relu(x):
    return np.maximum(0.01 * x, x)
def leaky_relu_derivative(x):
    # Retorna 1 se o valor for maior que 0, e 0.01 caso contrário
    return np.where(x > 0, 1.0, 0.01)
class perceptron:
    def __init__(self):
        self.pesos = pesos
        self.bias = vies

    def forward(self, x):
        self.x = x  # Guarda a entrada original para usar no backward
        self.soma_linear = np.dot(x, self.pesos) + self.bias
        self.saida = leaky_relu(self.soma_linear)
        return self.saida

    def backpropagation(self, alvo, lr=0.1):
        erro = self.saida - alvo
        delta = erro * leaky_relu_derivative(self.soma_linear)
        gradiente_pesos = delta * self.x
        self.pesos = self.pesos-lr * gradiente_pesos
        self.bias = self.bias-lr * delta



neuronio = perceptron()
for i in range(1000):
    forward = neuronio.forward(np.array(entradas))
    print(forward)
    neuronio.backpropagation(target)
