x=5
y=10.3
z="x+y"
M=[x,y,z]

def soma(a,b):
    return a+b


import pandas as pd

class Carro():
    def __init__(self, marca, modelo):
        self.marca = marca
        self.modelo = modelo

    def exibir_info(self):
        return f"Marca: {self.marca}, Modelo: {self.modelo}"
    def buzinar(self):
        return f"Beep beep!{self.modelo}"

meu_carrin=Carro("Toyota", "Corolla")


print(meu_carrin.buzinar())