import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def savitzky_golay_first_derivative(y, window_length, polyorder):
    """
    Aplica o filtro de Savitzky-Golay para calcular a primeira derivada de um conjunto de dados.

    Parâmetros:
    y : array_like
        Sinal de entrada.
    window_length : int
        O tamanho da janela do filtro (deve ser um número ímpar).
    polyorder : int
        A ordem do polinômio a ser ajustado aos dados.

    Retorna:
    y_deriv : ndarray
        Primeira derivada do sinal suavizado.
    """
    if window_length % 2 == 0:
        raise ValueError("window_length deve ser um número ímpar")
    if window_length < 1 or window_length > len(y):
        raise ValueError("window_length deve estar entre 1 e o comprimento do sinal")
    if polyorder < 0 or polyorder >= window_length:
        raise ValueError("polyorder deve ser maior ou igual a zero e menor que window_length")

    y_deriv = savgol_filter(y, window_length, polyorder, deriv=1)
    return y_deriv

# Exemplo de uso da função
if __name__ == "__main__":
    # Exemplo de dados: Sinal com ruído
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, x.size)

    # Parâmetros do filtro Savitzky-Golay
    window_length = 11  # Tamanho da janela (deve ser ímpar)
    polyorder = 3       # Ordem do polinômio

    # Calculando a primeira derivada usando a função definida
    y_deriv = savitzky_golay_first_derivative(y, window_length, polyorder)
    print(y_deriv)