import pandas as pd
from scipy.signal import savgol_filter
import numpy as np

def media_movel(x, tam_janela):
    """
    Aplica a média móvel a cada coluna de um DataFrame.

    Parâmetros:
    x : DataFrame
        DataFrame contendo os dados aos quais a média móvel será aplicada.
    tam_janela : int
        Tamanho da janela usada para calcular a média móvel.

    Retorna:
    DataFrame
        DataFrame com a média móvel aplicada a cada coluna.
    """
    x_copiado = x.copy()
    try:
        # Aplicando a média móvel em cada coluna
        df_media_movel = x_copiado.apply(lambda x_copiado: x_copiado.rolling(window=tam_janela, min_periods=1).mean())
    except Exception as e:
        print(f"Erro ao calcular a média móvel: {e}")
    return df_media_movel


from scipy.signal import savgol_filter


def sav_gol(df, janela, polyorder=1, derivada=0):
    """
    Função que aplica o filtro Savitzky-Golay em cada linha de um DataFrame.

    Parâmetros:
    df: Dataframe
        O DataFrame que contém os dados a serem filtrados.
    janela: int
        O tamanho da janela do filtro (deve ser ímpar).
    polyorder: int
        A ordem do polinômio utilizado para ajustar os dados.
    derivada: int
        Ordem da derivada a ser calculada (padrão é 0, ou seja, sem derivada).

    Retorna:
    Um novo DataFrame com as linhas filtradas.
    """
    if janela % 2 == 0:
        raise ValueError("janela deve ser um número ímpar")
    if janela < 1 or janela > df.shape[1]:
        raise ValueError("janela deve estar entre 1 e o número de colunas")
    if polyorder < 0 or polyorder >= janela:
        raise ValueError("polyorder deve ser maior ou igual a zero e menor que a janela")

    # Aplica o filtro Savitzky-Golay em cada linha e converte o resultado para um DataFrame
    df_filtrada = df.apply(lambda row: pd.Series(savgol_filter(row, janela, polyorder, deriv=derivada)), axis=1)

    # Define os mesmos índices e colunas
    df_filtrada.columns = df.columns
    df_filtrada.index = df.index

    return df_filtrada


def cut(df, lower_bound, upper_bound):
    """
    Corta um DataFrame do pandas com base nos limites inferior e superior no eixo das abcissas.

    Parâmetros:
    df (pd.DataFrame): O DataFrame original.
    lower_bound (float): O limite inferior do corte.
    upper_bound (float): O limite superior do corte.

    Retorna:
    pd.DataFrame: Um novo DataFrame contendo apenas os dados dentro do intervalo especificado.
    """
    # Faz uma cópia para não dar problema
    df_cut = df.copy()
    #Transformando nomes das colunas em valores inteiros
    x_values = [int(x) for x in df.columns.values]
    #Definindo uma lista de valores que não estão entre os limites upper e lower
    lista_para_drop=[str(i) for i in list(filter(lambda i : i < (lower_bound) or i >= (upper_bound+1), x_values))]
    # descartando as colunas que não estão entre os limites
    return df_cut.drop(columns=lista_para_drop)

def cut_abs(df, maxAbs=0.5, action=1):
    """
    Função para processar um DataFrame do pandas similar ao trecho de código Scilab.
    
    Parâmetros:
    - df: DataFrame do pandas a ser processado
    - maxAbs: valor absoluto máximo permitido (o padrão é 0.5)
    - action: flag para determinar se as linhas devem ser removidas (o padrão é 1)
    - graph: flag para plotagem (não usado nesta função, o padrão é 1)
    
    Retorna:
    - df: DataFrame do pandas processado
    """
    
    # Encontra índices onde os valores absolutos excedem maxAbs
    indices = df[df.abs() >= maxAbs].dropna(how='all').index
    
    # Verifica se existem tais índices
    if not indices.empty:
        df = df.drop(indices)
    
    return df

def BLCtr(df, ini_lamb, final_lamb, Abs_value):
    """
    Função para correção de linha de base em uma matriz de absorbância.

    Parâmetros:
    - absor: DataFrame do pandas contendo a matriz de absorbância
    - ini_lamb: comprimento de onda inicial para a região não informativa (nome da coluna)
    - final_lamb: comprimento de onda final para a região não informativa (nome da coluna)
    - Abs_value: valor médio de absorbância na região não informativa

    Retorna:
    - absor: DataFrame do pandas com a linha de base corrigida
    """
    
    # Encontra os índices das colunas onde lambda está dentro do intervalo especificado
    ind_ini = df.columns.get_loc(str(ini_lamb))
    ind_final = df.columns.get_loc(str(ini_lamb))
    
    # Seleciona as colunas com base nos índices encontrados
    ind = df.columns[ind_ini:ind_final+1]
    
    # Calcula a diferença média na região não informativa
    Dabs = df.loc[:, ind].mean(axis=1) - Abs_value
    
    # Cria uma matriz de diferença média para ajustar a matriz de absorbância
    DabsMatr = np.tile(Dabs.values[:, np.newaxis], (1, df.shape[1]))
    
    # Ajusta a matriz de absorbância subtraindo a matriz de diferença
    absor_corrected = df - DabsMatr
    
    return absor_corrected

