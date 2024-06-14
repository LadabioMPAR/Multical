import pandas as pd
from scipy.signal import savgol_filter


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


def sav_gol(df, janela, polyorder,derivada=0):
    """
    Função que aplica o filtro Savitzky-Golay em cada coluna de um DataFrame.

    Parâmetros:
    df: Dataframe
    O DataFrame que contém os dados a serem filtrados.
    janela: int
    O tamanho da janela do filtro.
    polyorder: int
    A ordem do polinômio utilizado para ajustar os dados.

    Retorna:
    Um novo DataFrame com as colunas filtradas.
    """
    if janela % 2 == 0:
        raise ValueError("janela deve ser um número ímpar")
    if janela < 1 or janela > len(df):
        raise ValueError("janela deve estar entre 1 e o comprimento do sinal")
    if polyorder < 0 or polyorder >= janela:
        raise ValueError("polyorder deve ser maior ou igual a zero e menor que window_length")

    # Cria uma cópia do DataFrame para evitar modificar o original
    df_filtrada = df.copy()

    # Aplica o filtro de Savitzky-Golay a cada coluna do Dataframe
    for column in df_filtrada:
        df_filtrada[column] = savgol_filter(df_filtrada[column], janela, polyorder,deriv=derivada)

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

