import pandas as pd
from scipy.signal import savgol_filter
import numpy as np

def media_movel(x, tam_janela):
    """
    Apply a moving average to each column of a DataFrame.

    :param x: DataFrame containing the data to which the moving average will be applied.
    :type x: pandas.DataFrame
    :param tam_janela: The window size for calculating the moving average.
    :type tam_janela: int

    :returns: DataFrame with the moving average applied to each column.
    :rtype: pandas.DataFrame

    :raises Exception: If an error occurs during the moving average calculation.
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
    Apply the Savitzky-Golay filter to each row of a DataFrame.

    :param df: The DataFrame containing the data to be filtered.
    :type df: pandas.DataFrame
    :param janela: The size of the filter window (must be odd).
    :type janela: int
    :param polyorder: The polynomial order to use for smoothing the data (default is 1).
    :type polyorder: int, optional
    :param derivada: The order of the derivative to compute (default is 0, i.e., no derivative).
    :type derivada: int, optional

    :returns: A new DataFrame with the filtered rows.
    :rtype: pandas.DataFrame

    :raises ValueError: If the window size is even or invalid.
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
    Slice a DataFrame based on lower and upper bounds of the x-axis.

    :param df: The original DataFrame.
    :type df: pandas.DataFrame
    :param lower_bound: The lower bound of the slice.
    :type lower_bound: float
    :param upper_bound: The upper bound of the slice.
    :type upper_bound: float

    :returns: A new DataFrame containing only the data within the specified interval.
    :rtype: pandas.DataFrame
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
    Process a DataFrame to remove rows where absolute values exceed a given threshold.

    :param df: The DataFrame to be processed.
    :type df: pandas.DataFrame
    :param maxAbs: The maximum absolute value allowed (default is 0.5).
    :type maxAbs: float, optional
    :param action: A flag to determine if rows should be removed (default is 1).
    :type action: int, optional

    :returns: The processed DataFrame.
    :rtype: pandas.DataFrame
    """
    
    # Encontra índices onde os valores absolutos excedem maxAbs
    indices = df[df.abs() >= maxAbs].dropna(how='all').index
    
    # Verifica se existem tais índices
    if not indices.empty:
        df = df.drop(indices)
    
    return df

def BLCtr(df, ini_lamb, final_lamb, Abs_value):
    """
    Correct the baseline in an absorbance matrix.

    :param df: DataFrame containing the absorbance matrix.
    :type df: pandas.DataFrame
    :param ini_lamb: The starting wavelength for the non-informative region (column name).
    :type ini_lamb: int or str
    :param final_lamb: The ending wavelength for the non-informative region (column name).
    :type final_lamb: int or str
    :param Abs_value: The average absorbance value in the non-informative region.
    :type Abs_value: float

    :returns: DataFrame with the corrected baseline.
    :rtype: pandas.DataFrame
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

