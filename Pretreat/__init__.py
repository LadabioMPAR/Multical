import pandas as pd

def media_movel(x, window_size):
    """

    """
    x_copiado = x.copy()
    try:
        # Aplicando a média móvel em cada coluna
        df_moving_avg = x_copiado.apply(lambda x_copiado: x_copiado.rolling(window=window_size, min_periods=1).mean())
    except Exception as e:
        print(f"Erro ao calcular a média móvel: {e}")
    return df_moving_avg
