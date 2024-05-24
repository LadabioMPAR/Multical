import pandas as pd

def apply_moving_average_to_csv(input_file, output_file, window_size):
    """

    """
    try:
        # Lendo o arquivo CSV com delimitador de espaços
        df = pd.read_csv(input_file, delim_whitespace=True)
    except FileNotFoundError:
        print(f"Erro: O arquivo {input_file} não foi encontrado.")
        return
    except Exception as e:
        print(f"Erro ao ler o arquivo {input_file}: {e}")
        return

    try:
        # Aplicando a média móvel em cada coluna
        df_moving_avg = df.apply(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    except Exception as e:
        print(f"Erro ao calcular a média móvel: {e}")
        return

    try:
        # Salvando o DataFrame resultante em um arquivo CSV não compactado
        df_moving_avg.to_csv(output_file, index=False)
        output_file = 'teste_abs01.txt'
        print(f"Média móvel salva com sucesso em {output_file}")
    except Exception as e:
        print(f"Erro ao salvar o arquivo {output_file}: {e}")
    # arquivo resultante é muito grande e faltou plotar gráfico


# Definindo o período da média móvel
window_size = 4
input_file = 'abs01.txt'
output_file = 'teste_abs01.txt'

apply_moving_average_to_csv(input_file, output_file, window_size)
