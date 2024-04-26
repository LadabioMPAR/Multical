import os
import pandas as pd
'''
Código para importação dos arquivos de absorbância (espectros)
Cada função deve ser nomeada conforme o tipo de arquivo que está sendo lido e deve retornar um dataframe (pandas) com os comprimentos de onda como label
Linhas=amostras, colunas=comprimentos

'''
def txt(file_name, **kwargs):
    '''
    Função para importar um arquivo .txt em um dataframe pandas formatado conforme o padrão do arquivo para os espectros.
    Caso não sejam especificados os comprimentos de onda, o programa considera a primeira linha como sendo os comprimentos de onda.
    Para especificar os comprimentos de onda, use o argumento comprimentos. 
    '''
    
    cwd = os.getcwd()
    
    # Define o caminho até a pasta dos arquivos
    directory_path = os.path.join(cwd, 'dados')

    # Constrói o caminho até o arquivo
    file_path = os.path.join(directory_path, file_name)
    
    # Checando se o arquivo existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo '{file_name}' não encontrado na pasta {directory_path}")
    
    # Lendo o texto em um DataFrame
    if 'comprimentos' in kwargs:
        # Verifica se o número de colunas é igual ao número de comprimentos especificados
        with open(file_path, 'r') as f:
            first_line = f.readline().strip().split()
            if len(kwargs['comprimentos']) != len(first_line):
                raise ValueError("O número de comprimentos especificados não corresponde ao número de colunas no arquivo.")
        df = pd.read_csv(file_path, delim_whitespace=True, names=kwargs['comprimentos'])
    else:
        df = pd.read_csv(file_path, delim_whitespace=True)
    
    return df

def spc(arquivo,comprimentos):
    pass

def xlsx(arquivo,comprimentos):
    pass

