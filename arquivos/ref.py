import os
import pandas as pd

'''
Código para importação dos arquivos de referência (concentração)
Cada função deve ser nomeada conforme o tipo de arquivo que está sendo lido e deve retornar um dataframe (pandas) com os analitos como label
Linhas=amostras, colunas=analitos

'''
def txt(file_name, **kwargs):
    '''
    Função para importar um arquivo .txt em um dataframe pandas formatado conforme o padrão do arquivo para os analitos.
    Caso não sejam especificados os analitos, o programa considera a primeira linha como sendo os analitos.
    Para especificar os analitos, use o argumento analitos. 

    Parâmetros:

        File_name (str): Nome do arquivo, deve estar na pasta dados
        analitos (list/tuple/set, opcional):  Objeto contendo analitos.

    Retorna:

        pandas.DataFrame: Um dataframe contendo os valores do arquivo, onde as linhas representam amostras e as colunas os respectivos analitos
    
    Levanta:

        ValueError: Caso o número de analitos especificados não corresponda ao número de colunas presentes no arquivo
        FileNotFoundError: Caso o nome do arquivo fornecido não seja encontrado na 'pasta dados'
    '''
    
    cwd = os.getcwd()
    
    # Define o caminho até a pasta dos arquivos
    directory_path = os.path.join(cwd, 'dados')

    # Constrói o caminho até o arquivo
    file_path = os.path.join(directory_path, file_name)
    
    # Checando se o arquivo existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo '{file_name}' não encontrado na pasta {directory_path}")
    
    # Escrevendo o texto em um DataFrame
    if 'analitos' in kwargs:
        # Verifica se o número de colunas é igual ao número de analitos especificados
        with open(file_path, 'r') as f:
            first_line = f.readline().strip().split()
            if len(kwargs['analitos']) != len(first_line):
                raise ValueError("O número de analitos especificados não corresponde ao número de colunas no arquivo.")
        df = pd.read_csv(file_path, delim_whitespace=True, names=kwargs['analitos'])
    else:
        df = pd.read_csv(file_path, delim_whitespace=True)
    
    return df


def spc(arquivo,analitos):
    pass

def xlsx(file_name, **kwargs):
    """
    Função para importar um arquivo .xlsx em um dataframe pandas formatado conforme o padrão do arquivo para os espectros.
    Caso não sejam especificados os analitos, o programa considera a primeira linha como sendo os analitos.
    Para especificar os analitos, use o argumento analitos. 
    
    Parâmetros:
        file_name (str): O nome do arquivo Excel dentro da pasta 'dados'.
        **kwargs: Argumentos nomeados opcionais:
            - aba (str ou int, padrão 0): O nome ou índice da planilha a ser lida.
            - analitos (list/tuple/set, opcional):  Objeto contendo analitos. Se não especificado, usa a primeira linha da planilha.
        
    Retorna:
        pandas.DataFrame: Um dataframe contendo os valores do arquivo, onde as linhas representam amostras e as colunas os respectivos analitos.
        
    Levanta:
        FileNotFoundError: Se o arquivo especificado não for encontrado na pasta dados.
        ValueError: Se o número de analitos não corresponder ao número de colunas na planilha.
    """

    directory_path = os.path.join("dados", file_name)
    
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Arquivo não encontrado '{file_name}' na pasta {directory_path}. Por favor, verifique se o arquivo esttá na pasta correta.")
    
    aba = kwargs.get('aba', 0)
    analitos = kwargs.get('analitos', None)
    
    if analitos is not None:
        with pd.ExcelFile(directory_path) as xls:
            num_cols = len(pd.read_excel(xls, sheet_name=aba).columns)
            if len(analitos) != num_cols:
                raise ValueError(f"O número de analitos ({len(analitos)}) não corresponde ao número de colunas na planilha ({num_cols}).")
        
        df = pd.read_excel(directory_path, sheet_name=aba, names=analitos)
    else:
        df = pd.read_excel(directory_path, sheet_name=aba)
        
    return df
