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

    Parâmetros:

        File_name (str): Nome do arquivo, deve estar na pasta dados.
        **kwargs: Argumentos nomeados opcionais:
            - comprimentos (list/tuple/set, opcional):  Objeto contendo comprimentos de onda. Se não especificado, usa a primeira linha.

    Retorna:

        pandas.DataFrame: Um dataframe contendo os valores do arquivo, onde as linhas representam amostras e as colunas os respectivos comprimentos de onda.
    
    Levanta:

        ValueError: Caso o número de comprimentos de onda especificados não corresponda ao número de colunas presentes no arquivo.
        FileNotFoundError: Caso o nome do arquivo fornecido não seja encontrado na 'pasta dados'
    '''
    
    cwd = os.getcwd()
    
    # Define o caminho até a pasta dos arquivos
    directory_path = os.path.join(cwd, 'dados')

    # Constrói o caminho até o arquivo
    file_path = os.path.join(directory_path, file_name)
    
    # Checando se o arquivo existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo '{file_name}' não encontrado na pasta {directory_path}. Por favor, verifique se o arquivo esttá na pasta correta.")
    
    # Lendo o texto em um DataFrame
    comprimentos = kwargs.get('comprimentos', None)
    if comprimentos is not None:
        #caso os comprimentos sejam especificados
        # Verifica se o número de colunas é igual ao número de comprimentos especificados
        with open(file_path, 'r') as f:
            first_line = f.readline().strip().split()
            if len(kwargs['comprimentos']) != len(first_line):
                raise ValueError("O número de comprimentos especificados não corresponde ao número de colunas no arquivo.")
        df = pd.read_csv(file_path, delim_whitespace=True, names=kwargs['comprimentos'])
    else:
        #caso os comprimentos não sejam especificados
        df = pd.read_csv(file_path, delim_whitespace=True)
    
    return df

def spc(arquivo,comprimentos):
    pass

def xlsx(file_name, **kwargs):
    """
    Função para importar um arquivo .xlsx em um dataframe pandas formatado conforme o padrão do arquivo para os espectros.
    Caso não sejam especificados os comprimentos de onda, o programa considera a primeira linha como sendo os comprimentos de onda.
    Para especificar os comprimentos de onda, use o argumento comprimentos. 
    
    Parâmetros:
        file_name (str): O nome do arquivo Excel dentro da pasta 'dados'.
        **kwargs: Argumentos nomeados opcionais:
            - aba (str ou int, padrão 0): O nome ou índice da planilha a ser lida.
            - comprimentos (list/tuple/set, opcional):  Objeto contendo comprimentos de onda. Se não especificado, usa a primeira linha da planilha.
        
    Retorna:
        pandas.DataFrame: Um dataframe contendo os valores do arquivo, onde as linhas representam amostras e as colunas os respectivos comprimentos de onda.
        
    Levanta:
        FileNotFoundError: Se o arquivo especificado não for encontrado na pasta dados.
        ValueError: Se o número de comprimentos não corresponder ao número de colunas na planilha.
    """

    directory_path = os.path.join("dados", file_name)
    
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Arquivo não encontrado '{file_name}' na pasta {directory_path}. Por favor, verifique se o arquivo esttá na pasta correta.")
    
    aba = kwargs.get('aba', 0)
    comprimentos = kwargs.get('comprimentos', None)
    
    if comprimentos is not None:
        with pd.ExcelFile(directory_path) as xls:
            num_cols = len(pd.read_excel(xls, sheet_name=aba).columns)
            if len(comprimentos) != num_cols:
                raise ValueError(f"O número de comprimentos de onda ({len(comprimentos)}) não corresponde ao número de colunas na planilha ({num_cols}).")
        
        df = pd.read_excel(directory_path, sheet_name=aba, names=comprimentos)
    else:
        df = pd.read_excel(directory_path, sheet_name=aba)
        
    return df

