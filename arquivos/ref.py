import os
import pandas as pd
"""
Module for importing reference files (concentration data).
Each function should be named according to the file type being read and should return a pandas DataFrame
with analytes as labels. Rows represent samples, and columns represent analytes.
"""
def txt(file_name, **kwargs):
    """
    Imports a `.txt` file into a pandas DataFrame formatted for analytes.

    If analytes are not specified, the program considers the first line as analytes.
    To specify analytes, use the 'analitos' argument.

    :param file_name: Name of the file located in the 'dados' folder.
    :type file_name: str
    :param **kwargs: Optional arguments:
        - analitos (list/tuple/set, optional): Object containing analytes.

    :return: DataFrame with samples as rows and analytes as columns.
    :rtype: pandas.DataFrame
    :raises ValueError: If the number of specified analytes doesn't match the number of columns in the file.
    :raises FileNotFoundError: If the file is not found in the 'dados' folder.
    """
    
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
    """
    Placeholder function for importing files in '.spc' format.

    :param arquivo: File to be imported.
    :type arquivo: str
    :param analitos: List of analytes.
    :type analitos: list
    """
    pass

def xlsx(file_name, **kwargs):
    """
    Imports an `.xlsx` file into a pandas DataFrame formatted for analytes.

    If analytes are not specified, the program considers the first line as analytes.
    To specify analytes, use the 'analitos' argument.

    :param file_name: Name of the Excel file located in the 'dados' folder.
    :type file_name: str
    :param **kwargs: Optional named arguments:
        - aba (str or int, default=0): Sheet name or index to be read.
        - analitos (list/tuple/set, optional): Object containing analytes.

    :return: DataFrame with samples as rows and analytes as columns.
    :rtype: pandas.DataFrame
    :raises FileNotFoundError: If the file is not found in the 'dados' folder.
    :raises ValueError: If the number of analytes doesn't match the number of columns in the sheet.
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
