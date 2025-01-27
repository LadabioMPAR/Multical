import os
import pandas as pd
"""
Module for importing absorbance files (spectra).
Each function should be named according to the file type being read and should return a pandas DataFrame
with wavelengths as labels. Rows represent samples, and columns represent wavelengths.
"""
def txt(file_name, **kwargs):
    """
    Imports a `.txt` file into a pandas DataFrame formatted for spectra.

    If wavelengths are not specified, the program considers the first line as wavelengths.
    To specify wavelengths, use the 'comprimentos' argument.

    :param file_name: Name of the file located in the 'dados' folder.
    :type file_name: str
    :param **kwargs: Optional named arguments:
        - comprimentos (list/tuple/set, optional): Object containing wavelengths. If not specified, uses the first line.

    :return: DataFrame with samples as rows and wavelengths as columns.
    :rtype: pandas.DataFrame
    :raises ValueError: If the number of specified wavelengths doesn't match the number of columns in the file.
    :raises FileNotFoundError: If the file is not found in the 'dados' folder.
    """
    
    cwd = os.getcwd()
    
    # Define o caminho até a pasta dos arquivos
    directory_path = os.path.join(cwd, 'dados')

    # Constrói o caminho até o arquivo
    file_path = os.path.join(directory_path, file_name)
    
    # Checando se o arquivo existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo '{file_name}' não encontrado na pasta {directory_path}. Por favor, verifique se o arquivo está na pasta correta.")
    
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
    """
    Placeholder function for importing files in '.spc' format.

    :param arquivo: File to be imported.
    :type arquivo: str
    :param comprimentos: List of wavelengths.
    :type comprimentos: list
    """
    pass

def xlsx(file_name, **kwargs):
    """
    Imports an `.xlsx` file into a pandas DataFrame formatted for spectra.

    If wavelengths are not specified, the program considers the first line as wavelengths.
    To specify wavelengths, use the 'comprimentos' argument.

    :param file_name: Name of the Excel file located in the 'dados' folder.
    :type file_name: str
    :param **kwargs: Optional named arguments:
        - aba (str or int, default=0): Sheet name or index to be read.
        - comprimentos (list/tuple/set, optional): Object containing wavelengths. If not specified, uses the first line.

    :return: DataFrame with samples as rows and wavelengths as columns.
    :rtype: pandas.DataFrame
    :raises FileNotFoundError: If the file is not found in the 'dados' folder.
    :raises ValueError: If the number of wavelengths doesn't match the number of columns in the sheet.
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

