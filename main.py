import json
import os
import pandas as pd
from arquivos import spec as spc
from arquivos import ref
import subprocess
import time
start=time.time()
class Modelo:
    '''
    Classe principal do repositório, ela armazena os dados de referência e absorbâncias.
    por padrão, ela lê o arquivo 'workspace.json' e inicializa as referências e absorbâncias nele contidos. 
    Caso o workspace esteja vazio, um script para adicionar arquivos é rodado.

    Alternativamente, podem ser utilizados outros arquivos .json como workspace, contanto que o arquivo fornecido possua as chaves 'comprimentos' e 'referencias'.
    As chaves devem conter, cada uma, um array de strings com os caminhos dos respectivos arquivos.

    Atributos:
        X - Contém as absorbâncias (Dataframe do pandas)
        Y - Contém as referências (Dataframe do pandas)
        comprimentos -  Contém uma lista com os comprimentos de onda utilizados (lista de inteiros)
        analitos - Contém uma lista com os nomes dos analitos no modelo (lista de strings)

    '''
    def __init__(self, arquivo_json='workspace.json', comprimentos=None, analitos=None):
        self.comprimentos=comprimentos
        self.analitos=analitos
        if os.path.getsize(arquivo_json) == 0:
            subprocess.run(["python", "Import.py"])
        self.X, self.Y, self.comprimentos, self.analitos = self.lendo_workspace(arquivo_json)

    def lendo_workspace(self, arquivo_json):
        '''
        Método para ler os arquivos contidos no workspace.
        Os arquivos são lidos conforme caminho contido nas chaves 'comprimentos' e 'referencias' de workspace.json
        '''
        if not os.path.exists(arquivo_json):
            raise FileNotFoundError(f"O arquivo {arquivo_json} não foi encontrado.")
        
        with open(arquivo_json, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        if 'comprimentos' not in data:
            raise KeyError(f"A chave 'comprimentos' não foi encontrada no arquivo {arquivo_json}.")
        
        if 'referencias' not in data:
            raise KeyError(f"A chave 'referencias' não foi encontrada no arquivo {arquivo_json}.")

        comprimentos = data['comprimentos']
        referencias = data['referencias']
        
        # Processamento dos arquivos para X (absorbâncias) e Y (referências)
        X = [self.tipo_arquivo(comprimento, for_x=True) for comprimento in comprimentos]
        Y = [self.tipo_arquivo(referencia, for_x=False) for referencia in referencias]
        
        # Verifica se todos os DataFrames em X possuem os mesmos comprimentos de onda
        if len(X) > 1:
            nomes_colunas = set(X[0].columns)
            for df, caminho_arquivo in zip(X[1:], comprimentos[1:]):
                if set(df.columns) != nomes_colunas:
                    raise ValueError(f"O arquivo {caminho_arquivo} de absorbância não possui comprimentos de onda consistente com os demais")
        
        # Verifica se todos os DataFrames em Y apresentam a mesma quantidade de analitos
        num_colunas = len(Y[0].columns)
        for df, caminho_arquivo in zip(Y[1:], referencias[1:]):
            if len(df.columns) != num_colunas:
                raise ValueError(f"O arquivo {caminho_arquivo} de referência não apresenta número de analitos consistente com os demais")
        comp_ondas = [int(coluna) for coluna in X[0].columns.tolist()] if X else []
        analit_ref = Y[0].columns.tolist() if Y else []
        
        return X, Y, comp_ondas, analit_ref

    def tipo_arquivo(self, caminho_arquivo, for_x=True):
        '''
        Método usado na leitura do workspace
        Serve para identificar a extensão dos arquivos no workspace e lê-los corretamente em um dataframe.
        Arquivos de referência vêm da biblioteca ref e comprimentos da bilbioteca spc.

        Para adicionar uma extensão:
            1-tenha certeza que a função para importar um arquivo da extensão esteja corretamente implementada nos arquivos ref.py e spec.py.
            2- Adicione a linha de código:
                        elif extensao == '.nova_extensao':
                            return spc.nova_extensao(caminho_arquivo) if for_x else ref.nova_extensao(caminho_arquivo)

        '''
        extensao = os.path.splitext(caminho_arquivo)[1].lower()  # Obtém a extensão do arquivo em letras minúsculas
        if extensao in ['.txt', '.dat']:
            return spc.txt(caminho_arquivo) if for_x else ref.txt(caminho_arquivo) 
        elif extensao == '.xlsx':
            return spc.xlsx(caminho_arquivo) if for_x else ref.xlsx(caminho_arquivo)
        else:
            raise ValueError(f"Extensão não suportada para o arquivo: {caminho_arquivo}")
        
    def novo_dado(self, X, Y=None, comprimento=None, analito=None):
        '''
        Método simples para importar novos dados à instância da classe Modelo após ser inicializada.
        '''
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X deve ser um dataframe do pandas.")
        
        if Y is not None and not isinstance(Y, pd.DataFrame):
            raise ValueError("Y deve ser um dataframe do pandas.")
        
        self.X.append(X)
        if Y is not None:
            self.Y.append(Y)
        if comprimento is not None:
            self.comprimentos.append(comprimento)
        if analito is not None:
            self.analitos.append(analito)
    def stack_x(self):
        '''
        Empilha os valores de X
        Retorna:
            - Dataframe: valores das absorbâncias de todos os arquivos em um único dataframe
        Levanta:
            ValueError: Se não houverem absorbâncias
        '''
        if not self.X:
            raise ValueError("A lista de absorbâncias está vazia.")
        return pd.concat(self.X, axis=0, ignore_index=True)
    def stack_y(self):
        '''
        Empilha os valores de X
        Retorna:
            - Dataframe: valores das referências de todos os arquivos em um único dataframe
        Levanta:
            ValueError: Se não houverem referências
        '''
        if not self.Y:
            raise ValueError("A lista de referências está vazia.")
        return pd.concat(self.Y, axis=0, ignore_index=True)

