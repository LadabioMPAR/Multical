import json
import os
import pandas as pd
from arquivos import spec as spc
from arquivos import ref
import Pretreat as pt
import subprocess
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk
import time
from matplotlib.lines import Line2D
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
'''
Lista de bugs ^^

1- se o workspace estiver definido mas com as chaves vazias, o objeto quebra na hora de ler
    provavelmente a condição para rodar o subprocesso no __init__ pode ser melhorada
'''

class Dados_exp:
    '''
    Classe principal do repositório, ela armazena os dados de referência e absorbâncias.
    Por padrão, ela lê o arquivo 'workspace.json' e inicializa as referências e absorbâncias nele contidos. 
    Caso o workspace esteja vazio, um script para adicionar arquivos é rodado.

    Alternativamente, podem ser utilizados outros arquivos .json como workspace, contanto que o arquivo fornecido possua as chaves 'comprimentos' e 'referencias'.
    As chaves devem conter, cada uma, um array de strings com os caminhos dos respectivos arquivos.

    Atributos:
        X - Contém as absorbâncias (Dataframe do pandas)
        Y - Contém as referências (Dataframe do pandas)
        comprimentos -  Contém uma lista com os comprimentos de onda utilizados (lista de inteiros)
        analitos - Contém uma lista com os nomes dos analitos nos dados experimentais (lista de strings)

    '''

    def __init__(self, arquivo_json='workspace.json', X=[], y=[], comprimentos=None, analitos=None):
        self.comprimentos=comprimentos
        self.analitos=analitos
        self.X=X
        self.y=y
        if (self.X==[]) and (self.y==[]):
            if os.path.getsize(arquivo_json) == 0:
                subprocess.run(["python", "Import.py"])
            self.X, self.y, self.comprimentos, self.analitos = self.lendo_workspace(arquivo_json)

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
        Método simples para importar novos dados à instância da classe Dados_exp após ser inicializada.
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
        return pd.concat(self.X.copy(), axis=0, ignore_index=True)
    
    def stack_y(self):
        '''
        Empilha os valores de y
        Retorna:
            - Dataframe: valores das referências de todos os arquivos em um único dataframe
        Levanta:
            ValueError: Se não houverem referências
        '''
        if not self.y:
            raise ValueError("A lista de referências está vazia.")
        return pd.concat(self.y.copy(), axis=0, ignore_index=True)

    def pretreat(self, pretratamentos,salvar=False):
        '''
        Aplica uma lista de pré-tratamentos a cada DataFrame em X separadamente.
        Parâmetros:
            pretratamentos: lista de tuplas, cada uma contendo:
                            (nome da função de pré-tratamento, dicionário de argumentos)
            salvar: Booleano, se True salva os dataframes em arquivo e gera workspace
    
         Retorna:
            Lista: novo objeto Dados_exp
        '''
        X_tratado = []
        
        # Configurando a barra de progresso para os DataFrames
        total_steps = len(self.X) * len(pretratamentos) #n de passos
        with tqdm(total=total_steps, desc="Processando pré-tratamentos", unit="step") as pbar: #barrinha de progresso
            for idx, x in enumerate(self.X): 
                df = x.copy() #usa-se uma cópia do dataframe para evitar mexer no objeto original
                for nome_pretratamento, params in pretratamentos:
                    # Atualizando a descrição da barra de progresso para o pré-tratamento e experimento atuais
                    pbar.set_description(f"Aplicando {nome_pretratamento} ao arquivo {idx + 1}/{len(self.X)}")
                    funcao_pretratamento = getattr(pt, nome_pretratamento) #Aqui o pré-tratamento é encontrado dinamicamente
                    df = funcao_pretratamento(df, **params) #Aqui realmente rodamos o pré-tratamento
                    pbar.update(1)  # Atualizando a barra de progresso para cada pré-tratamento aplicado
                X_tratado.append(df)

        if salvar: #não testei ainda
            self.salvar(nome_x="abs-pretratadas",nome_y="refs-pretratadas",workspace="workspace-pretratado")
            
        return Dados_exp(X=X_tratado,y=self.y,comprimentos=self.comprimentos,analitos=self.analitos)

    def salvar(self, nome_x="X_",nome_y="y_",workspace="workspace"):
        '''
        Função para salvar os valores de X e y de um objeto Dados_exp em arquivos de texto, também gera um novo workspace para uso futuro
        Parâmetros:
            nome_x: string, contém o prefixo no qual serão salvos os arquivos de absorbâncias

            nome_y: string, contém o prefixo no qual serão salvos os arquivos de referências

            workspace: string, nome do workspace criado
    
         Retorna:
            nada :) Ele só cria os arquivos  

        '''
        # Define o caminho até a pasta dados
        cwd = os.getcwd()
        pasta = os.path.join(cwd, 'dados')
        if not os.path.exists(pasta):
            os.makedirs(pasta)

        comprimentos_paths = []
        referencias_paths = []

        # Salvando DataFrames de X
        for i, df in enumerate(self.X):
            file_path = os.path.join(pasta, f'{nome_x}{i+1}.txt')
            df.to_csv(file_path, sep='\t', index=False)
            comprimentos_paths.append(file_path)

        # Salvando DataFrames de y
        for i, df in enumerate(self.y):
            file_path = os.path.join(pasta, f'{nome_y}{i+1}.txt')
            df.to_csv(file_path, sep='\t', index=False)
            referencias_paths.append(file_path)

        # Criando o arquivo JSON com os caminhos
        json_data = {
            "comprimentos": comprimentos_paths,
            "referencias": referencias_paths
        }

        with open(f'{workspace}.json', 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

    def plot_espectros(self,nome="fig"):
        '''
        Método para plotar os espectros contidos por matriz em X
        atualmente funciona para até 6 arquivos diferentes
        '''
        colors = ['y', 'm', 'c', 'r', 'g', 'b']
        plt.figure(figsize=(10, 6))
        for df,color in zip(self.X,colors):
            df_t=df.transpose()
            plt.plot(df_t.index,df_t,color=color,linewidth=0.3)
        legend_elements = [Line2D([0], [0], color=color, lw=2, label=f'Ensaio {i+1}') for i, color in enumerate(colors)]
        plt.legend(handles=legend_elements, fontsize=14)   
        plt.xlabel('Número de onda (cm$^{-1}$)', fontsize=22)
        plt.ylabel('Absorbância', fontsize=22)
        plt.xticks(df_t.index[::500], rotation=45)
        plt.tight_layout()
        plt.savefig(nome)
        plt.show()
    
    def LB(self):

        '''
        Método faz a análise de mínimos quadrados clássicos para verificar a aplicabilidade da lei de lambert-beer
        Plota a relação entre absorbância calculada e as absorbâncias de referência com e sem termo independente

        Retorna- Tupla com as matrizes de coeficientes K, sem e com termo independente (K sem o termo, K com termo)
        '''
        absor= self.stack_x()
        x=self.stack_y()

        nd, nl = absor.shape

        # Lambert-Beer sem termo independente
        xone = x
        Ks = np.linalg.lstsq(xone, absor, rcond=None)[0]
        absorc = np.dot(xone, Ks)

        # convertendo para arrays
        xymax = max(np.max(absor.values), np.max(absorc))
        xymin = min(np.min(absor.values), np.min(absorc))

        plt.figure(1)
        plt.plot(absor, absorc, 'o', markersize=5, markeredgewidth=1, markeredgecolor='black')
        plt.plot([xymin, xymax], [xymin, xymax], '-k')
        plt.xlabel('Absorbância de referência')
        plt.ylabel('Absorbância calculada L-B')
        plt.title('ajuste absorbância SEM termo idependente')

        # Lambert-Beer com termo independente
        xone = np.hstack((np.ones((nd, 1)), x))
        Kc = np.linalg.lstsq(xone, absor, rcond=None)[0]
        absorc = np.dot(xone, Kc)

        plt.figure(2)
        plt.plot(absor, absorc, 'o', markersize=5, markeredgewidth=1, markeredgecolor='black')
        plt.plot([xymin, xymax], [xymin, xymax], '-k')
        plt.xlabel('Absorbância de referência')
        plt.ylabel('Absorbância calculada L-B')
        plt.title('ajuste absorbância COM termo idependente')

        plt.show()
        return (Ks,Kc)
    
    def PCA(self,plots=False):
        """
        Código para implementação do PCA

        Parâmetros:

            plots: Valor booleano indicando se deve-se plotar os gráficos usuais ou apenas retornas os valores. Falso por padrão

        Retorna:
            eigvec, eigval, var_rel, var_ac
            eigvec (numpy.ndarray): matriz de componentes principais, autovetores
            eigval (numpy.ndarray): autovalores da matriz de covariância
            var_rel (numpy.ndarray): Lista de variância relativa das PCs
            var_ac (numpy.ndarray): Lista de variância acumulada das PCs

        """
        absor = self.stack_x()  
        lambda_values = self.comprimentos 

        # Normalizando
        anorm = (absor - np.mean(absor, axis=0)) / np.std(absor, axis=0)

        # Matriz de covariância
        cov_matrix = np.cov(anorm, rowvar=False)

        # Autovalores e autovetores
        eigval, eigvec = np.linalg.eigh(cov_matrix)

        # Ordenando valores e vetores em ordem decrescente
        sorted_indices = np.argsort(eigval)[::-1]
        eigval = eigval[sorted_indices]
        eigvec = eigvec[:, sorted_indices]

        # Variância explicada
        vartot = np.sum(eigval)
        var_rel = eigval / vartot
        var_ac = np.cumsum(var_rel)

        # Selecionando as variâncias até um certo limite para print
        limite = 0.9999
        maxind = np.argmax(var_ac >= limite) + 1
        
        if plots:
            
            # Plotando as 3 primeiras PC's
            plt.figure(1)
            plt.plot(lambda_values, eigvec[:, 0], label='PC1')
            plt.plot(lambda_values, eigvec[:, 1], label='PC2')
            plt.plot(lambda_values, eigvec[:, 2], label='PC3')
            plt.axhline(0, color='k')
            plt.xlabel('Comprimento')
            plt.ylabel('PC')
            plt.title('Componentes principais')
            plt.legend()
            

            # plot para as duas primeiras componentes principais
            pc1 = np.dot(anorm, eigvec[:, 0])
            pc2 = np.dot(anorm, eigvec[:, 1])
            plt.figure(2)
            plt.scatter(pc1, pc2, marker='x')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('Componentes principais')
            plt.show(block=False)
            # Criando uma janela com tkinter
            root = tk.Tk()
            root.title('Variância Explicada')

            # Criando a tabela
            cols = ('PC#', 'Variância Relativa (%)', 'Variância Acumulada (%)')
            tree = ttk.Treeview(root, columns=cols, show='headings')

            for col in cols:
                tree.heading(col, text=col)

            # Inserindo os dados na tabela
            for i in range(maxind):
                tree.insert("", "end", values=(f'{i+1}', f'{var_rel[i]*100:.3f}', f'{var_ac[i]*100:.3f}'))

            tree.pack(expand=True, fill='both')
            
            # Iniciando a interface
            root.mainloop()
        return eigvec, eigval, var_rel, var_ac[:maxind]
    
teste=Dados_exp()
print(teste.stack_x().shape)
t0 = time.time()

varac=teste.PCA(plots=True)[3]

t1 = time.time()

norm = (teste.stack_x() - teste.stack_x().mean()) / teste.stack_x().std()
from sklearn.decomposition import PCA
pca_ = PCA()
principalComponents_ = pca_.fit_transform(norm)

t2 = time.time()

print(f"tempo para o meu codigo: {t1-t0}")
print(f"tempo para o scikit learn: {t2-t1}")
print(len(varac))
print(len(pca_.explained_variance_ratio_))
#usando o scikit learn, nós não normalizamos, por isso os resultados dão diferentes
