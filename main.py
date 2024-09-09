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
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA as sklearnPCA
from matplotlib.gridspec import GridSpec
from Calibration import Multical
from Infer import Infer


'''
Lista de bugs ^^

1- se o workspace estiver definido mas com as chaves vazias, o objeto quebra na hora de ler
    provavelmente a condição para rodar o subprocesso no __init__ pode ser melhorada

2- No PCA, se você fechar a tabela antes de fechar os gráficos, o python quebra xDD
'''

class Dados_exp:
    """
    Classe para manipulação de dados experimentais.

    Atributos:
        arquivo_json (str): Caminho para o arquivo JSON do workspace.
        X (list): Lista de DataFrames contendo os dados de absorbância.
        y (list): Lista de DataFrames contendo os dados de referência.
        comprimentos (list): Lista de comprimentos de onda.
        analitos (list): Lista de analitos.
    """

    def __init__(self, arquivo_json='workspace.json', X=[], y=[], comprimentos=None, analitos=None):
        """
        Inicializa uma instância da classe Dados_exp.

        Parâmetros:
            arquivo_json (str): Caminho para o arquivo JSON contendo o workspace.
            X (list): Lista de DataFrames de absorbância.
            y (list): Lista de DataFrames de referência.
            comprimentos (list): Lista de comprimentos de onda.
            analitos (list): Lista de analitos.
        """
        self.comprimentos=comprimentos
        self.analitos=analitos
        self.X=X
        self.Y=y
        if (X.empty if isinstance(X, pd.DataFrame) else not X) and (y.empty if isinstance(y, pd.DataFrame) else not y):
            if os.path.getsize(arquivo_json) == 0:
                subprocess.run(["python", "Import.py"])
            self.X, self.y, self.comprimentos, self.analitos = self.lendo_workspace(arquivo_json)
            #código fita-crepe pra sempre dar uma matrizona:
            self.X=self.stack_x()
            self.y= self.stack_y

    def lendo_workspace(self, arquivo_json):
        """
        Lê o arquivo JSON do workspace e retorna os dados.

        Parâmetros:
           arquivo_json (str): Caminho do arquivo JSON.

        Retorna:
           tuple: (X, Y, comprimentos, analitos) os dados extraídos do workspace.

        Levanta:
           FileNotFoundError: Se o arquivo JSON não for encontrado.
           KeyError: Se as chaves 'comprimentos' ou 'referencias' não forem encontradas no JSON.
        """
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
        """
        Identifica e lê arquivos com base na extensão para criar DataFrames.

        Parâmetros:
            caminho_arquivo (str): Caminho para o arquivo.
            for_x (bool): Se True, retorna os dados de absorbância; se False, os dados de referência.

        Retorna:
            pandas.DataFrame: Dados lidos do arquivo.

        Levanta:
            ValueError: Se a extensão do arquivo não for suportada.
        """
        extensao = os.path.splitext(caminho_arquivo)[1].lower()  # Obtém a extensão do arquivo em letras minúsculas
        if extensao in ['.txt', '.dat']:
            return spc.txt(caminho_arquivo) if for_x else ref.txt(caminho_arquivo) 
        elif extensao == '.xlsx':
            return spc.xlsx(caminho_arquivo) if for_x else ref.xlsx(caminho_arquivo)
        else:
            raise ValueError(f"Extensão não suportada para o arquivo: {caminho_arquivo}")
        
    def novo_dado(self, X, Y=None, comprimento=None, analito=None):
        """
        Adiciona novos dados ao objeto Dados_exp.

        Parâmetros:
            X (pandas.DataFrame): Dados de absorbância.
            Y (pandas.DataFrame, opcional): Dados de referência.
            comprimento (int, opcional): Comprimento de onda.
            analito (str, opcional): Nome do analito.

        Levanta:
            ValueError: Se X ou Y não forem DataFrames.
        """
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
        """
        Empilha os valores de X.

        Retorna:
            pandas.DataFrame: DataFrame contendo todos os valores de absorbância.

        Levanta:
            ValueError: Se a lista de absorbâncias estiver vazia.
        """
        if not self.X:
            raise ValueError("A lista de absorbâncias está vazia.")
        return pd.concat(self.X.copy(), axis=0, ignore_index=True)


    def stack_y(self):
        """
        Empilha os valores de y.

        Retorna:
            pandas.DataFrame: DataFrame contendo todos os valores de referência.

        Levanta:
            ValueError: Se a lista de referências estiver vazia.
        """
        if not self.y:
            raise ValueError("A lista de referências está vazia.")
        return pd.concat(self.y.copy(), axis=0, ignore_index=True)

    def pretreat(self, pretratamentos, salvar=False):
        """
        Aplica uma lista de pré-tratamentos aos dados.

        Parâmetros:
            pretratamentos (list): Lista de tuplas contendo o nome da função e parâmetros do pré-tratamento.
            salvar (bool): Se True, salva os dados tratados e o novo workspace.

        Retorna:
            Dados_exp: Objeto com o DataFrame tratado.
        """
        X_tratado = self.X.copy()  # Copia o DataFrame para evitar alterar o original
        
        # Configurando a barra de progresso para os pré-tratamentos
        total_steps = len(pretratamentos)
        with tqdm(total=total_steps, desc="Processando pré-tratamentos", unit="step") as pbar:
            for nome_pretratamento, params in pretratamentos:
                # Atualizando a descrição da barra de progresso para o pré-tratamento atual
                pbar.set_description(f"Aplicando {nome_pretratamento}")
                funcao_pretratamento = getattr(pt, nome_pretratamento)
                X_tratado = funcao_pretratamento(X_tratado, **params)
                pbar.update(1)  # Atualizando a barra de progresso para cada pré-tratamento aplicado

        if salvar:
            self.salvar(nome_x="abs-pretratadas", nome_y="refs-pretratadas", workspace="workspace-pretratado")
        
        return Dados_exp(X=X_tratado, y=self.y, comprimentos=[int(coluna) for coluna in X_tratado.columns.tolist()] if not X_tratado.empty else [], analitos=self.analitos)


    def salvar(self, nome_x="X_",nome_y="y_",workspace="workspace"):
        """
        Salva os valores de X e y em arquivos de texto e gera um novo workspace.

        Parâmetros:
            nome_x (str): Prefixo para os arquivos de absorbâncias.
            nome_y (str): Prefixo para os arquivos de referências.
            workspace (str): Nome do workspace gerado.
        """
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
        """
        Plota os espectros contidos em X.

        Parâmetros:
            nome (str): Nome da figura.
        """
        colors = ['y', 'm', 'c', 'r', 'g', 'b']
     
        plt.figure(figsize=(10, 6))
        df_t=self.X.transpose()
        plt.plot(df_t.index,df_t,color=colors[2],linewidth=0.3)
        #legend_elements = [Line2D([0], [0], color=color, lw=2, label=f'Ensaio {i+1}') for i, color in enumerate(colors)]
        #plt.legend(handles=legend_elements, fontsize=14)   
        plt.xlabel('Número de onda (cm$^{-1}$)', fontsize=22)
        plt.ylabel('Absorbância', fontsize=22)
        plt.xticks(df_t.index[::500].astype(int), rotation=45)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)  # pause para deixar renderizar

        # input pra dar um pouse
        input("Aperte enter para continuar") 
    
    def LB(self,plots=False):
        """
        Realiza a análise de mínimos quadrados clássicos para a lei de Lambert-Beer.

        Parâmetros:
            plots (bool): Se True, gera gráficos dos resultados.

        Retorna:
            tuple: Matrizes de coeficientes Ks (sem termo independente) e Kc (com termo independente).
        """
        absor= self.X
        x=self.Y

        nd, nl = absor.shape

        # Lambert-Beer sem termo independente
        xone = x
        Ks = np.linalg.lstsq(xone, absor, rcond=None)[0]
        absorc1 = np.dot(xone, Ks)

        # convertendo para arrays
        xymax = max(np.max(absor.values), np.max(absorc1))
        xymin = min(np.min(absor.values), np.min(absorc1))

        # Lambert-Beer com termo independente
        xone = np.hstack((np.ones((nd, 1)), x))
        Kc = np.linalg.lstsq(xone, absor, rcond=None)[0]
        absorc2 = np.dot(xone, Kc)
        if plots:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Plot for Lambert-Beer sem termo independente
            ax1.plot(absor, absorc1, 'o', markersize=5, markeredgewidth=1, markeredgecolor='black')
            ax1.plot([np.min(absor.values), np.max(absor.values)], [np.min(absorc1), np.max(absorc1)], '-k')
            ax1.set_xlabel('Absorbância de referência')
            ax1.set_ylabel('Absorbância calculada L-B')
            ax1.set_title('Ajuste absorbância SEM termo independente')

            # Plot for Lambert-Beer com termo independente
            ax2.plot(absor, absorc2, 'o', markersize=5, markeredgewidth=1, markeredgecolor='black')
            ax2.plot([np.min(absor.values), np.max(absor.values)], [np.min(absorc2), np.max(absorc2)], '-k')
            ax2.set_xlabel('Absorbância de referência')
            ax2.set_ylabel('Absorbância calculada L-B')
            ax2.set_title('Ajuste absorbância COM termo independente')

            # ajustando layout
            fig.tight_layout()
            plt.show(block=False)
            plt.pause(0.001)  # pause para deixar renderizar

            # input pra dar um pouse
            input("Aperte enter para continuar") 
        return (Ks,Kc)
    
    def PCA_manual(self,plots=False):
        """
        Implementa a análise de componentes principais (PCA) manualmente.

        Parâmetros:
            plots (bool): Se True, gera gráficos dos resultados.

        Retorna:
            tuple: Autovetores, autovalores, variância relativa, e variância acumulada.
        """
        absor = self.X
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

            fig = plt.figure(figsize=(12, 10))
            gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.1],width_ratios=[1, 1])

            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax5 = fig.add_subplot(gs[1:, :])

            # Plotando as 3 primeiras PCs
            ax1.plot(lambda_values, eigvec[:, 0], label='PC1')
            ax1.plot(lambda_values, eigvec[:, 1], label='PC2')
            ax1.plot(lambda_values, eigvec[:, 2], label='PC3')
            ax1.axhline(0, color='k')
            ax1.set_xlabel('Comprimento')
            ax1.set_ylabel('PC')
            ax1.set_title('Componentes principais')
            ax1.legend()
            

            # plot para as duas primeiras componentes principais
            pc1 = np.dot(anorm, eigvec[:, 0])
            pc2 = np.dot(anorm, eigvec[:, 1])
            ax2.scatter(pc1, pc2, marker='x')
            ax2.set_xlabel('PC1')
            ax2.set_ylabel('PC2')
            ax2.set_title('Componentes principais')

            # Plotando a variância explicada e a variância acumulada
            ax6 = ax5.twinx()

            color = 'tab:blue'
            ax5.set_xlabel('Componente Principal')
            ax5.set_ylabel('Variância Explicada', color=color)
            ax5.plot(range(1, len(var_rel[:maxind]) + 1), var_rel[:maxind], color=color, label='Variância Explicada')
            ax5.tick_params(axis='y')

            color = 'tab:red'
            ax6.set_ylabel('Variância Acumulada', color=color)
            ax6.plot(range(1, len(var_ac[:maxind]) + 1), var_ac[:maxind], color=color, label='Variância Acumulada')
            ax6.tick_params(axis='y')
            ax5.set_title('Variância Explicada e Acumulada por Componente Principal')

            fig.tight_layout(pad=2.0)
            fig.suptitle('Análise de PCA manual', fontsize=16)
            plt.show(block=False)
            plt.pause(0.001)  # pause para deixar renderizar

            # input pra dar um pouse
            input("Aperte enter para continuar") 


            '''
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
            '''

        return eigvec, eigval, var_rel[:maxind], var_ac[:maxind]
    
    def PCA(self, plots=False):
        """
        Implementa a análise de componentes principais (PCA) usando scikit-learn.

        Parâmetros:
            plots (bool): Se True, gera gráficos dos resultados.

        Retorna:
            tuple: Autovetores, autovalores, variância relativa, e variância acumulada.
        """
        absor = self.X
        lambda_values = self.comprimentos

        # Normalizando
        anorm = (absor - np.mean(absor, axis=0)) / np.std(absor, axis=0)

        # Realizando PCA
        pca = sklearnPCA(svd_solver="covariance_eigh")
        pca.fit(anorm)

        eigvec = pca.components_.T
        eigval = pca.explained_variance_
        var_rel = pca.explained_variance_ratio_
        var_ac = np.cumsum(var_rel)

        # Selecionando as variâncias até um certo limite para print
        limite = 0.9999
        maxind = np.argmax(var_ac >= limite) + 1
        
        if plots:
            fig = plt.figure(figsize=(12, 10))
            gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.1],width_ratios=[1, 1])

            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax5 = fig.add_subplot(gs[1:, :])

            # Plotando as 3 primeiras PCs
            ax1.plot(lambda_values, eigvec[:, 0], label='PC1')
            ax1.plot(lambda_values, eigvec[:, 1], label='PC2')
            ax1.plot(lambda_values, eigvec[:, 2], label='PC3')
            ax1.axhline(0, color='k')
            ax1.set_xlabel('Comprimento')
            ax1.set_ylabel('PC')
            ax1.set_title('Componentes principais')
            ax1.legend()

            # Plot para as duas primeiras componentes principais
            pc1 = pca.transform(anorm)[:, 0]
            pc2 = pca.transform(anorm)[:, 1]
            ax2.scatter(pc1, pc2, marker='x')
            ax2.set_xlabel('PC1')
            ax2.set_ylabel('PC2')
            ax2.set_title('Componentes principais')

            # Plotando a variância explicada e a variância acumulada
            ax6 = ax5.twinx()

            color = 'tab:blue'
            ax5.set_xlabel('Componente Principal')
            ax5.set_ylabel('Variância Explicada', color=color)
            ax5.plot(range(1, len(var_rel[:maxind]) + 1), var_rel[:maxind], color=color, label='Variância Explicada')
            ax5.tick_params(axis='y')

            color = 'tab:red'
            ax6.set_ylabel('Variância Acumulada', color=color)
            ax6.plot(range(1, len(var_ac[:maxind]) + 1), var_ac[:maxind], color=color, label='Variância Acumulada')
            ax6.tick_params(axis='y')
            ax5.set_title('Variância Explicada e Acumulada por Componente Principal')

            fig.tight_layout(pad=2.0,rect=[0, 0.03, 1, 0.95])
            fig.suptitle('Análise de PCA', fontsize=16)
            plt.show(block=False)
            plt.pause(0.001)  # pause para deixar renderizar

            # input pra dar um pouse
            input("Aperte enter para continuar") 



        '''
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
        '''
        return eigvec, eigval, var_rel[:maxind], var_ac[:maxind]

    def multicalib(self):
        """
        Realiza a calibração multivariada nos dados.
        """
        # Xtot = self.X
        Xtot = self.X.to_numpy()
        ytot = self.Y.to_numpy()
        print(Xtot[:,0])
        print(ytot.shape)
        print(ytot[:,0])
        
        cname = self.analitos
        print(cname)

        return Multical.multical(Xtot,ytot,cname)
    
    def inferlib(self,model_matrix,error_matrix):
        """
        Realiza a inferência usando as bibliotecas e modelos calibrados.

        Parâmetros:
            model_matrix: Matriz do modelo calibrado.
            error_matrix: Matriz de erros associada ao modelo.
        """
       Xtot = self.X.to_numpy()
       ytot = self.Y.to_numpy()
       
       Xtest = Xtot[0:18,:]
       ytest = ytot[0:18,:]
       
       thplc = np.linspace(0,17,18).astype(int)
       cname = self.analitos


       return Infer.infer(Xtot,Xtest,ytot,ytest,thplc,model_matrix,error_matrix,cname)

teste=Dados_exp()
pretratamentos_exemplo=[
    ('sav_gol',{'janela':11,'polyorder':3,'derivada':1})]
teste_pret=teste.pretreat(pretratamentos=pretratamentos_exemplo)
teste_pret.multicalib()
