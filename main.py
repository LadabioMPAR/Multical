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
    Class to handle experimental data.

    Attributes:
    arquivo_json (str): Path to the JSON file containing workspace data.
    X (list): List of DataFrames containing absorbance data.
    y (list): List of DataFrames containing reference data.
    comprimentos (list): List of wavelengths.
    analitos (list): List of analytes.
    """

    def __init__(self, arquivo_json='workspace.json', X=[], y=[], comprimentos=None, analitos=None):
        """
        Initializes an instance of Dados_exp.

        :param arquivo_json: Path to the JSON file containing workspace data, defaults to 'workspace.json'.
        :type arquivo_json: str, optional
        :param X: List of absorbance DataFrames, defaults to an empty list.
        :type X: list, optional
        :param y: List of reference DataFrames, defaults to an empty list.
        :type y: list, optional
        :param comprimentos: List of wavelengths, defaults to None.
        :type comprimentos: list, optional
        :param analitos: List of analytes, defaults to None.
        :type analitos: list, optional
        """

        self.comprimentos=comprimentos
        self.analitos=analitos
        self.X=X
        self.Y=y
        if (X.empty if isinstance(X, pd.DataFrame) else not X) and (y.empty if isinstance(y, pd.DataFrame) else not y):
            if os.path.getsize(arquivo_json) == 0:
                subprocess.run(["python", "Import.py"])
            self.X, self.Y, self.comprimentos, self.analitos = self.lendo_workspace(arquivo_json)
            #código fita-crepe pra sempre dar uma matrizona:
            self.X=self.stack_x()
            self.Y= self.stack_y()

    def lendo_workspace(self, arquivo_json):
        """
        Reads the workspace JSON file and returns the data.

        :param arquivo_json: Path to the JSON file.
        :type arquivo_json: str
        :return: Tuple containing absorbance data (X), reference data (Y), wavelengths, and analytes.
        :rtype: tuple
        :raises FileNotFoundError: If the JSON file is not found.
        :raises KeyError: If 'comprimentos' or 'referencias' keys are not found in the JSON file.
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
        Identifies and reads files based on their extension to create DataFrames.

        :param caminho_arquivo: Path to the file.
        :type caminho_arquivo: str
        :param for_x: If True, returns absorbance data; if False, returns reference data, defaults to True.
        :type for_x: bool, optional
        :return: Data read from the file.
        :rtype: pandas.DataFrame
        :raises ValueError: If the file extension is not supported.
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
        Adds new data to the Dados_exp object.

        :param X: Absorbance data.
        :type X: pandas.DataFrame
        :param Y: Reference data, defaults to None.
        :type Y: pandas.DataFrame, optional
        :param comprimento: Wavelength, defaults to None.
        :type comprimento: int, optional
        :param analito: Analyte name, defaults to None.
        :type analito: str, optional
        :raises ValueError: If X or Y are not DataFrames.
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
        Stacks the absorbance data (X) into a single DataFrame.

        :return: DataFrame containing all absorbance values.
        :rtype: pandas.DataFrame
        :raises ValueError: If the list of absorbances is empty.
        """
        if not self.X:
            raise ValueError("A lista de absorbâncias está vazia.")
        return pd.concat(self.X.copy(), axis=0, ignore_index=True)

    def stack_y(self):
        """
        Stacks the reference data (Y) into a single DataFrame.

        :return: DataFrame containing all reference values.
        :rtype: pandas.DataFrame
        :raises ValueError: If the list of references is empty.
        """
        if not self.Y:
            raise ValueError("A lista de referências está vazia.")
        return pd.concat(self.Y.copy(), axis=0, ignore_index=True)

    def pretreat(self, pretratamentos, salvar=False):
        """
        Applies a list of preprocessing treatments to the data.

        :param pretratamentos: List of tuples containing the name of the preprocessing function and its parameters.
        :type pretratamentos: list
        :param salvar: If True, saves the treated data and the new workspace, defaults to False.
        :type salvar: bool, optional
        :return: A Dados_exp object with the treated DataFrame.
        :rtype: Dados_exp
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
        
        return Dados_exp(X=X_tratado, y=self.Y, comprimentos=[int(coluna) for coluna in X_tratado.columns.values.tolist()] if not X_tratado.empty else [], analitos=self.analitos)

    def salvar(self, nomex="X_",nomey="y_",workspace="workspace"):
        """
        Saves X and Y values to text files and generates a new workspace.

        :param nomex: Prefix for absorbance files (X values).
        :type nomex: str, optional
        :param nomey: Prefix for reference files (Y values).
        :type nomey: str, optional
        :param workspace: Name of the workspace file, defaults to "workspace".
        :type workspace: str, optional
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
            file_path = os.path.join(pasta, f'{nomex}{i+1}.txt')
            df.to_csv(file_path, sep='\t', index=False)
            comprimentos_paths.append(file_path)

        # Salvando DataFrames de y
        for i, df in enumerate(self.Y):
            file_path = os.path.join(pasta, f'{nomey}{i+1}.txt')
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
        Plots the spectra contained in X.

        :param nome: Name of the figure, defaults to "fig".
        :type nome: str, optional
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
        plt.show()  # pause para deixar renderizar

        # input pra dar um pouse
        input("Aperte enter para continuar") 
    
    def LB(self,plots=False):
        """
        Performs classical least squares analysis for the Lambert-Beer law.

        :param plots: If True, generates graphs of the results, defaults to False.
        :type plots: bool, optional
        :return: Tuple containing the Ks (without intercept) and Kc (with intercept) matrices.
        :rtype: tuple
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
        Manually implements Principal Component Analysis (PCA).

        :param plots: If True, generates standard PCA plots, defaults to False.
        :type plots: bool, optional
        :return: Tuple containing eigenvectors, eigenvalues, relative variance, and cumulative variance.
        :rtype: tuple
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




        return eigvec, eigval, var_rel[:maxind], var_ac[:maxind]
    
    def PCA(self, plots=False):
        """
        Implements Principal Component Analysis (PCA) using scikit-learn.

        :param plots: If True, generates standard PCA plots, defaults to False.
        :type plots: bool, optional
        :return: Tuple containing eigenvectors, eigenvalues, relative variance, and cumulative variance.
        :rtype: tuple
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




        return eigvec, eigval, var_rel[:maxind], var_ac[:maxind]

    def multicalib(self):
        """
        Performs multivariate calibration on the data.
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
        Performs inference using the libraries and calibrated models.

        :param model_matrix: Calibrated model matrix.
        :type model_matrix: numpy.ndarray
        :param error_matrix: Error matrix associated with the model.
        :type error_matrix: numpy.ndarray
        """
        Xtot = self.X.to_numpy()
        ytot = self.Y.to_numpy()
       
        Xtest = Xtot[0:18,:]
        ytest = ytot[0:18,:]
       
        thplc = np.linspace(0,17,18).astype(int)
        cname = self.analitos


        return Infer.infer(Xtot,Xtest,ytot,ytest,thplc,model_matrix,error_matrix,cname)


