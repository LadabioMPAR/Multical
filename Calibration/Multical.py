import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
import copy
import random
from scipy.stats import t
from scipy.stats import f
import csv
import pandas as pd

def multical(X,y,cname,regmax=15):
    '''
    Xtot e ytot são arrays de mesmo número de linhas, contendo as matrizes de absorbância (X) e concentrações de HPLC
    das espécies (y) correspondentes. Ambos fazem parte do conjunto de dados de treinamento. 
    
    A multicalibração é feita via PLS. Aumentando o número de regressores do PLS, é feita a validação cruzada por k-fold.
    Os dados são juntados em duas matrizes Xs e ys, embaralhados e faz-se a previsão pra cada fold, armazenada em ycv.
    
    O teste F é realizado comparando a variação de RMSECV ao incrementar regressores no PLS, para ver se a melhoria
    no modelo complexo é estatisticamente relevante.
    '''

    nesp = y.shape[1] ## n  de especies medidas

    #%% PLS
    nregmax = regmax ## n de regressores maximo a ser testado
    y_cv = np.zeros(y.shape)
    rmsecv   = np.zeros([nregmax,nesp])

    nd = y.shape[0]
    
    Xs = np.zeros(X.shape)
    ys = np.zeros(y.shape)
  
    indval = np.linspace(0, nd-1,nd).astype(int)
    random.seed(4)
    random.shuffle(indval)
    
    Xs = copy.deepcopy(X[indval,:])
    ys = copy.deepcopy(y[indval,:])
    
    rmsecv   = np.zeros([nregmax,nesp])
    for nregs in range(1,nregmax+1):
        for esp in range(nesp):
            ## encontra os y de val. cruzada, calcula os erros
            pls = PLSRegression(n_components=nregs,scale=True)
            y_cv[:,esp] = cross_val_predict(pls, Xs[:,:],ys[:,esp],cv=4).flatten()
            rmsecv[nregs-1,esp] = np.sqrt((ys[:,esp]- y_cv[:,esp]).dot(ys[:,esp]- y_cv[:,esp])/nd)
    

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    regs = np.linspace(1,15,15).astype(int)
    for esp in range(nesp):
        ## definindo coordenada dos plots
        i, j = divmod(esp, 3)
        ax = axes[i, j]
        ax.plot(regs,rmsecv[:,esp])
        # if indiceteste.size: ##se existir yt no cluster atual:
            # ax.plot(range(1,nregmax+1),rmsetest[:,esp])
        ax.legend(['RMSECV','RMSEtest'])
        ax.set_xlabel('n regressores')
        ax.set_ylabel('RMSE')
        ax.set_title(cname[esp])
        
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()
    
    #%% Escolhendo modelo - Teste F - Não é exatamente igual ao do Nelles
    model_matrix = np.ones([1,nesp])*1
    model_matrix=model_matrix.astype(int)
    error_matrix = np.zeros(model_matrix.shape)
    
    # Teste F
    alpha= 0.95
    N = y.shape[0]

    for esp in range(nesp):
        nregoptim=0
        for nreg in range(nregs-1):
            Isimp = rmsecv[nregoptim,esp]**2
            Icomp = rmsecv[nreg+1,esp]**2

            T = Isimp/Icomp
            F = f.ppf(alpha,N,N)
            
            if T> F: # Hipotese nula rejeitada, modelo simples ruim
                nregoptim = nreg+1
            
        model_matrix[0,esp] = nregoptim+1
        error_matrix[0,esp] = rmsecv[nregoptim,esp]
    
    #%% Gráfico VC para modelo ótimo
    optimregs=model_matrix[0]
    
    # Create a figure and a 3x3 grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    aux=0
    for esp in range(nesp):
        nregs = optimregs[aux]
        ## encontra os y de val. cruzada, calcula os erros
        pls = PLSRegression(n_components=nregs,scale=True)
        y_cv[:,esp] = cross_val_predict(pls, Xs[:,:],ys[:,esp],cv=4).flatten()
        
        ## definindo coordenada dos plots
        i, j = divmod(esp, 3)
        ax = axes[i, j]
        ax.scatter(ys[:,esp], y_cv[:,esp])
  
        ax.legend(['valid','test'])
        ax.plot([-3, 3], [-3, 3], color='red', linestyle='--')  # bissetriz
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('y ref')
        ax.set_ylabel('y pred')
        ax.set_title(cname[esp])
        aux+=1
            
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()
    
    y_cv = copy.deepcopy(y_cv[:])
    
    #%% Matriz covar
    # desv = (ys-y_cv)*np.repeat([np.max(y,axis=0)],repeats=y.shape[0],axis=0) ## pois média do erro é assumida como 0, desvio é o próprio erro
    # covar_erro = np.dot(desv.T, desv)/ys.shape[0] # Matriz R
    # np.savetxt("matriz_R.csv", covar_erro, delimiter=" ",fmt='%.5e')
    # erro_pad = np.sqrt(np.diagonal(covar_erro))
    # erro_pad= erro_pad.reshape(erro_pad.shape[0],1)
    # # correl = np.dot((1/erro_pad),(1/erro_pad).T) * covar_erro
    
    # regs = np.linspace(1, 15,15)
    
    # with open('RMSECV.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(error_matrix)
    #     writer.writerow(model_matrix)
    
    # all = [X,model_matrix,error_matrix,covar_erro,rmsecv,y_cv,ys,regs]
    # print(all)
    return model_matrix,error_matrix,rmsecv,y_cv,ys

