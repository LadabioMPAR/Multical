import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
import os

def func_multi_ILS_python(Selecao, kmax, cname, x0, absor0, frac_test=0.0, 
                          dadosteste=None, OptimModel=None, pretreat=None, 
                          outlier_flag=False, gravarXval=True):
    """
    Tradução da função func_multi_ILS_new do Scilab para Python.
    
    Selecao: 1 = PLS, 3 = PCR (SPA requer implementação customizada)
    """
    
    # -----------------------------------------------------------
    # 1. PREPARAÇÃO DOS DADOS
    # -----------------------------------------------------------
    
    # Separar lambda (primeira linha no Scilab)
    # Assumindo que absor0 é numpy array. Se for arquivo, usar np.loadtxt antes.
    lambdas = absor0[0, :]
    X_absor = absor0[1:, :] # Matriz espectral (X)
    Y_conc = x0             # Matriz de concentração (Y)
    
    nd, nl = X_absor.shape
    ndx, nc = Y_conc.shape
    
    # Verificação de erro de dimensão
    if nd != ndx:
        print(f"Erro: Núm. dados concentração ({ndx}) != Núm. dados absorbância ({nd})")
        return -1
    
    # -----------------------------------------------------------
    # 2. PRÉ-PROCESSAMENTO (Placeholder)
    # -----------------------------------------------------------
    # No Scilab você chama func_pretreatment. Aqui você usaria bibliotecas
    # como scipy.signal.savgol_filter ou sklearn.preprocessing.StandardScaler.
    
    # Exemplo simples: Centralizar na média (Mean Center é padrão no PLS do sklearn)
    # Se precisar de derivadas ou SNV, aplique em X_absor aqui.
    pass 

    # -----------------------------------------------------------
    # 3. DIVISÃO DE DADOS E NORMALIZAÇÃO
    # -----------------------------------------------------------
    
    # Normalizando Y de 0 a 1 (como no original)
    ymax = Y_conc.max(axis=0)
    Y_norm = Y_conc / ymax
    
    # Divisão Teste/Treino
    X_test = None
    Y_test = None
    
    if dadosteste is not None:
        # Assumindo dadosteste como tupla (Y_test, X_test_com_lambda)
        yt = dadosteste[0]
        xt_raw = dadosteste[1]
        X_test = xt_raw[1:, :] # remove lambda
        # Aplicar mesmo pré-tratamento em X_test aqui
        Y_test = yt / ymax # Normaliza teste com maximo do treino
        
    elif frac_test > 0:
        # Separação aleatória
        X_absor, X_test, Y_norm, Y_test = train_test_split(
            X_absor, Y_norm, test_size=frac_test, random_state=42
        )
        nd = X_absor.shape[0] # Atualiza número de dados de treino

    # -----------------------------------------------------------
    # 4. CONFIGURAÇÃO DA VALIDAÇÃO (Cross-Validation)
    # -----------------------------------------------------------
    
    # Inicializa matrizes de erro
    RMSECV = np.zeros((kmax, nc))
    RMSEcal = np.zeros((kmax, nc))
    RMSEtest = np.zeros((kmax, nc))
    
    # Configura K-Fold
    n_splits = nd # Default LOOCV (Leave-one-out)
    
    if OptimModel and OptimModel['type'] == 'kfold':
        kpart = OptimModel['value']
        if kpart != -1:
            n_splits = int(np.ceil(nd / kpart)) if kpart < nd else nd
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    print(f"Iniciando calibração com {n_splits} folds...")

    # -----------------------------------------------------------
    # 5. LOOP DE VARIÁVEIS LATENTES (k)
    # -----------------------------------------------------------
    
    lado = int(np.ceil(np.sqrt(nc)))
    fig, axs = plt.subplots(lado, lado, figsize=(10, 8))
    if nc == 1: axs = [axs]
    else: axs = axs.flatten()

    for k in range(1, kmax + 1):
        print(f"Processando k = {k}")
        
        # Matrizes para guardar previsões da Validação Cruzada
        Y_cv_pred = np.zeros_like(Y_norm)
        
        # --- LOOP DE K-FOLD ---
        for train_index, val_index in kf.split(X_absor):
            X_train_cv, X_val_cv = X_absor[train_index], X_absor[val_index]
            Y_train_cv, Y_val_cv = Y_norm[train_index], Y_norm[val_index]
            
            # Escolha do Modelo
            if Selecao == 1: # PLS
                model = PLSRegression(n_components=k, scale=False) 
                # scale=False pq já normalizamos ou tratamos antes geralmente
                model.fit(X_train_cv, Y_train_cv)
                pred = model.predict(X_val_cv)
                
            elif Selecao == 3: # PCR
                pca = PCA(n_components=k)
                X_pca_train = pca.fit_transform(X_train_cv)
                X_pca_val = pca.transform(X_val_cv)
                model = LinearRegression()
                model.fit(X_pca_train, Y_train_cv)
                pred = model.predict(X_pca_val)
                
            elif Selecao == 2: # SPA
                print("SPA requer implementação customizada (algoritmo não nativo do sklearn).")
                return
            
            # Armazena previsão CV (pode ser multivariado)
            Y_cv_pred[val_index] = pred

        # --- FIM DO K-FOLD PARA ESTE K ---

        # 1. Calcular RMSECV (Erro de Validação Cruzada)
        # Desnormalizando para calcular o erro real
        err_cv = (Y_cv_pred - Y_norm) * ymax 
        rmsecv_k = np.sqrt(np.mean(err_cv**2, axis=0))
        RMSECV[k-1, :] = rmsecv_k
        
        # 2. Calcular RMSEC (Erro de Ajuste/Calibração com TODOS os dados)
        if Selecao == 1:
            full_model = PLSRegression(n_components=k, scale=False)
            full_model.fit(X_absor, Y_norm)
            Y_cal_pred = full_model.predict(X_absor)
        elif Selecao == 3:
            full_pca = PCA(n_components=k)
            X_pca_all = full_pca.fit_transform(X_absor)
            full_model = LinearRegression()
            full_model.fit(X_pca_all, Y_norm)
            Y_cal_pred = full_model.predict(X_pca_all)
            
        err_cal = (Y_cal_pred - Y_norm) * ymax
        rmsecal_k = np.sqrt(np.mean(err_cal**2, axis=0))
        RMSEcal[k-1, :] = rmsecal_k
        
        # 3. Calcular RMSEP (Teste Externo), se houver
        if X_test is not None:
            if Selecao == 1:
                Y_test_pred = full_model.predict(X_test)
            elif Selecao == 3:
                X_test_pca = full_pca.transform(X_test)
                Y_test_pred = full_model.predict(X_test_pca)
            
            err_test = (Y_test_pred - Y_test) * ymax
            rmsetest_k = np.sqrt(np.mean(err_test**2, axis=0))
            RMSEtest[k-1, :] = rmsetest_k
        
        # Salvar previsões CV do último k para análise de outliers ou arquivos
        if gravarXval:
            np.savetxt(f"Xvalconc_{k}.txt", Y_cv_pred * ymax, fmt='%.6f')

    # -----------------------------------------------------------
    # 6. PLOTAGEM FINAL E RESULTADOS
    # -----------------------------------------------------------
    
    # Plotar RMSE vs K
    plt.figure()
    for i in range(nc):
        plt.plot(range(1, kmax+1), RMSECV[:, i], label=f'{cname[i]} CV')
        plt.plot(range(1, kmax+1), RMSEcal[:, i], '--', label=f'{cname[i]} Cal')
        if X_test is not None:
            plt.plot(range(1, kmax+1), RMSEtest[:, i], ':', label=f'{cname[i]} Test')
            
    plt.xlabel('Número de Variáveis Latentes (k)')
    plt.ylabel('RMSE (unidade real)')
    plt.legend()
    plt.title('Erro vs Complexidade do Modelo')
    plt.show()

    # Exibição de texto simples
    print("\n--- Resultados Finais (RMSECV) ---")
    res_df = pd.DataFrame(RMSECV, columns=cname, index=range(1, kmax+1))
    print(res_df)
    
    # -----------------------------------------------------------
    # 7. ANÁLISE DE OUTLIERS (Simplificada)
    # -----------------------------------------------------------
    if outlier_flag:
        # Exemplo simples baseado no resíduo do último k
        errors = Y_cv_pred - Y_norm
        std_dev = np.std(errors, axis=0)
        z_scores = np.abs(errors / std_dev)
        outliers = np.where(z_scores > 3) # > 3 desvios padrão
        if outliers[0].size > 0:
            print("\nPossíveis outliers detectados (índices):")
            print(np.unique(outliers[0]))

    return RMSECV, RMSEcal, RMSEtest