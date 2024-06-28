import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
import copy
import random

def pls_model(Xtreino,ytreino,nregs,Xteste):
    # PLSRegression() normaliza X e Y automaticamente
    pls = PLSRegression(n_components=nregs,scale=False)
    pls.fit(Xtreino,ytreino)
    y_predito = pls.predict(Xteste).flatten()
    return y_predito

def infer(X,Xt,y,yt,thplc,model_matrix,error_matrix,cname):
    
    texp = np.linspace(0,Xt.shape[0]-1,Xt.shape[0]).astype(int)
        
    ## Seleciona conjuntos pra aleatorizar
    
    nd = y.shape[0]
    indval = np.linspace(0, nd-1,nd).astype(int)
    random.seed(4)
    random.shuffle(indval)
    X = copy.deepcopy(X[indval,:])
    y = copy.deepcopy(y[indval,:])
    
    nesp = y.shape[1]
    y_infer = np.zeros([Xt.shape[0],yt.shape[1]])
    
    ## Escolhe n de regressores e infere com PLS
    for esp in range(nesp):
        nregs = model_matrix[0,esp]
        y_infer[:,esp] = pls_model(X[:,:],y[:,esp], nregs, Xt[:,:])
    
    
    plt.figure(figsize=(12, 6))
    plt.title("x - HPLC, o - PLS")
    plt.plot(thplc,yt,'x')
    plt.gca().set_prop_cycle(None)
    plt.plot(texp,y_infer,'o',fillstyle='none')
    plt.xlabel("t (min)");
    plt.ylabel("C (mol/L)");
    plt.legend(cname)
    plt.show()
    
    return y_infer