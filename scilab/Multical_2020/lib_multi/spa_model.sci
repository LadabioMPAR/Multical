function [Yp,Ytp,Par_norm] = spa_model(X,Y,k,cini,Xt)
    // Ajusta modelo utilizando X, Y e k. 
    // Calcula da resposta dado Xt 
    // SPA
    // X = matriz de dados #amostras x #variáveis
    // Y = matriz de respostas (ou vetor) #amostras x #respostas
    // Xt = matriz de dados para predição da resposta
    // Yp = Y predito X
    // Ytp = Y predito a partir de Xt
    // k = número de regressores
    // cini = número da 1a coluna da matriz X a ser escolhida (não o no. de comprimento de onda)

teste = 1; // 1 - normaliza a matriz X levando em conta a matriz Xt; 0 - normalização da matriz X é independente de Xt
// observe que se sempre que for realizar uma inferência, será ajustado o modelo, então que a princípio, não há problema.
    switch teste
    case 0 
        [Xnorm,Xmed,Xsig]=zscore(X);
        [Ynorm,Ymed,Ysig]=zscore(Y);

        ind = spa_clean(X,cini,k)
        Beta =  Xnorm(:,ind) \Ynorm;

        Ynormp = Xnorm(:,ind)*Beta;
        m=size(Ynormp,1)

        Yp = Ynormp.*repmat(Ysig,m,1) + repmat(Ymed,m,1);
        if ~isempty(Xt) then
            m=size(Xt,1);
            Xtnorm = (Xt-repmat(Xmed,m,1))./repmat(Xsig,m,1);
            Ytnormp = Xtnorm(:,ind)*Beta;
            Ytp = Ytnormp.*repmat(Ysig,m,1) + repmat(Ymed,m,1);
        end
        Par_norm = mlist(['','Xmed','Xsig','Ymed','Ysig','Beta','ind'],[Xmed],[Xsig],[Ymed],[Ysig],[Beta],[ind])

    case 1
        n = size(X,1)
        [Xnorm,Xmed,Xsig]=zscore([X;Xt]);
        Xtnorm = Xnorm(n+1:$,:);
        Xnorm(n+1:$,:) = [];


        [Ynorm,Ymed,Ysig]=zscore(Y);

        ind = spa_clean(X,cini,k)
        Beta =  Xnorm(:,ind) \Ynorm;

        Ynormp = Xnorm(:,ind)*Beta;
        m=size(Ynormp,1)

        Yp = Ynormp.*repmat(Ysig,m,1) + repmat(Ymed,m,1);
        if ~isempty(Xt) then
            m=size(Xt,1);
            //Xtnorm = (Xt-repmat(Xmed,m,1))./repmat(Xsig,m,1);
            Ytnormp = Xtnorm(:,ind)*Beta;
            Ytp = Ytnormp.*repmat(Ysig,m,1) + repmat(Ymed,m,1);
        end
        Par_norm = mlist(['','Xmed','Xsig','Ymed','Ysig','Beta','ind'],[Xmed],[Xsig],[Ymed],[Ysig],[Beta],[ind])

    end
endfunction
