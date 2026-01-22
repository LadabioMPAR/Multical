function [Yp,Ytp,Par_norm] = pls_model(X,Y,k,Xt)
    // Ajusta modelo utilizando X, Y e k. 
    // Calcula da resposta dado Xt 
    // PLS
    // X = matriz de dados #amostras x #variáveis
    // Y = matriz de respostas (ou vetor) #amostras x #respostas
    // Xt = matriz de dados para predição da resposta
    // Yp = Y predito X
    // Ytp = Y predito a partir de Xt
    // k = número de regressores

    teste = 1; // 1 - normaliza a matriz X levando em conta a matriz Xt; 0 - normalização da matriz X é independente de Xt
    // observe que se sempre que for realizar uma inferência, será ajustada o modelo, então que a princípio, não há problema.

    switch teste
    case 0 
        [Xnorm,Xmed,Xsig]=zscore(X);
        [Ynorm,Ymed,Ysig]=zscore(Y);

        [Beta,T]=pls(Xnorm,Ynorm,k);
        Ynormp = Xnorm*Beta;
        m=size(Ynormp,1)
        Yp = Ynormp.*repmat(Ysig,m,1) + repmat(Ymed,m,1);
        if ~isempty(Xt) then
            m=size(Xt,1);
            Xtnorm = (Xt-repmat(Xmed,m,1))./repmat(Xsig,[m,1]);
            Ytnormp = Xtnorm*Beta;
            //        Ytp = Ytnormp.*ndgrid(Ysig,[1:m])' + ndgrid(Ymed,[1:m])';
            Ytp = Ytnormp.*repmat(Ysig,[m,1]) + repmat(Ymed,[m,1]);
        end
        Par_norm = mlist(['','Xmed','Xsig','Ymed','Ysig','Beta','T'],[Xmed],[Xsig],[Ymed],[Ysig],[Beta],T)        

    case 1
        n = size(X,1)
        [Xnorm,Xmed,Xsig]=zscore([X;Xt]);
        Xtnorm = Xnorm(n+1:$,:);
        Xnorm(n+1:$,:) = [];
        [Ynorm,Ymed,Ysig]=zscore(Y);

        [Beta,T]=pls(Xnorm,Ynorm,k);
        Ynormp = Xnorm*Beta;
        m=size(Ynormp,1)
        Yp = Ynormp.*repmat(Ysig,m,1) + repmat(Ymed,m,1);

        if ~isempty(Xt) then
            m=size(Xt,1);
            Ytnormp = Xtnorm*Beta;
            Ytp = Ytnormp.*repmat(Ysig,[m,1]) + repmat(Ymed,[m,1]);
        end
        Par_norm = mlist(['','Xmed','Xsig','Ymed','Ysig','Beta','T'],[Xmed],[Xsig],[Ymed],[Ysig],[Beta],T)        


    end

endfunction

