function [Yp,Ytp,Par_norm] = pcr_model(X,Y,k,Xt)
    // Ajusta modelo utilizando X, Y e k. 
    // Calcula a resposta dado Xt 
    // PCR
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

        [eigvec,eigval] = spec(Xnorm'*Xnorm);
        Xeig = Xnorm*eigvec(:,1:k);
        
        Beta =  Xeig\Ynorm;

        Ynormp = Xeig*Beta;
        m=size(Ynormp,1)

        Yp = Ynormp.*repmat(Ysig,m,1) + repmat(Ymed,m,1);
        if ~isempty(Xt) then
            m=size(Xt,1);
            Xtnorm = (Xt-repmat(Xmed,m,1))./repmat(Xsig,m,1);
            Xteig = Xtnorm*eigvec(:,1:k); // scores
            Ytnormp = Xteig*Beta;
            Ytp = Ytnormp.*repmat(Ysig,m,1) + repmat(Ymed,m,1);
        end
        Par_norm = mlist(['','Xmed','Xsig','Ymed','Ysig','Beta','T'],[Xmed],[Xsig],[Ymed],[Ysig],[Beta],[Xeig])

    case 1
        n = size(X,1)
        [Xnorm,Xmed,Xsig]=zscore([X;Xt]);
        Xtnorm = Xnorm(n+1:$,:);
        Xnorm(n+1:$,:) = [];


        [Ynorm,Ymed,Ysig]=zscore(Y);

        [eigvec,eigval] = spec(Xnorm'*Xnorm);
        Xeig = Xnorm*eigvec(:,1:k);
        
        Beta =  Xeig\Ynorm;
        Ynormp = Xeig*Beta;
        m=size(Ynormp,1)

        Yp = Ynormp.*repmat(Ysig,m,1) + repmat(Ymed,m,1);
        if ~isempty(Xt) then
            m=size(Xt,1);
            //Xtnorm = (Xt-repmat(Xmed,m,1))./repmat(Xsig,m,1);
            Xteig = Xtnorm*eigvec(:,1:k);
            Ytnormp = Xteig*Beta;
            Ytp = Ytnormp.*repmat(Ysig,m,1) + repmat(Ymed,m,1);
        end
        Par_norm = mlist(['','Xmed','Xsig','Ymed','Ysig','Beta','T'],[Xmed],[Xsig],[Ymed],[Ysig],[Beta],[Xeig])

    end
endfunction
