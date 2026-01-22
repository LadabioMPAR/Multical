function ind = spa_clean(X,cini,nk)
    // Algoritmo retirado do artigo Araujo, 2001, 65-73 
    //'The successive projections algorithm for variable selection in spectroscopic multicomponent analysis'
    // Arg de saída
    // ind é o vetor com os índices escolhidos (colunas de X) 
    // Args de entrada
    // X = matriz de dados (absorbâncias) #amostras x #variáveis 
    // nk   = numero de comprimentos de onda a serem utilizados
    // cini = primeira coluna (refernete a um comprimento de onda) a ser utilizada, ind(1)


    // Xn deve ter colunas com norma 1
    [m,n]=size(X)
    Xn = zeros(m,n);
    indices = [1:n]; // colunas originais
    for i=1:n
        Xn(:,i) = X(:,i)./norm(X(:,i))
    end
    ind = zeros(nk,1);
    ind(1) = cini;

    // extraindo a coluna k(1)
    xant = Xn(:,ind(1));
    Xn(:,ind(1)) = [];
    indices(ind(1)) = [];

    for ik=2:nk
        [mm,nn] = size(Xn)
        normPxj = [];
        for j=1:nn
            xj=Xn(:,j);
            Pxj= xj - (xj'*xant)*xant*inv(xant'*xant);
            normPxj(j)=norm(Pxj);
            Xn(:,j)=Pxj;
        end
        [maxnorm,indmax] = max(normPxj);
        //retirando coluna extraída
        ind(ik) = indices(indmax);
        xant = Xn(:,indmax);
        Xn(:,indmax) = [];
        indices(indmax) = [];
    end
endfunction
