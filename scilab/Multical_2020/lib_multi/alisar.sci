
function [Aalis,lambalis]=alisar(alis,absor,lambda,raio)
nl = length(lambda);
switch  alis
    case   1 // no. de dados permanece o mesmo
        A = absor;
        Aalis = zeros((A));
        for i=1:raio
            Aalis(:,i) = mean(A(:,1:i+raio),2); //  média entre i-raio e i+raio
        end
        for i=1+raio:nl-raio
            Aalis(:,i) = mean(A(:,i-raio:i+raio),2);
        end
        for i=nl-raio+1:nl
            Aalis(:,i) = mean(A(:,i-raio:nl),2);
        end
        lambalis = lambda;
    case  2 // reduz o no. de dados
        A = absor;
        nd=size(A,1)
        nalis = fix(nl/(2*raio+1));
        Aalis = zeros(nd,nalis);
        lambalis = zeros(1,nalis);
        j=raio+1;
        for i=1:nalis
            disp([string(i)+'/'+string(nalis)])
            Aalis(:,i) = mean(A(:,j-raio:j+raio),2);  //  média entre i-raio e i+raio
            lambalis(i) = lambda(j);
            j= j + 2 * raio + 1;
        end
    otherwise
        disp(' Alisamento não realizado')
        Aalis = alis;
        lambalis = lambda;
end
endfunction




