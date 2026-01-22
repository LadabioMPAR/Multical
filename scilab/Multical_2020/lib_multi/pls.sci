// Function:
// [B,T,P,U,Q,W,r2X,r2Y] = pls(X,Y,h)
//
// Aim:
// General partial least squares regression using the NIPALS algorithm
//
// Input:
// X, matrix (m,n), data matrix with m samples and n variables (assumed to
// be centered)
// Y, matrix (m,k) or vector (m,1), responses or response (assumed to
// be centered)
// h, number of components of the PLS model
// 
// Output:
// B, matrix (n,k), a set of regression coefficients for each response
// T, matrix (m,h), containing in columns X-scores
// P, matrix (n,h), containing in columns X-loadings
// U, matrix (k,h), containing in columns Y-scores
// Q, matrix (m,h), containing in columns Y-loadings
// W, matrix (n,h), containing in columns X-weights 
// r2X, vector (1,h), percentage of explained variance in X
// r2Y, vector (1,h), percentage of explained variance in Y
// 
// Reference:
// B.G.M. Vandeginste, D.L. Massart, L.M.C. Buydens, S. de Jong, P.J. Lewi, 
// J. Smeyers-Verbeke, Handbook of Chemometrics and Qualimetrics, Part B,
// Elsevier, Amsterdam, 1998, p. 336

function [B,T,P,U,Q,W,r2X,r2Y] = pls(X,Y,h)

[m,n] = size(X);

E = X;  //-ones(m,1)*mean(X);
F = Y;  //-ones(m,1)*mean(Y);
ssX = sum(E(:).^2);
ssY = sum(Y(:).^2);

uold = ones(m,1)*100;

for i = 1:h;
   
    u = F(:,1);

    while norm(uold-u)>1e-5
        uold = u;
        w = E'*u;
        w = w/norm(w);
        t = E*w;
        q = F'*t/(t'*t);
        u = F*q/sqrt(q'*q);
    end
    
    p = E'*t/(t'*t);

    W(:,i) = w;       // X-weights
    P(:,i) = p;       // X-loadings
    T(:,i) = t;       // X-scores
    U(:,i) = u;       // Y-scores
    Q(:,i) = q;       // Y-loadings
    
    E = E-t*p';
    F = F-t*q';
    
    uold = ones(m,1)*100;
    
    r2X(i) = 100*(t'*t*(p'*p)/ssX);     // Percentage of explained variance in X
    r2Y(i) = 100*(t'*t*(q'*q)/ssY);     // Percentage of explained variance in Y
end

B = W*pinv(P'*W)*Q';
// B = Xtr' * U * inv(T' * Xtr * Xtr' * U) * T' * Ytr; // Rafael - mais preciso!
endfunction
