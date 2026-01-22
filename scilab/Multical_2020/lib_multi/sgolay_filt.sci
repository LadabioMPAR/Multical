function yfilt = sgolay_filt(x,y,n,F,d)
    // Savitzky-Golay filtering - uniformemente espaçados
    // Data: (x,y) 
    // interpolating order n (Must be < n-1)
    // Window length: F (Must be < length(x))
    // d: differentiation order
    // y e x são vetores coluna

    N = length(x)
    // verificando se são linearmente espaçados
    dx = x(2:$)-x(1:$-1)
    ddx = dx(2:$) == dx(1:$-1)
    x_linspace = and(ddx)

    if F<n+1 then
        n = F-1
        disp("order reduction due to window size")
    end

    if ~x_linspace then
        disp("x must be linearly spaced")
    else

        // escalonando x = [xb .. xc] para xr = [-1..1]
        // x = Dx/2*(xr+1)+xb
        // xr = 2/Dx*(x-xb)-1

        Dx = x(F)-x(1)
        dxrdx = 2/Dx // para o cálculo da derivada
        xr = linspace(-1,1,F)        
        X = ones(F,n+1)
        for i=1:n
            X(:,i+1) = xr.^i
        end
        if d == 0 then
            S_uni = X*inv(X'*X)*X' // yfilt = S*y
        elseif d > 0 then
            dX = [zeros(F,1) X(:,1:n)]
            for i=2:n
                dX(:,i+1) = i*dX(:,i+1)
            end
            if d == 1 then // 1a derivada
                S_uni = dX*inv(X'*X)*X' // yfilt = S*y
            else // 2a derivada
                d2X = [zeros(F,1) dX(:,1:n)]
                for i=2:n
                    dX(:,i+1) = i*dX(:,i+1)
                end
                S_uni = d2X*inv(X'*X)*X' // yfilt = S*y   
            end
        end
    end

    raio = (F-1)/2
    yfilt = y

    // até o primeiro centro

    yfilt(1:raio+1) = S_uni(1:raio+1,:)*y(1:F)
    for i=raio+2:N-raio-1
        yfilt(i) = S_uni(raio+1,:)*y(i-raio:i+raio)
    end
    yfilt(N-raio:N) = S_uni(raio+1:F,:)*y(N-F+1:N)

    if d ~= 0  then
        yfilt = yfilt*dxrdx^d // passando de d^n(y)/dxr^n para d^n(y)/dxr^n     
    end


endfunction

//
//
//
//
//
//
//
//    [nlhs,nrhs] = argn();
//    if nrhs < 5
//        d=0; // default: no differentiation
//    end // if
//
//    d = round(d) // force integer
//    N = length(x) //
//
//    xr = max(x)-min(x);
//    x=(x-mean(x))/xr; // normalise x
//    xc = xr^d // correction for derivatives
//
//    if fix(F/2) == F/2
//        disp("Frame size must be odd")
//        F=F+1
//    end // if
//
//    if n>(F-1)
//        disp('Interpolating order > window width')
//        n = floor(F/2)
//    end // if
//
//    F2 = (F-1)/2; // should be integer
//    yf = y*%nan; // dummy start
//
//    for i=F2+1:N-F2
//        idx = i-F2:i+F2;
//        xloc = x(idx);
//        yloc = y(idx);
//        p = polyfit(xloc,yloc,n); // fitted polynomial
//        if d>0 // perhaps differentiate it
//            for k=1:d
//                p = derivat(p); // p = polyder(p);
//            end // for
//        end // if
//        yf(i) = horner(p,x(i))/xc; // centre point
//    end // for
//
//endfunction
