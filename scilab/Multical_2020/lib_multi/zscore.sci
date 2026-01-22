function [xnorm,xmed,xsig]=zscore(x)
    [m,n]=size(x);
    xnorm = x;
    xmed = mean(x,'r');
    xsig = stdev(x,'r');
    xnorm = (x-ndgrid(xmed,[1:m])')./ndgrid(xsig,[1:m])'
    
//    for i = 1:size(x,2)
//        xnorm(:,i) = (x(:,i)-xmed(i))./xsig(i);
//    end

endfunction
    
