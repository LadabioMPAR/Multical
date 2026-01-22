function [dA,d2A] = diffmeu(A,lambda)
    [I,J]=size(A);
    dA = zeros((A));
    delta = lambda(2)-lambda(1);
    dA(:,1) = (A(:,2)-A(:,1))/delta;
    delta = lambda(J)-lambda(J-1);
    dA(:,$) = (A(:,$)-A(:,$-1))/delta;
    delta = lambda(3:J)-lambda(1:J-2);
    for j=2:J-1
        dA(:,j)=(A(:,j+1)-A(:,j-1))/delta(j-1);
    end

    d2A = zeros((A));
    delta = lambda(2)-lambda(1);
    d2A(:,1) = (dA(:,2)-dA(:,1))/delta;
    d2A(:,$) = (dA(:,$)-dA(:,$-1))/delta;
    for j=2:J-1
        d2A(:,j)=(A(:,j+1)-2* A(:,j) + A(:,j-1))/(delta^2);
    end
endfunction
