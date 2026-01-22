// This file is part of the CardioVascular toolbox
// Copyright (C) 2012 - INRIA - Serge Steer
// This file must be used under the terms of the CeCILL.
// This source file is licensed as described in the file COPYING, which
// you should have received as part of this distribution.  The terms
// are also available at
// http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt

function [p,xt]=polyfit(x,y,n,w)
  weighted=argn(2)==4
  if size(x,'*')<>size(y,'*') then
    error(mprintf(_("%s: Arguments #%d and #%d must have the same sizes.\n"),"polyfit",1,2))
  end
  x=matrix(x,-1,1);
  y=matrix(y,-1,1);
  if weighted then
    if size(x,'*')<>size(w,'*') then
      error(mprintf(_("%s: Arguments #%d and #%d must have the same sizes.\n"),"polyfit",1,4))
    end
    w=matrix(w,-1,1);
    y=y.*w;
  end
  if argn(1)==2 then
    xt=[mean(x); stdev(x)];
    x=(x - xt(1))/xt(2);
  end

  // Construct Vandermonde matrix.
  nx=size(x,'*');
  V=zeros(nx,n+1);
  if  weighted then V(:,n+1)=w;else V(:,n+1)=1;end
  for j=n:-1:1
    V(:,j)=x.*V(:,j+1);
  end

  // Solve least squares problem.
  [Q,R]=qr(V,'e');
  p=R\(Q'*y);  
  if size(R,2)>size(R,1) then
    warning(mprintf(_("%s:  Solution is not unique; degree >= number of data points.\n"),"polyfit"))
  elseif cond(R)<1e-10
    if argn(1)==2 then
      warning(mprintf(_("%s:  Solution is poorly conditionned.\n"),"polyfit"))
    else
      warning(mprintf(_("%s:  Solution is poorly conditionned. Try centering and scaling.\n"),"polyfit"))
    end
  end
  p=poly(p($:-1:1),"x","c")
endfunction

