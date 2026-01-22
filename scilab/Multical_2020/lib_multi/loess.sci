function Y = loess(x,y,alpha,order)
// LOESS smoother: locally fitted polynomials 
// Parameters: 
// x,y: data points, x could be non regularly sampled.

// alpha: smoothing coefficient should be less than 1 (0.2-0.9) ratio
// used to select the nearest points
// order: polynomial order (1 or 2)
  if argn(2)<3 then
    error(msprintf(_("%s: Wrong number of input argument: %d to %d expected.\n"),"loess",3,4))
  end

  if type(x)<>1|~isreal(x)|and(size(x)>1) then
    error(msprintf(_("%s: Wrong type for argument %d: Real vector expected.\n"),"loess",1))
  end
  if type(y)<>1|~isreal(y)|and(size(y)>1) then
    error(msprintf(_("%s: Wrong type for argument %d: Real vector expected.\n"),"loess",2))
  end
  if size(x,'*')<>size(y,'*') then
        error(msprintf(_("%s: Wrong value for input arguments #%d and #%d: Same sizes expected.\n"),"loess",1,2))
  end
 
  if type(alpha)<>1|~isreal(alpha)|or(size(alpha)>1) then
    error(msprintf(_("%s: Wrong type for argument %d: Real scalar expected.\n"),"loess",3))
  end
  if alpha<=0|alpha>1 then
    error(msprintf(_("%s: Wrong value for input argument #%d: Must be in the interval [%s, %s].\n"),"loess",3,"0","1"))
  end
  if argn(2)==4 then
    if type(order)<>1|~isreal(order)|or(size(order)>1)|order<>int(order) then
      error(msprintf(_("%s: Wrong type for input argument #%d: An integer value expected.\n"),"loess",4))
    end
    if order<1 then
      error(msprintf(_("%s: Wrong value for input argument #%d: A positive integer expected"),"loess",4))
    end
  else
    order=2
  end
  // Scale x to [0,1] prevent ill conditioned fitting 
  x1 = mean(x); xr = max(x)-min(x); 
  x = (x-x1)/xr; 
 

  Y = zeros(x); // space holder & get correct dimensions) 

  n = length(x);      //  number of data points
  q = min(max(floor(alpha*n),order+3),n); // used for weight function width > 3 or so 

  //  perform a fit for each desired x point
  for i = 1:size(x,'*')
    deltax = abs(x(i)-x);     //  distances from this new point to data
    deltaxsort = gsort(deltax,'g','i'); 
    qthdeltax = deltaxsort(q);     // width of weight function
    arg = min(deltax/(qthdeltax*max(alpha,1)),1);
    weight = (1-abs(arg).^3).^3;  //  weight function for x distance
    index = find(weight>0);  //  select points with nonzero weights
    if size(index,"*") > order
      p = polyfit(x(index),y(index),order,weight(index));     
      Y(i) = horner(p,x(i));  //  evaluate fit at this new point
    else
      //disp('Not enough points')
      Y(i) = mean(y(index))+6; // keep same 
    end // if 
  end

endfunction 
