function [absor,lambda,x,ifig]=func_pretreatment(pretreat,absor,lambda,x,ifig)

    // Set of pretreatment 
    // Moving Average: {'MA',radius,Losing points = 1 or 2, plot=1 or 0}
    // LOESS: {'Loess',alpha = [0.2-0.9], order = [1,2,...],plot=1 or 0}
    // Savitzky and Golay Filter: {'SG',radius,Losing points = 1 or 2,poly order = integer, der order = integer, plot=1 or 0}    
    // Derivative: {'Deriv',order,plot=1 or 0}
    // Cutting regions: {'Cut',lower bound1,upper bound1, lb2, up2, ..., plot=1 or 0}
    // Cutting maxAbs: {'CutAbs',maxAbs, action: warning = 0 or cuting = 1 ,plot=1 or 0}
    // Control Baseline based on noninformation region: {'BLCtr',ini_lamb,final_lamb,Abs_value,plot=1 or 0}
    // example: 
    //pretreat = list({'MA',3,1,1}, {'Deriv',1,1},{'MA',3,1,1},{'Loess',0.2,1,1},{'Cut',240,350, 355,360,1});

    //==========================================
    //  Initializing Pretreatment
    //==========================================
    disp('-- Pretreatment --')
    scf(ifig); plot(lambda',absor'); ifig=ifig+1;
    ylabel = ('Absorbance units');
    xlabel = ('Wavelength (\lambda,nm)');
    xtitle('original data',xlabel,ylabel);

    for i=1:length(pretreat)
        ylabel = ('Absorbance units');
        pret = pretreat(i)
        title = string(i)+' - ' + pret{1}
        disp(title)
        select pret{1}
        case 'MA' then 
            // Moving Average: {'MA',radius,Losing points = 1 or 2, plot=1 or 0}
            raio = pret{2};
            alis = pret{3};
            graph = pret{4};
            [absor,lambda]=alisar(alis,absor,lambda,raio);  
        case 'SG' then 
            // Savitzky and Golay Filter: {'SG',radius,Losing points = 1 or 2,poly order = integer, der order = integer, plot=1 or 0}
            [lambda,indx] = gsort(lambda,'g','i') // increasing order
            absor = absor(:,indx); 
            raio = pret{2};
            alis = pret{3};
            ordem = pret{4};
            der_ordem = pret{5};
            graph = pret{6};
            janela = 2*raio+1;
            [nd,nl]=size(absor)
            absortemp=zeros(nd,nl)

            for i=1:nd
                absortemp(i,:)=sgolay_filt(lambda',absor(i,:)',ordem,janela,der_ordem)';
            end
            if alis==2 then
                indx = [raio+1:F:nl]
                absor =absortemp(:,indx)
                lambda = lambda(indx)
            else
                absor=absortemp
            end


        case 'Loess'then 
            // LOESS: {'Loess',alpha = [0.2-0.9], order = [1,2,...],plot=1 or 0]
            alpha = pret{2};
            order = pret{3};
            graph = pret{4};
            for i=1:size(absor,1)
                absor(i,:) = loess(lambda,absor(i,:),alpha,order)            
            end
        case 'Deriv' then
            // Derivative: {'Deriv',order,plot=1 or 0}
            order = pret{2};
            graph = pret{3};
            [dA,d2A] = diffmeu(absor,lambda);
            if order == 1
                absor = dA;
                title = title + ' 1st order';
                ylabel = 'a.u./nm';
            elseif order == 2
                absor = d2A;
                title = title + ' 2nd order';
                ylabel = 'a.u./nm\^2';
            else
                disp('order must be 1 or 2')
                disp('aborting')
                abort
            end
        case 'Cut' then
            // Cutting regions: {'Cut',lower bound1,upper bound1, lb2, ub2, ..., lb ncut, ub ncut, plot=1 or 0}
            ncut = (length(pret)-1)/2;
            lambdatemp = []
            absortemp = []
            for icut = 1:ncut
                ilb = icut*2
                iub = ilb+1
                lb = pret{ilb};
                ub = pret{iub};
                indc = find((lambda>=lb) & (lambda<=ub));
                lambdatemp = [lambdatemp lambda(indc)];
                absortemp = [absortemp absor(:,indc)];
            end
            //lb = pret{2};
            //ub = pret{3};
            [lambdatemp,indx] = gsort(lambdatemp,'g','i') // increasing order
            lambda = lambdatemp
            absor = absortemp(:,indx); 
            lambdatemp = []
            absortemp = []
            graph = pret{$};

        case 'CutAbs' then
            // Cutting maxAbs: {'CutAbs',maxAbs, action: warning = 0 or cutting = 1 ,plot=1 or 0}
            maxAbs = pret{2};
            action = pret{3};
            graph = pret{4};
            [indr,indc]=find(absor>=maxAbs);
            if ~isempty(indr) then
                disp('Warning: abs > absmax in the following spectra')
                disp(unique(indr)')
                if action == 1 then
                    disp('These lines were removed')
                end
                absor(indr,:)=[];
                if ~isempty(x) then
                    x(indr,:)=[];
                end
            end
        case 'BLCtr' then
            // Control Baseline based on noninformation region: {'BLCtr',ini_lamb,final_lamb,Abs_value}
            ini_lamb=pret{2}
            final_lamb=pret{3}
            Abs_value=pret{4} // mean value of absorbance at noninformation region
            graph = pret{$};
            ind = find((lambda>=ini_lamb) & (lambda<=final_lamb));
            Dabs = mean(absor(:,ind),'c')-Abs_value
            DabsMatr = repmat(Dabs,1,length(lambda))
            absor = absor - DabsMatr
            
        end
        if graph == 1 then
            scf(ifig); plot(lambda',absor'); ifig=ifig+1;
            xlabel = ('Wavelength (\lambda,nm)');
            xtitle(title,xlabel,ylabel);
        end

    end
endfunction
