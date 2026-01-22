function [absor,lambda,x,ifig]=func_analysis(analysis,absor,lambda,x,ifig)

    //==========================================
    //  Initializing Analysis
    //==========================================
    disp('-- Analysis --')

    for i=1:length(analysis)
        anal = analysis(i);
        disp(string(i)+' - '+anal{1})
        select anal{1}
        case 'LB' then 
            if isempty(x) then
                disp('WARNING: cannot run LB - no x matrix' );
            else

                // Lambert-Beer: {'LB'}
                [nd,nl]=size(absor);
                xone = x;
                K=xone\absor;
                absorc = xone*K;
                xymax = max([absor;absorc])
                xymin = min([absor;absorc])
                scf(ifig); plot(absor,absorc,'o',[xymin xymax], [xymin,xymax],'-k'); ifig = ifig+1;
                xlabel=('absor'); ylabel=('absor calculada L-B');
                title=('ajuste absorbância SEM termo idependente');  
                xtitle(title,xlabel,ylabel)

                xone = [ones(nd,1) x];
                K=xone\absor;
                absorc = xone*K;
                scf(ifig); plot(absor,absorc,'o',[xymin xymax], [xymin,xymax],'-k'); ifig = ifig+1;
                xlabel=('absor'); ylabel=('absor calculada L-B');
                title=('ajuste absorbância COM termo idependente'); 
                xtitle(title,xlabel,ylabel)

                disp('pausa')
                disp('escreva ''resume'' ou ''abort''');
                pause; disp('cont.'); disp('')
            end

        case 'PCA'then 
            // Principal Component Analysis (PCA): {'PCA'} 
            [anorm,amed,asig]=zscore(absor)
            [eigvec,eigval] = spec(anorm'*anorm);
            eigval = diag(eigval);
            scf(ifig); plot (lambda',eigvec(:,1:3),[min(lambda) max(lambda)],[0 0], 'k'); ifig=ifig+1;
            xlabel = ('Wavelength (\lambda,nm)');ylabel = ('PC');
            title = ('Principal Components')
            legend('PC1','PC2','PC3');
            xtitle(title,xlabel,ylabel);

            scf(ifig); plot(absor*eigvec(:,1),absor*eigvec(:,2),'x'); ifig=ifig+1;
            xlabel = ('PC1');ylabel = ('PC2');
            title = ('Principal Components')
            xtitle(title,xlabel,ylabel);

            vartot = sum(real(eigval))
            var_rel = eigval/vartot;
            var_ac = zeros(var_rel);

            for i=1:length(lambda)
                var_ac(i) = sum(var_rel(1:i));
            end
            
            ind = find(real(var_ac)<.9999);
//            disp(ind,'ind')
            maxind = max([ind 10]);            
            disp('Explained variance first PCs')
            disp(['PC#','var_relative', 'var_acummulated'])
            disp([[1:maxind]' var_rel(1:maxind) var_ac(1:maxind)] )
            disp('pausa');
            disp('escreva ''resume'' ou ''abort''');
            pause; disp('cont.'); disp('')
        end
    end


endfunction
