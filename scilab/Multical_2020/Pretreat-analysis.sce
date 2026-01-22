
//   ----------------------------------------------------------- //
//
//   Pretreat-analysis.sce
//
//   Marcelo P. A. Ribeiro
//   ----------------------------------------------------------- //

// Este programa realiza pré-tratamento nos dados de absorbância e gera 
// arquivo com o resultado.

// O arquivo de entrada deve ter na primeira linha os valores dos comprimentos 
// de onda. Da segunda linha para baixo, a matriz deve conter as absorbâncias 
// dos espectros.

clc; clear ; for i = 1:100; close ; end;

exec('lib_multi/diffmeu.sci');
exec('lib_multi/alisar.sci');
exec('lib_multi/zscore.sci');
exec('lib_multi/loess.sci');
exec('lib_multi/polyfit.sci');
exec('lib_multi/func_pretreatment.sci');
exec('lib_multi/func_analysis.sci');

////===========================================================================////
////                   Data input                                               ////
////===========================================================================////


// --------------------------------------------------------------
//Carregando os dados
// --------------------------------------------------------------
// absor = matriz com as absorbancias, sendo a primeira linha o comprimento
// de onda [nd+1 x nl]

// input file of absorbances
arq_ent = 'AbsMultiCalExp_250-320.txt';
arq_sai = 'Alis-' + arq_ent;

// input file of concentrations -- needed if Cutting maxAbs or LB analsis are used
//arq_entx = []; 
arq_entx = 'XMultiCalExp.txt';  
arq_saix = 'Alis-' + arq_entx;   

// Save results in output file?  
arq_grava = 1; // 0 = no; 1 = yes (output files: ('pret-'+arq_ent) and ('pret-'+arq_entx) )


// Set of pretreatment 
// Moving Average: {'MA',radius,Losing points = 1 or 2, plot=1 or 0}
// LOESS: {'Loess',alpha = [0.2-0.9], order = [1,2,...],plot=1 or 0}
// Derivative: {'Deriv',order,plot=1 or 0}
// Cutting regions: {'Cut',lower bound,upper bound,plot=1 or 0}
// Cutting maxAbs: {'CutAbs',maxAbs, action: warning = 0 or cuting = 1 ,plot=1 or 0}
// Control Baseline based on noninformation region: {'BLCtr',ini_lamb,final_lamb,Abs_value,plot=1 or 0}
// example: 
//pretreat = list({'MA',3,1,1}, {'Deriv',1,1},{'MA',3,1,1},{'Loess',0.2,1,1},{'Cut',240,350,1});
pretreat = list({'MA',3,1,1}, {'Deriv',1,1},{'MA',3,1,1},{'Loess',0.2,1,1},{'Cut',240,350,1});

// Set of Analysis
// Lambert-Beer: {'LB'}
// Principal Component Analysis (PCA): {'PCA'} 
// examples:
//analysis = list({'LB'},{'PCA'});
//analysis = list({'PCA'})
analysis = list({'LB'},{'PCA'});
//analysis = [];


//==========================================
//  Initializing Pretreatment
//==========================================

if ~isempty(arq_entx) then
    x0 = fscanfMat(arq_entx);     
else
    x0 = [];
end

absor0 = fscanfMat(arq_ent);
lambda0=absor0(1,:); // first row is the wavenumber (or wavelength)
absor0=absor0(2:$,:);
absor = absor0;
lambda=lambda0;
x = x0;
ifig = 1;

[absor,lambda,x,ifig]=func_pretreatment(pretreat,absor,lambda,x,ifig)

[absor,lambda,x,ifig]=func_analysis(analysis,absor,lambda,x,ifig)


//==========================================
//  Saving data file
//==========================================

disp(' -- Saving files -- ')

if  arq_grava == 1 then

    disp('Saving absorbance file')
    disp(arq_sai)
    resultado = [lambda;absor];
    formato= '%f \t';
    for i=2:length(lambda)-1
        formato = formato + '%f \t';
    end
    formato = formato +' %f \n';
    [fd, err] = mopen(arq_sai , 'w')
    mfprintf(fd,formato,resultado);
    mclose(fd)

    if ~isempty(x) then
        disp('Saving concentration file:')
        disp(arq_saix)
        resultadox = x;
        formato= '%f \t';
        for i=2:size(x,2)-1
            formato = formato + '%f \t';
        end
        formato = formato +' %f \n';

        [fd, err] = mopen(arq_saix , 'w')
        mfprintf(fd,formato,resultadox);
        mclose(fd)
    end
end
disp('-- END --')
