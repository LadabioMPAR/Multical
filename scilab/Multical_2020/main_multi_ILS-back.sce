
//   ----------------------------------------------------------- //
//    MULTI_ILS.M            
//
//   ultima atualizaçao 2020
//   Marcelo P. A. Ribeiro
//   ----------------------------------------------------------- //

// Este programa AJUSTA e VALIDA - UTILIZANDO VALIDAÇAO CRUZADA - multicalibraçao
// utilizando o metodo de Regressao Linear Multipla (multiple linear regression,
// MLR ou, tambem chamada de inverse Least Square, ILS).
// Dois métodos para a reduçao da singularidade da matriz A'A podem ser
// utilizadas.
//
// A entrada ocorre em dois arquivos: 1o arquivo contém a matriz com
// concentrações, onde a linha é a amostra e a coluna é o componente; o
// 2o arquivo é uma matriz onde a primeira coluna é o número do comprimento,
// depois as linhas indicam amostras e colunas o valor de absorbância em
// cada comprimento de onda.
//
// Neste programa apenas um conjunto de dados deve ser inserido.
// Deste conjunto serao tirados os dados de ajuste e validaçao.
//

////==========================================================================================================////
////                                   INICIO DE INSERÇAO DE DADOS                                            ////
////==========================================================================================================////

clc; clear ; for i = 1:20; close ; end;

exec('lib_multi/func_multi_ILS.sci');


//// Método
Selecao = 1; // 1 = PLS; 2 = SPA; 3=PCR
// Para o SPA é necessário dar o primeiro comprimento de onda a ser
// utilizado, lini.
optkini = 2; // 0=> lini = lambda(1); 1=> lini = dado abaixo; 2=> otimiza lini.
lini = 0; // [nm]. Só tem sentido se optkini = 1

//// número máximo de regressores para avaliar na  Validacao Cruzada
kmax = 7; // está relacionado com os componentes que estão contidos na amostra (não colocar muito )

//------------------------------------------------------------------
//Número de Analitos (colunas no arquivo de concentrações)
// -----------------------------------------------------------------
nc = 5;
//------------------------------------------------------------------
//Nome dos Analitos e unidade de concentração
// -----------------------------------------------------------------
cname = ['di', 'gli', 'gal', 'gos3','gos4'];
unid = 'M';

// --------------------------------------------------------------
//Carregando os dados
// --------------------------------------------------------------
// x = matriz de concentracoes [nc x nd]
// absor = matriz com as absorbancias, sendo a primeira linha o comprimento
// de onda [nd+1 x nl]

//x0 = fscanfMat('X_Multi_NIRA2020.txt');     
//absor0 = fscanfMat('Abs_Multi_NIRA2020.txt'); 
//x0 = fscanfMat('X_Multi_UV2020.txt');     
//absor0 = fscanfMat('Abs_Multi_UV2020.txt'); 
//x00 = fscanfMat('X_Multi_UV2019.txt');     
//absor00 = fscanfMat('Abs_Multi_UV2019.txt');
//
//x0 = [x0 ; x00]
//absor0 = [absor0 ; absor00(2:$,:)] 

//x0 = fscanfMat('X_Multi_UV2019.txt');     
//absor0 = fscanfMat('Abs_Multi_UV2019.txt');

x0=[]
absor0 = []
for i=[3]
    if i<10 then
        arqent = '0'+string(i)+'.dat'
    else
        arqent = string(i)+'.dat'
    end
    xlendo = fscanfMat('x'+arqent);
    absorlendo = fscanfMat('abs'+arqent); 
    x0=[x0;xlendo]
    absor0=[absor0;absorlendo(2:$,:)]
end
absor0 = [absorlendo(1,:);absor0]


//x0(1,:)=[]
//absor0(2,:)=[] 

//x0 = fscanfMat('Conc_aj-8-UVNIR.txt');     
//absor0 = fscanfMat('UV_aj_8.txt'); 
//absor0 = fscanfMat('NIR_aj_8.txt'); 

// Fração de dados aleatórios para testes
frac_test =.0 ; //em geral de 0.0 a 0.4


// Set of pretreatment 
// Moving Average: {'MA',radius,Losing points = 1 or 2, plot=1 or 0}
// LOESS: {'Loess',alpha = [0.2-0.9], order = [1,2,...],plot=1 or 0}
// Savitzky and Golay Filter: {'SG',radius,Losing points = 1 or 2,poly order = integer, der order = integer, plot=1 or 0}   
// Derivative: {'Deriv',order,plot=1 or 0}
// Cutting regions: {'Cut',lower bound,upper bound,plot=1 or 0}
// Cutting maxAbs: {'CutAbs',maxAbs, action: warning = 0 or cutting = 1 ,plot=1 or 0}
// Control Baseline based on noninformation region: {'BLCtr',ini_lamb,final_lamb,Abs_value, plot=1 or 0}
// example: 
pretreat = [];
//pretreat = list({'MA',3,1,1}, {'Deriv',1,1},{'MA',1,2,1},{'Cut',240,350,1});
//pretreat = list({'MA',3,1,1}, {'Loess',0.2,1,1},{'Cut',240,350,1},{'CutAbs',3.5,1,1});

// UV
//pretreat = list({'MA',6,1,1}, {'Deriv',1,1},{'MA',6,1,1},{'Cut',260,550,1});
pretreat = list({'MA',6,1,1}, {'Deriv',1,1},{'MA',4,2,1},{'Cut',270,470,1});// der todos 2
pretreat = list({'SG',9,1,3,0,1},{'Cut',270,470,1});//  todos 2
pretreat = list({'SG',9,1,2,1,1},{'Cut',270,470,1});// der todos 2
pretreat = list({'SG',9,1,2,1,1},{'Cut',270,470,1});// der todos 2
pretreat = list({'MA',6,1,1},{'BLCtr',600,650,0,1},{'Cut',230,470,1});// controle de baseline todos

//pretreat = list({'SG',6,1,3,0,1},{'SG',6,1,2,1,1},{'Cut',270,470,1});// der todos 2
//pretreat = list({'SG',6,1,3,0,1}, {'Deriv',1,1},{'SG',6,1,3,0,1},{'Cut',270,470,1});// der todos 2

//pretreat = list({'Cut',270,550,1},{'MA',6,1,1});
//pretreat = list({'Cut',270,550,1},{'MA',6,1,1});
//pretreat = list({'MA',6,1,1},{'Cut',270,470,1}); // todos 

//NIRA
//pretreat = list({'MA',6,2,1});
//pretreat = list({'MA',6,1,1}, {'Deriv',1,1},{'MA',6,1,1},{'Cut',4000,5200,5400,8500,1});
//pretreat = list({'MA',6,1,1}, {'Deriv',1,1},{'MA',6,1,1},{'Cut',4000,5200,5400,8500,1});
//pretreat = list({'SG',9,1,3,0,1}); // der todos 2

//pretreat = list({'MA',6,1,1}, {'Deriv',2,1},{'MA',6,2,1});



// Set of Analysis
// Lambert-Beer: {'LB'}
// Principal Component Analysis (PCA): {'PCA'} 
analysis = list({'LB'},{'PCA'});
//analysis = list({'PCA'})
//analysis = []

// outlier analysis
outlier = 0; // 0 = no; 1 = yes.

// Gravar Xval para os diferentes número de regressores
gravarXval = 0; //1 = grava, 0 = não grava



flag = func_multi_ILS(Selecao,optkini,lini,kmax,nc,cname,unid,x0,absor0,frac_test,pretreat,analysis,outlier,gravarXval)


