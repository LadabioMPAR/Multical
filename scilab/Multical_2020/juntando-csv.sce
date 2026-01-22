clc
clear
close


// pelo console vá à pasta onde tem os arquivos csv com as varreduras
arqsai = "Todos.csv" // nome do arquivo de saída com todos os espectros



//diret = dir()
//pasta = diret.name
//dire_pasta = dir(pasta)
dire_pasta = dir()
//fullpath = dire_pasta.name // diretorio da pasta selecionada
[path, fname, extension] = fileparts(dire_pasta.name) 
ind = find((extension == ".csv")& (fname+".csv" <>arqsai))// ind com arquivos csv
arqcsv = dire_pasta.name(ind) // arquivos csv
nomearqcsv = fname(ind)
nfiles = length(ind) //

for i =1:nfiles
    arq = arqcsv(i)
    pularlinhas = 2
    M = csvRead(arq, ";", ",", [], [], [], [], pularlinhas)
    if i==1 then
        Todos = zeros(size(M,1),nfiles+1)
        Todos(:,1:2) = M
    else
        Todos(:,i+1) = M(:,2)
    end
end
plot(Todos(:,1),Todos(:,2:$))

nomearqcsv_n=strsubst(nomearqcsv,",",".")
legend(nomearqcsv_n,-1)

Todosstr = [["cm-1" nomearqcsv_n'];string(Todos)]

filename = fullfile(arqsai);
csvWrite(Todosstr, filename);
