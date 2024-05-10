
from arquivos import spec
from arquivos import ref as r
import pandas as pd
class Modelo:
    def __init__(self, X=[], y=[],comprimentos=[],analitos=''):
        self.X= X
        self.y= y
        self.comprimentos = comprimentos
        self.analitos = analitos


    def import_spec(self, arq_abs,tipo):
        arquivo=tipo(arq_abs)
        self.X.append(arquivo)

    def import_ref(self, arq_ref, tipo):
        pass





'''
from tkinter import *
from tkinter import filedialog
arquivo=[]
def openFile():
    filepath = filedialog.askopenfilename(
                                          title="Open file okay?",
                                          filetypes= (("text files","*.txt"),
                                          ("all files",".")))
    arquivo.append(pd.read_csv(filepath, delim_whitespace=True))


window = Tk()
button = Button(text="Open",command=openFile)
button.pack()
window.mainloop()
print(arquivo[0].columns)

'''