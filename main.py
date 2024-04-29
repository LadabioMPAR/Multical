
from arquivos import spec as s
from arquivos import ref as r

class Modelo:
    def __init__(self, X=[], y=[],comprimentos=[],analitos=''):
        self.X= X
        self.y= y
        self.comprimentos = comprimentos
        self.analitos = analitos


    def import_spec(self, arq_abs, tipo):
        pass
    def import_ref(self, arq_ref, tipo):
        pass





if __name__=='__main__':
    print('ok')
