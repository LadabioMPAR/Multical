import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os

# Estilos
fonte_titulo = ("Helvetica", 16, "bold")
fonte_normal = ("Helvetica", 12)
cor_fundo = "#77B0AA"
cor_botao = "#003C43"
cor_botao_texto = "#FFFFFF"
cor_listbox = "#E3FEF7"  # Nova cor de fundo da Listbox

def selecionar_comprimentos():
    arquivos = filedialog.askopenfilenames(
        title="Selecione os arquivos de espectros",
        filetypes=[("Arquivos de texto", "*.txt"), ("Arquivos .DAT", "*.dat"), ("Arquivos Excel", "*.xlsx")],
        initialdir="."
    )
    
    lista_comprimentos.delete(0, tk.END)
    for arquivo in arquivos:
        nome_arquivo = os.path.basename(arquivo)
        lista_comprimentos.insert(tk.END, nome_arquivo)
        arquivos_comprimentos.append(arquivo)

def confirmar_comprimentos():
    frame_comprimentos.pack_forget()
    frame_referencias.pack()

def selecionar_referencias():
    arquivos = filedialog.askopenfilenames(
        title="Selecione os arquivos de referências",
        filetypes=[("Arquivos de texto", "*.txt"), ("Arquivos .DAT", "*.dat"), ("Arquivos Excel", "*.xlsx")],
        initialdir="."
    )
    
    lista_referencias.delete(0, tk.END)
    for arquivo in arquivos:
        nome_arquivo = os.path.basename(arquivo)
        lista_referencias.insert(tk.END, nome_arquivo)
        arquivos_referencias.append(arquivo)

def finalizar():
    workspace = {
        "comprimentos": arquivos_comprimentos,
        "referencias": arquivos_referencias
    }
    
    with open("workspace.json", "w") as file:
        json.dump(workspace, file, indent=4)
    
    # Exibe uma mensagem de sucesso e fecha o aplicativo
    messagebox.showinfo("Sucesso", "Dados salvos com sucesso!")
    janela.destroy()

# Inicialização da janela principal
janela = tk.Tk()
janela.title("Importar dados")
janela.geometry("600x400")  # Define o tamanho da janela


janela.configure(bg=cor_fundo)

# Listas para armazenar os caminhos dos arquivos
arquivos_comprimentos = []
arquivos_referencias = []

# Frame para a primeira etapa (seleção de comprimentos)
frame_comprimentos = tk.Frame(janela, bg=cor_fundo)
frame_comprimentos.pack(pady=10)

label_comprimentos = tk.Label(frame_comprimentos, text="Selecione arquivos de espectros", font=fonte_titulo, bg=cor_fundo)
label_comprimentos.pack(pady=10)

botao_comprimentos = tk.Button(frame_comprimentos, text="Selecionar Espectros", font=fonte_normal, bg=cor_botao, fg=cor_botao_texto, command=selecionar_comprimentos)
botao_comprimentos.pack(pady=10)

lista_comprimentos = tk.Listbox(frame_comprimentos, width=80, height=10, font=fonte_normal, bg=cor_listbox)  # Define a cor de fundo da Listbox
lista_comprimentos.pack(pady=10)

botao_confirmar_comprimentos = tk.Button(frame_comprimentos, text="OK", font=fonte_normal, bg=cor_botao, fg=cor_botao_texto, command=confirmar_comprimentos)
botao_confirmar_comprimentos.pack(pady=10)

# Frame para a segunda etapa (seleção de referências)
frame_referencias = tk.Frame(janela, bg=cor_fundo)

label_referencias = tk.Label(frame_referencias, text="Selecione arquivos de referências", font=fonte_titulo, bg=cor_fundo)
label_referencias.pack(pady=10)

botao_referencias = tk.Button(frame_referencias, text="Selecionar Referências", font=fonte_normal, bg=cor_botao, fg=cor_botao_texto, command=selecionar_referencias)
botao_referencias.pack(pady=10)

lista_referencias = tk.Listbox(frame_referencias, width=80, height=10, font=fonte_normal, bg=cor_listbox)  # Define a cor de fundo da Listbox
lista_referencias.pack(pady=10)

botao_finalizar = tk.Button(frame_referencias, text="OK", font=fonte_normal, bg=cor_botao, fg=cor_botao_texto, command=finalizar)
botao_finalizar.pack(pady=10)

# Executa o loop principal da interface
janela.mainloop()
