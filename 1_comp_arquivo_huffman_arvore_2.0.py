import tkinter as tk
from tkinter import filedialog, messagebox
import heapq
import pickle
import os
from PIL import Image, ImageTk
import numpy as np

# Função para formatar tamanho do arquivo
def formatar_tamanho(bytes):
    kb = bytes / 1024
    if kb < 1024:
        return f"{kb:.2f} KB"
    else:
        return f"{kb / 1024:.2f} MB"

# Definição do nó da árvore de Huffman
class NoHuffman:
    def __init__(self, valor, frequencia):
        self.valor = valor
        self.frequencia = frequencia
        self.esquerda = None
        self.direita = None

    def __lt__(self, outro):
        return self.frequencia < outro.frequencia

# Construção da árvore de Huffman
def construir_arvore_huffman(frequencias):
    fila_prioridade = [NoHuffman(valor, freq) for valor, freq in frequencias.items()]
    heapq.heapify(fila_prioridade)

    while len(fila_prioridade) > 1:
        esquerda = heapq.heappop(fila_prioridade)
        direita = heapq.heappop(fila_prioridade)
        combinado = NoHuffman(None, esquerda.frequencia + direita.frequencia)
        combinado.esquerda = esquerda
        combinado.direita = direita
        heapq.heappush(fila_prioridade, combinado)

    return fila_prioridade[0]

# Gera os códigos binários baseados na árvore de Huffman
def gerar_codigos_huffman(no, prefixo="", tabela=None):
    if tabela is None:
        tabela = {}
    if no.valor is not None:
        tabela[no.valor] = prefixo
    else:
        if no.esquerda:
            gerar_codigos_huffman(no.esquerda, prefixo + "0", tabela)
        if no.direita:
            gerar_codigos_huffman(no.direita, prefixo + "1", tabela)
    return tabela

# Desenha a árvore de Huffman no Canvas
def desenhar_arvore(canvas, no, x, y, dx):
    if no is not None:
        raio = 25
        cor_fundo = "#00A2E8"
        cor_borda = "#005F8B"

        if no.esquerda:
            canvas.create_line(x, y + raio, x - dx, y + 80 - raio, width=2, fill="black", tags="arvore")
            desenhar_arvore(canvas, no.esquerda, x - dx, y + 80, dx // 2)

        if no.direita:
            canvas.create_line(x, y + raio, x + dx, y + 80 - raio, width=2, fill="black", tags="arvore")
            desenhar_arvore(canvas, no.direita, x + dx, y + 80, dx // 2)

        canvas.create_oval(x - raio, y - raio, x + raio, y + raio,
                           fill=cor_fundo, outline=cor_borda, width=3, tags="arvore")

        if no.valor is None:
            texto = f'{no.frequencia}'
        else:
            texto = f'{no.valor}\n({no.frequencia})'

        canvas.create_text(x, y, text=texto, font=("Arial", 9, "bold"), fill="white", tags="arvore")

# Processa as frequências e gera a árvore
def processar_frequencias(frequencias, imagem_array, nome_arquivo, tamanho_arquivo):
    canvas.delete("arvore")
    canvas.delete("logo")

    soma_frequencias = sum(frequencias.values())
    largura, altura = imagem_array.shape

    info_label.config(text=f"Arquivo: {nome_arquivo} | Tamanho: {largura}x{altura} | "
                           f"Soma das Frequências: {soma_frequencias} | Original: {tamanho_arquivo}")

    TOP_N = 256
    top_frequencias = dict(sorted(frequencias.items(), key=lambda item: item[1], reverse=True)[:TOP_N])

    arvore = construir_arvore_huffman(top_frequencias)
    desenhar_arvore(canvas, arvore, 550, 50, 250)

    tabela_huffman = gerar_codigos_huffman(arvore)
    imagem_codificada = "".join(tabela_huffman.get(pixel, "") for pixel in imagem_array.flatten())

    dados_comprimidos = {
        "tabela_huffman": tabela_huffman,
        "imagem_codificada": imagem_codificada,
        "tamanho": imagem_array.shape
    }

    with open("imagem_comprimida.bin", "wb") as arquivo:
        pickle.dump(dados_comprimidos, arquivo)

    tamanho_comprimido = os.path.getsize("imagem_comprimida.bin")
    tamanho_formatado = formatar_tamanho(tamanho_comprimido)

    messagebox.showinfo("Sucesso", f"Imagem comprimida salva como 'imagem_comprimida.bin'\n"
                                   f"Tamanho comprimido: {tamanho_formatado}")

# Carrega e processa a imagem
def carregar_imagem():
    arquivo = filedialog.askopenfilename(filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp")])
    if not arquivo:
        return

    tamanho_arquivo = formatar_tamanho(os.path.getsize(arquivo))

    imagem = Image.open(arquivo).convert("L")
    dados = np.array(imagem)
    valores, contagens = np.unique(dados, return_counts=True)
    frequencias = dict(zip(valores, contagens))

    nome_arquivo = os.path.basename(arquivo)
    processar_frequencias(frequencias, dados, nome_arquivo, tamanho_arquivo)

# Descompressão da imagem
def descomprimir_imagem():
    try:
        with open("imagem_comprimida.bin", "rb") as arquivo:
            dados_comprimidos = pickle.load(arquivo)
    except FileNotFoundError:
        messagebox.showerror("Erro", "Arquivo de compressão não encontrado!")
        return

    tabela_huffman = dados_comprimidos["tabela_huffman"]
    imagem_codificada = dados_comprimidos["imagem_codificada"]
    tamanho = dados_comprimidos["tamanho"]

    tabela_inversa = {codigo: valor for valor, codigo in tabela_huffman.items()}
    codigo_atual = ""
    imagem_decodificada = []

    for bit in imagem_codificada:
        codigo_atual += bit
        if codigo_atual in tabela_inversa:
            imagem_decodificada.append(tabela_inversa[codigo_atual])
            codigo_atual = ""

    imagem_recuperada = np.array(imagem_decodificada, dtype=np.uint8).reshape(tamanho)
    exibir_imagem(imagem_recuperada)

# Exibe a imagem descomprimida
def exibir_imagem(array_imagem):
    imagem = Image.fromarray(array_imagem)
    imagem.show()

# Interface gráfica
root = tk.Tk()
root.title("Compressão de Imagem com Huffman")
root.geometry("1100x800")

# Cabeçalho informativo
info_label = tk.Label(root, text="Nenhum arquivo carregado", font=("Arial", 12, "bold"), bg="#f0f0f0")
info_label.pack(fill="x", pady=5)

# Carrega a imagem de fundo (logo)
logo_image = Image.open("logo.png").resize((1100, 800), Image.Resampling.LANCZOS)
background_image = ImageTk.PhotoImage(logo_image)

# Canvas com a logo
canvas = tk.Canvas(root, width=1100, height=800)
canvas.pack()
canvas.create_image(0, 0, image=background_image, anchor="nw", tags="logo")

# Botões centralizados
frame_botoes = tk.Frame(root, bg="white")
frame_botoes.place(relx=0.5, rely=0.05, anchor="n")

botao_comprimir = tk.Button(frame_botoes, text="Carregar e Comprimir Imagem", command=carregar_imagem,
                            font=("Arial", 11, "bold"), bg="#008CBA", fg="white")
botao_comprimir.pack(side=tk.LEFT, padx=10)

botao_descomprimir = tk.Button(frame_botoes, text="Descomprimir e Exibir Imagem", command=descomprimir_imagem,
                               font=("Arial", 11, "bold"), bg="#4CAF50", fg="white")
botao_descomprimir.pack(side=tk.LEFT, padx=10)

root.mainloop()
