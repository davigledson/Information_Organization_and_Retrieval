import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import heapq
import pickle
import os
from PIL import Image
import numpy as np
from math import log2


# Função para formatar tamanho do arquivo
def formatar_tamanho(bytes):
    kb = bytes / 1024
    if kb < 1024:
        return f"{kb:.2f} KB"
    else:
        return f"{kb / 1024:.2f} MB"


## Implementação do Huffman
class NoHuffman:
    def __init__(self, valor, frequencia):
        self.valor = valor
        self.frequencia = frequencia
        self.esquerda = None
        self.direita = None

    def __lt__(self, outro):
        return self.frequencia < outro.frequencia


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


## Implementação do Shannon-Fano
def shannon_fano(frequencias):
    # Ordena os símbolos por frequência (decrescente)
    simbolos = sorted(frequencias.items(), key=lambda x: -x[1])
    return _shannon_fano_rec(simbolos, "")


def _shannon_fano_rec(simbolos, prefixo):
    if len(simbolos) == 1:
        return {simbolos[0][0]: prefixo}

    # Encontra o ponto de divisão mais equilibrado
    total = sum(freq for _, freq in simbolos)
    metade = total / 2
    soma = 0
    for i, (_, freq) in enumerate(simbolos):
        soma += freq
        if soma >= metade:
            ponto_divisao = i + 1
            break

    # Divide e conquista
    esquerda = simbolos[:ponto_divisao]
    direita = simbolos[ponto_divisao:]

    codigos = {}
    codigos.update(_shannon_fano_rec(esquerda, prefixo + "0"))
    codigos.update(_shannon_fano_rec(direita, prefixo + "1"))

    return codigos


## Funções de compressão
def comprimir_huffman(frequencias, imagem_array):
    arvore = construir_arvore_huffman(frequencias)
    tabela = gerar_codigos_huffman(arvore)
    bits = "".join(tabela[pixel] for pixel in imagem_array.flatten())
    return tabela, bits


def comprimir_shannon(frequencias, imagem_array):
    tabela = shannon_fano(frequencias)
    bits = "".join(tabela[pixel] for pixel in imagem_array.flatten())
    return tabela, bits


## Funções de descompressão
def descomprimir_huffman(dados_comprimidos):
    tabela = dados_comprimidos["tabela"]
    bits = dados_comprimidos["bits"]
    tamanho = dados_comprimidos["tamanho"]

    tabela_inversa = {v: k for k, v in tabela.items()}
    codigo_atual = ""
    pixels = []

    for bit in bits:
        codigo_atual += bit
        if codigo_atual in tabela_inversa:
            pixels.append(tabela_inversa[codigo_atual])
            codigo_atual = ""

    return np.array(pixels, dtype=np.uint8).reshape(tamanho)


def descomprimir_shannon(dados_comprimidos):
    # Shannon-Fano usa a mesma abordagem de descompressão que Huffman
    return descomprimir_huffman(dados_comprimidos)


## Interface e processamento
def processar_imagem():
    arquivo = filedialog.askopenfilename(filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp")])
    if not arquivo:
        return

    # Carrega imagem
    imagem = Image.open(arquivo).convert("L")
    dados = np.array(imagem)
    valores, contagens = np.unique(dados, return_counts=True)
    frequencias = dict(zip(valores, contagens))

    nome_arquivo = os.path.basename(arquivo)
    tamanho_original = os.path.getsize(arquivo)
    largura, altura = dados.shape

    # Atualiza informações na interface
    info_label.config(
        text=f"Arquivo: {nome_arquivo} | Tamanho: {largura}x{altura} | Original: {formatar_tamanho(tamanho_original)}")

    # Processa com ambos algoritmos
    resultados = []
    for metodo, nome in [(comprimir_huffman, "Huffman"), (comprimir_shannon, "Shannon-Fano")]:
        tabela, bits = metodo(frequencias, dados)

        # Calcula tamanho teórico (em bits)
        tamanho_bits = len(bits)
        tamanho_bytes = (tamanho_bits + 7) // 8  # Arredonda para cima

        # Calcula entropia e comprimento médio
        total_pixels = largura * altura
        entropia = sum(-(freq / total_pixels) * log2(freq / total_pixels) for freq in frequencias.values())
        comp_medio = sum(len(tabela[val]) * freq for val, freq in frequencias.items()) / total_pixels

        eficiencia = (entropia / comp_medio) * 100 if comp_medio > 0 else 0

        resultados.append({
            "metodo": nome,
            "tabela": tabela,
            "bits": bits,
            "tamanho_bits": tamanho_bits,
            "tamanho_bytes": tamanho_bytes,
            "entropia": entropia,
            "comp_medio": comp_medio,
            "eficiencia": eficiencia
        })

    # Exibe resultados
    exibir_resultados(resultados, dados.shape)


def exibir_resultados(resultados, tamanho_imagem):
    # Cria nova janela para resultados
    result_window = tk.Toplevel()
    result_window.title("Resultados da Compressão")
    result_window.geometry("800x600")

    # Frame para tabela de resultados
    frame_tabela = tk.Frame(result_window)
    frame_tabela.pack(pady=10)

    # Tabela de comparação
    cols = ("Método", "Tamanho (bits)", "Tamanho (bytes)", "Entropia", "Comp. Médio", "Eficiência")
    tree = ttk.Treeview(frame_tabela, columns=cols, show="headings", height=2)

    for col in cols:
        tree.heading(col, text=col)
        tree.column(col, width=120, anchor="center")

    for res in resultados:
        tree.insert("", "end", values=(
            res["metodo"],
            res["tamanho_bits"],
            res["tamanho_bytes"],
            f"{res['entropia']:.2f}",
            f"{res['comp_medio']:.2f}",
            f"{res['eficiencia']:.2f}%"
        ))

    tree.pack()

    # Frame para botões de descompressão
    frame_botoes = tk.Frame(result_window)
    frame_botoes.pack(pady=20)

    for i, res in enumerate(resultados):
        btn = tk.Button(
            frame_botoes,
            text=f"Descomprimir com {res['metodo']}",
            command=lambda r=res, t=tamanho_imagem: descomprimir_e_exibir(r, t)
        )
        btn.pack(side="left", padx=10)


def descomprimir_e_exibir(dados_comprimidos, tamanho_imagem):
    # Prepara dados no formato esperado
    dados = {
        "tabela": dados_comprimidos["tabela"],
        "bits": dados_comprimidos["bits"],
        "tamanho": tamanho_imagem
    }

    # Descomprime
    if "Huffman" in dados_comprimidos["metodo"]:
        imagem = descomprimir_huffman(dados)
    else:
        imagem = descomprimir_shannon(dados)

    # Exibe imagem
    Image.fromarray(imagem).show()


# Interface principal
root = tk.Tk()
root.title("Compressão de Imagem - Huffman e Shannon-Fano")
root.geometry("800x600")

# Cabeçalho
info_label = tk.Label(root, text="Nenhum arquivo carregado", font=("Arial", 12), pady=10)
info_label.pack()

# Botão principal
btn_carregar = tk.Button(
    root,
    text="Carregar e Comprimir Imagem",
    command=processar_imagem,
    font=("Arial", 12),
    padx=20,
    pady=10
)
btn_carregar.pack(pady=50)

# Informações sobre os algoritmos
frame_info = tk.Frame(root)
frame_info.pack(pady=20)

tk.Label(frame_info, text="Algoritmos Implementados:", font=("Arial", 10, "bold")).pack()

tk.Label(frame_info, text="1. Huffman - Ótimo para distribuições de probabilidade arbitrárias", justify="left").pack(
    anchor="w")
tk.Label(frame_info, text="2. Shannon-Fano - Mais simples mas nem sempre ótimo", justify="left").pack(anchor="w")

root.mainloop()