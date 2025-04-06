import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import heapq
import pickle
import os
from PIL import Image
import numpy as np
from math import log2
from scipy.fftpack import dct, idct
import zlib


# Função para formatar tamanho do arquivo
def formatar_tamanho(bytes):
    kb = bytes / 1024
    if kb < 1024:
        return f"{kb:.2f} KB"
    else:
        return f"{kb / 1024:.2f} MB"


# Implementação do Huffman
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


# Implementação do Shannon-Fano
def shannon_fano(frequencias):
    simbolos = sorted(frequencias.items(), key=lambda x: -x[1])
    return _shannon_fano_rec(simbolos, "")


def _shannon_fano_rec(simbolos, prefixo):
    if len(simbolos) == 1:
        return {simbolos[0][0]: prefixo}

    total = sum(freq for _, freq in simbolos)
    metade = total / 2
    soma = 0
    for i, (_, freq) in enumerate(simbolos):
        soma += freq
        if soma >= metade:
            ponto_divisao = i + 1
            break

    esquerda = simbolos[:ponto_divisao]
    direita = simbolos[ponto_divisao:]

    codigos = {}
    codigos.update(_shannon_fano_rec(esquerda, prefixo + "0"))
    codigos.update(_shannon_fano_rec(direita, prefixo + "1"))

    return codigos


# Funções de compressão originais (sem DCT)
def comprimir_huffman(frequencias, imagem_array, nome_arquivo):
    os.makedirs("output_huffman", exist_ok=True)

    arvore = construir_arvore_huffman(frequencias)
    tabela = gerar_codigos_huffman(arvore)
    bits = "".join(tabela[pixel] for pixel in imagem_array.flatten())

    nome_saida = os.path.splitext(nome_arquivo)[0] + "_huffman.bin"
    caminho_saida = os.path.join("output_huffman", nome_saida)

    dados_comprimidos = {
        "tabela": tabela,
        "bits": bits,
        "tamanho": imagem_array.shape
    }

    with open(caminho_saida, "wb") as arquivo:
        pickle.dump(dados_comprimidos, arquivo)

    return tabela, bits, caminho_saida


def comprimir_shannon(frequencias, imagem_array, nome_arquivo):
    os.makedirs("output_shannon", exist_ok=True)

    tabela = shannon_fano(frequencias)
    bits = "".join(tabela[pixel] for pixel in imagem_array.flatten())

    nome_saida = os.path.splitext(nome_arquivo)[0] + "_shannon.bin"
    caminho_saida = os.path.join("output_shannon", nome_saida)

    dados_comprimidos = {
        "tabela": tabela,
        "bits": bits,
        "tamanho": imagem_array.shape
    }

    with open(caminho_saida, "wb") as arquivo:
        pickle.dump(dados_comprimidos, arquivo)

    return tabela, bits, caminho_saida


# Funções para DCT (Transformada Discreta de Cosseno)
def aplicar_dct(imagem_array, tamanho_bloco=8):
    """Aplica DCT em blocos da imagem, preenchendo se necessário"""
    altura, largura = imagem_array.shape

    # Calcula novo tamanho que seja múltiplo de 8
    nova_altura = ((altura + tamanho_bloco - 1) // tamanho_bloco) * tamanho_bloco
    nova_largura = ((largura + tamanho_bloco - 1) // tamanho_bloco) * tamanho_bloco

    # Cria uma nova imagem com padding se necessário
    imagem_padded = np.zeros((nova_altura, nova_largura), dtype=np.float32)
    imagem_padded[:altura, :largura] = imagem_array

    coeficientes = np.zeros_like(imagem_padded, dtype=np.float32)

    for i in range(0, nova_altura, tamanho_bloco):
        for j in range(0, nova_largura, tamanho_bloco):
            bloco = imagem_padded[i:i + tamanho_bloco, j:j + tamanho_bloco]
            coeficientes[i:i + tamanho_bloco, j:j + tamanho_bloco] = dct(dct(bloco.T, norm='ortho').T, norm='ortho')

    return coeficientes[:altura, :largura]  # Retorna sem o padding


def quantizar_coeficientes(coeficientes, fator_qualidade=50):
    """Quantiza os coeficientes DCT garantindo blocos 8x8"""
    if fator_qualidade <= 0:
        fator_qualidade = 1
    if fator_qualidade > 100:
        fator_qualidade = 100

    # Matriz de quantização básica (similar ao JPEG)
    matriz_quant = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    # Ajusta a matriz de quantização baseada no fator de qualidade
    if fator_qualidade < 50:
        escala = 5000 / fator_qualidade
    else:
        escala = 200 - 2 * fator_qualidade

    matriz_quant = np.floor((matriz_quant * escala + 50) / 100)
    matriz_quant[matriz_quant < 1] = 1

    altura, largura = coeficientes.shape
    quantizados = np.zeros_like(coeficientes)

    # Garante que processamos blocos completos de 8x8
    for i in range(0, altura, 8):
        for j in range(0, largura, 8):
            # Pega o bloco atual (pode ser menor que 8x8 nas bordas)
            bloco = coeficientes[i:i + 8, j:j + 8]
            bloco_shape = bloco.shape

            # Se o bloco for menor que 8x8, cria um bloco temporário 8x8
            if bloco_shape != (8, 8):
                temp_bloco = np.zeros((8, 8))
                temp_bloco[:bloco_shape[0], :bloco_shape[1]] = bloco
                temp_quant = np.round(temp_bloco / matriz_quant)
                quantizados[i:i + bloco_shape[0], j:j + bloco_shape[1]] = temp_quant[:bloco_shape[0], :bloco_shape[1]]
            else:
                quantizados[i:i + 8, j:j + 8] = np.round(bloco / matriz_quant)

    return quantizados.astype(np.int16)


def comprimir_com_dct(imagem_array, nome_arquivo, metodo_compressao, fator_qualidade=50):
    coeficientes = aplicar_dct(imagem_array.astype(np.float32))
    coeficientes_quant = quantizar_coeficientes(coeficientes, fator_qualidade)

    coeficientes_flat = coeficientes_quant.flatten()
    nao_zeros = coeficientes_flat[coeficientes_flat != 0]

    valores, contagens = np.unique(nao_zeros, return_counts=True)
    frequencias = dict(zip(valores, contagens))

    if metodo_compressao == "huffman":
        os.makedirs("output_huffman_dct", exist_ok=True)
        caminho_saida = os.path.join("output_huffman_dct", os.path.splitext(nome_arquivo)[0] + "_huffman_dct.bin")

        arvore = construir_arvore_huffman(frequencias)
        tabela = gerar_codigos_huffman(arvore)
        bits = "".join(tabela[val] for val in nao_zeros)
    else:  # shannon-fano
        os.makedirs("output_shannon_dct", exist_ok=True)
        caminho_saida = os.path.join("output_shannon_dct", os.path.splitext(nome_arquivo)[0] + "_shannon_dct.bin")

        tabela = shannon_fano(frequencias)
        bits = "".join(tabela[val] for val in nao_zeros)

    dados_comprimidos = {
        "metodo": metodo_compressao,
        "tabela": tabela,
        "bits": bits,
        "tamanho": imagem_array.shape,
        "coeficientes_zeros": len(coeficientes_flat) - len(nao_zeros),
        "fator_qualidade": fator_qualidade
    }

    with open(caminho_saida, "wb") as arquivo:
        dados_compactados = zlib.compress(pickle.dumps(dados_comprimidos))
        arquivo.write(dados_compactados)

    return tabela, bits, caminho_saida


def reconstruir_imagem_dct(coeficientes_quantizados, fator_qualidade=50):
    """Reconstrói a imagem lidando com blocos incompletos"""
    if fator_qualidade <= 0:
        fator_qualidade = 1
    if fator_qualidade > 100:
        fator_qualidade = 100

    matriz_quant = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    if fator_qualidade < 50:
        escala = 5000 / fator_qualidade
    else:
        escala = 200 - 2 * fator_qualidade

    matriz_quant = np.floor((matriz_quant * escala + 50) / 100)
    matriz_quant[matriz_quant < 1] = 1

    altura, largura = coeficientes_quantizados.shape
    coeficientes = np.zeros_like(coeficientes_quantizados, dtype=np.float32)

    for i in range(0, altura, 8):
        for j in range(0, largura, 8):
            bloco = coeficientes_quantizados[i:i + 8, j:j + 8]
            bloco_shape = bloco.shape

            if bloco_shape != (8, 8):
                temp_bloco = np.zeros((8, 8))
                temp_bloco[:bloco_shape[0], :bloco_shape[1]] = bloco
                temp_bloco = temp_bloco * matriz_quant
                coeficientes[i:i + bloco_shape[0], j:j + bloco_shape[1]] = temp_bloco[:bloco_shape[0], :bloco_shape[1]]
            else:
                coeficientes[i:i + 8, j:j + 8] = bloco * matriz_quant

    # Reconstrução da imagem com IDCT
    imagem_reconstruida = np.zeros_like(coeficientes, dtype=np.float32)

    for i in range(0, altura, 8):
        for j in range(0, largura, 8):
            bloco = coeficientes[i:i + 8, j:j + 8]
            bloco_shape = bloco.shape

            if bloco_shape != (8, 8):
                temp_bloco = np.zeros((8, 8))
                temp_bloco[:bloco_shape[0], :bloco_shape[1]] = bloco
                temp_bloco = idct(idct(temp_bloco.T, norm='ortho').T, norm='ortho')
                imagem_reconstruida[i:i + bloco_shape[0], j:j + bloco_shape[1]] = temp_bloco[:bloco_shape[0],
                                                                                  :bloco_shape[1]]
            else:
                imagem_reconstruida[i:i + 8, j:j + 8] = idct(idct(bloco.T, norm='ortho').T, norm='ortho')

    imagem_reconstruida = np.clip(imagem_reconstruida, 0, 255)
    return imagem_reconstruida.astype(np.uint8)


# Funções de descompressão
def descomprimir_huffman(caminho_arquivo):
    with open(caminho_arquivo, "rb") as arquivo:
        dados_comprimidos = pickle.load(arquivo)

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


def descomprimir_shannon(caminho_arquivo):
    return descomprimir_huffman(caminho_arquivo)


def descomprimir_com_dct(caminho_arquivo):
    with open(caminho_arquivo, "rb") as arquivo:
        dados_compactados = arquivo.read()
        dados_comprimidos = pickle.loads(zlib.decompress(dados_compactados))

    tabela = dados_comprimidos["tabela"]
    bits = dados_comprimidos["bits"]
    tamanho = dados_comprimidos["tamanho"]
    zeros = dados_comprimidos["coeficientes_zeros"]
    fator_qualidade = dados_comprimidos["fator_qualidade"]

    tabela_inversa = {v: k for k, v in tabela.items()}
    codigo_atual = ""
    coeficientes_nao_zero = []

    for bit in bits:
        codigo_atual += bit
        if codigo_atual in tabela_inversa:
            coeficientes_nao_zero.append(tabela_inversa[codigo_atual])
            codigo_atual = ""

    total_coeficientes = tamanho[0] * tamanho[1]
    coeficientes_flat = np.zeros(total_coeficientes, dtype=np.int16)

    indices_nao_zero = np.where(coeficientes_flat == 0)[0][:len(coeficientes_nao_zero)]
    coeficientes_flat[indices_nao_zero] = coeficientes_nao_zero

    coeficientes_quant = coeficientes_flat.reshape(tamanho)

    return reconstruir_imagem_dct(coeficientes_quant, fator_qualidade)


# Interface e processamento
def processar_imagem():
    arquivo = filedialog.askopenfilename(filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp")])
    if not arquivo:
        return

    # Carrega imagem
    imagem = Image.open(arquivo).convert("L")
    dados = np.array(imagem)
    nome_arquivo = os.path.basename(arquivo)
    tamanho_original = os.path.getsize(arquivo)
    largura, altura = dados.shape

    # Atualiza informações na interface
    info_label.config(
        text=f"Arquivo: {nome_arquivo} | Tamanho: {largura}x{altura} | Original: {formatar_tamanho(tamanho_original)}")

    # Processa com ambos algoritmos
    resultados = []
    for metodo, nome in [("huffman", "Huffman"), ("shannon", "Shannon-Fano")]:
        # Compressão original
        valores, contagens = np.unique(dados, return_counts=True)
        frequencias = dict(zip(valores, contagens))

        if metodo == "huffman":
            tabela, bits, caminho_saida = comprimir_huffman(frequencias, dados, nome_arquivo)
        else:
            tabela, bits, caminho_saida = comprimir_shannon(frequencias, dados, nome_arquivo)

        tamanho_comprimido = os.path.getsize(caminho_saida)
        total_pixels = largura * altura
        entropia = sum(-(freq / total_pixels) * log2(freq / total_pixels) for freq in frequencias.values())
        comp_medio = sum(len(tabela[val]) * freq for val, freq in frequencias.items()) / total_pixels
        eficiencia = (entropia / comp_medio) * 100 if comp_medio > 0 else 0

        resultados.append({
            "metodo": nome,
            "tabela": tabela,
            "bits": bits,
            "caminho_saida": caminho_saida,
            "tamanho_comprimido": tamanho_comprimido,
            "tamanho_bits": len(bits),
            "entropia": entropia,
            "comp_medio": comp_medio,
            "eficiencia": eficiencia,
            "versao": "Original"
        })

        # Compressão com DCT
        tabela_dct, bits_dct, caminho_saida_dct = comprimir_com_dct(dados, nome_arquivo, metodo, 75)

        tamanho_comprimido_dct = os.path.getsize(caminho_saida_dct)

        valores_dct, contagens_dct = np.unique([tabela_dct[val] for val in tabela_dct], return_counts=True)
        freq_dct = dict(zip(valores_dct, contagens_dct))
        total_bits_dct = sum(len(cod) * freq for cod, freq in freq_dct.items())
        comp_medio_dct = total_bits_dct / sum(freq_dct.values()) if freq_dct else 0

        entropia_dct = sum(-(freq / sum(freq_dct.values())) * log2(freq / sum(freq_dct.values())) for freq in
                           freq_dct.values()) if freq_dct else 0
        eficiencia_dct = (entropia_dct / comp_medio_dct) * 100 if comp_medio_dct > 0 else 0

        resultados.append({
            "metodo": nome,
            "tabela": tabela_dct,
            "bits": bits_dct,
            "caminho_saida": caminho_saida_dct,
            "tamanho_comprimido": tamanho_comprimido_dct,
            "tamanho_bits": len(bits_dct),
            "entropia": entropia_dct,
            "comp_medio": comp_medio_dct,
            "eficiencia": eficiencia_dct,
            "versao": "Com DCT"
        })

    # Exibe resultados
    exibir_resultados(resultados)


def exibir_resultados(resultados):
    # Cria nova janela para resultados
    result_window = tk.Toplevel()
    result_window.title("Resultados da Compressão")
    result_window.geometry("1000x700")

    # Frame para tabela de resultados
    frame_tabela = tk.Frame(result_window)
    frame_tabela.pack(pady=10)

    # Tabela de comparação
    cols = ("Método", "Versão", "Tamanho (bytes)", "Tamanho (bits)", "Entropia", "Comp. Médio", "Eficiência", "Arquivo")
    tree = ttk.Treeview(frame_tabela, columns=cols, show="headings", height=4)

    for col in cols:
        tree.heading(col, text=col)
        tree.column(col, width=100, anchor="center")

    tree.column("Versão", width=80)
    tree.column("Arquivo", width=150)

    for res in resultados:
        tree.insert("", "end", values=(
            res["metodo"],
            res["versao"],
            formatar_tamanho(res["tamanho_comprimido"]),
            res["tamanho_bits"],
            f"{res['entropia']:.2f}",
            f"{res['comp_medio']:.2f}",
            f"{res['eficiencia']:.2f}%",
            os.path.basename(res["caminho_saida"])
        ))

    tree.pack()

    # Frame para botões de descompressão
    frame_botoes = tk.Frame(result_window)
    frame_botoes.pack(pady=20)

    for i, res in enumerate(resultados):
        btn = tk.Button(
            frame_botoes,
            text=f"Descomprimir {res['metodo']} {res['versao']}",
            command=lambda r=res: descomprimir_e_exibir(r)
        )
        btn.pack(side="left", padx=5)


def descomprimir_e_exibir(dados_comprimidos):
    # Descomprime
    if "DCT" in dados_comprimidos["versao"]:
        imagem = descomprimir_com_dct(dados_comprimidos["caminho_saida"])
    else:
        if "Huffman" in dados_comprimidos["metodo"]:
            imagem = descomprimir_huffman(dados_comprimidos["caminho_saida"])
        else:
            imagem = descomprimir_shannon(dados_comprimidos["caminho_saida"])

    # Exibe imagem
    Image.fromarray(imagem).show()


# Interface principal
root = tk.Tk()
root.title("Compressão de Imagem - Huffman, Shannon-Fano e DCT")
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
tk.Label(frame_info, text="1. Huffman (Original e com DCT)", justify="left").pack(anchor="w")
tk.Label(frame_info, text="2. Shannon-Fano (Original e com DCT)", justify="left").pack(anchor="w")

root.mainloop()