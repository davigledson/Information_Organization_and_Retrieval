import cv2
import numpy as np
from collections import defaultdict
import os
import pickle
from math import log2


class ShannonFanoNode:
    def __init__(self, symbol=None, freq=0):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None
        self.code = ''


def build_shannon_fano_tree(symbols):
    """Constroi a árvore de Shannon-Fano top-down"""
    if len(symbols) == 1:
        return symbols[0]

    # Ordena por frequência decrescente
    symbols.sort(key=lambda x: -x.freq)

    # Encontra o ponto de divisão mais equilibrado
    total = sum(node.freq for node in symbols)
    half = 0
    split_idx = 0

    for i, node in enumerate(symbols):
        half += node.freq
        if half >= total / 2:
            split_idx = i
            break

    left = symbols[:split_idx + 1]
    right = symbols[split_idx + 1:]

    parent = ShannonFanoNode()
    parent.left = build_shannon_fano_tree(left) if left else None
    parent.right = build_shannon_fano_tree(right) if right else None

    return parent


def assign_codes(node, code='', code_map=None):
    """Atribui códigos binários recursivamente"""
    if code_map is None:
        code_map = {}

    if node is None:
        return code_map

    if node.symbol is not None:
        node.code = code
        code_map[node.symbol] = code
        return code_map

    assign_codes(node.left, code + '0', code_map)
    assign_codes(node.right, code + '1', code_map)

    return code_map


def compress_image(image_path, output_dir='output_shannon'):
    """Compressão completa de imagem com Shannon-Fano"""
    # 1. Pré-processamento
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    pixels = img.flatten()
    original_size = img.nbytes

    # 2. Cálculo de frequências
    freq = defaultdict(int)
    for pixel in pixels:
        freq[pixel] += 1

    # 3. Construção da árvore
    nodes = [ShannonFanoNode(symbol=p, freq=f) for p, f in freq.items()]
    root = build_shannon_fano_tree(nodes)
    code_map = assign_codes(root)

    # 4. Codificação
    compressed_bits = ''.join([code_map[pixel] for pixel in pixels])

    # 5. Empacotamento dos bits
    padding = (8 - len(compressed_bits) % 8) % 8
    compressed_bits += '0' * padding

    byte_array = bytearray()
    for i in range(0, len(compressed_bits), 8):
        byte = compressed_bits[i:i + 8]
        byte_array.append(int(byte, 2))

    # 6. Salvamento
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'compressed.sf')

    metadata = {
        'shape': img.shape,
        'code_map': code_map,
        'padding': padding
    }

    with open(output_path, 'wb') as f:
        pickle.dump((metadata, byte_array), f)

    compressed_size = os.path.getsize(output_path)
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0

    print(f"\n{' ESTATÍSTICAS ':=^40}")
    print(f"Original: {original_size / 1024:.2f} KB")
    print(f"Comprimido: {compressed_size / 1024:.2f} KB")
    print(f"Taxa de compressão: {compression_ratio:.2f}x")
    print(f"Entropia: {calculate_entropy(freq):.2f} bits/símbolo")

    return output_path


def decompress_image(input_path, output_dir='output_shannon'):
    """Descompressão da imagem"""
    with open(input_path, 'rb') as f:
        metadata, byte_array = pickle.load(f)

    # Reconstrução do bitstream
    bitstream = ''.join(f'{byte:08b}' for byte in byte_array)
    bitstream = bitstream[:-metadata['padding']]

    # Inversão do mapa de códigos
    reverse_map = {v: k for k, v in metadata['code_map'].items()}

    # Decodificação
    current_code = ''
    decoded_pixels = []

    for bit in bitstream:
        current_code += bit
        if current_code in reverse_map:
            decoded_pixels.append(reverse_map[current_code])
            current_code = ''

    # Reconstrução da imagem
    img_array = np.array(decoded_pixels, dtype=np.uint8).reshape(metadata['shape'])

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'decompressed.png')
    cv2.imwrite(output_path, img_array)

    print(f"\nImagem reconstruída salva em: {output_path}")
    return output_path


def calculate_entropy(freq_dict):
    """Calcula a entropia da fonte"""
    total = sum(freq_dict.values())
    entropy = -sum((freq / total) * log2(freq / total) for freq in freq_dict.values() if freq > 0)
    return entropy


if __name__ == "__main__":
    # Exemplo de uso
    input_img = 'imgs/img_P.jpg'
    print("Iniciando compressão...")
    compressed_file = compress_image(input_img)

    print("\nIniciando descompressão...")
    decompress_image(compressed_file)
