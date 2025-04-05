
"""
Bibliotecas necessárias:
- torch: Para computação tensorial com suporte a GPU (requer CUDA da NVIDIA)
- cv2: Processamento de imagens
- os: Operações de sistema de arquivos
- matplotlib: Visualização dos resultados
- time: Medição de tempo de processamento
- numpy: Manipulação de arrays
"""
import torch  # Importa a biblioteca PyTorch para computação tensorial com suporte a GPU
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import time  # Para medição de tempo

def pca_gpu_compress(image_path, k_components=30, output_dir='output_gpu'):
    """Compressão de imagem com PCA acelerado por GPU"""

    # Verificação inicial da GPU
    if not torch.cuda.is_available():
        raise RuntimeError("GPU não disponível. Instale o CUDA da NVIDIA: https://developer.nvidia.com/cuda-downloads")

    # Início da medição do tempo total
    inicio_total = time.time()

    # Criar diretório de saída
    os.makedirs(output_dir, exist_ok=True)

    # 1. Carregar a imagem (CPU)
    inicio_carregamento = time.time()
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada em {image_path}")

    original_size = os.path.getsize(image_path)
    tempo_carregamento = time.time() - inicio_carregamento

    # 2. Transferir dados para GPU e normalizar
    inicio_transferencia = time.time()
    img_tensor = torch.from_numpy(img.astype(np.float32)).cuda() / 255.0
    tempo_transferencia = time.time() - inicio_transferencia

    # 3. Calcular PCA na GPU
    inicio_pca = time.time()
    mean = torch.mean(img_tensor, dim=0)
    centered = img_tensor - mean

    # PCA via SVD (otimizado para GPU)
    _, _, V = torch.pca_lowrank(centered, q=k_components)
    components = V[:, :k_components]
    scores = torch.matmul(centered, components)
    tempo_pca = time.time() - inicio_pca

    # 4. Salvar dados comprimidos (transferindo para CPU)
    inicio_salvamento = time.time()
    compressed_path = os.path.join(output_dir, 'compressed_data.npz')
    np.savez_compressed(
        compressed_path,
        mean=mean.cpu().numpy().astype(np.float32),
        components=components.cpu().numpy().astype(np.float32),
        scores=scores.cpu().numpy().astype(np.float32),
        original_shape=img.shape
    )
    compressed_size = os.path.getsize(compressed_path)
    tempo_salvamento = time.time() - inicio_salvamento

    # 5. Reconstruir imagem (GPU)
    inicio_reconstrucao = time.time()
    reconstructed = torch.matmul(scores, components.T) + mean
    reconstructed = torch.clamp(reconstructed, 0, 1)
    reconstructed_img = (reconstructed.cpu().numpy() * 255).astype(np.uint8)
    tempo_reconstrucao = time.time() - inicio_reconstrucao

    # 6. Salvar imagens (CPU)
    inicio_salvamento_img = time.time()
    reconstructed_path = os.path.join(output_dir, 'reconstructed_PCA_GPU.jpg')
    cv2.imwrite(reconstructed_path, reconstructed_img)
    original_path = os.path.join(output_dir, 'original.jpg')
    cv2.imwrite(original_path, img)
    tempo_salvamento_img = time.time() - inicio_salvamento_img

    # 7. Métricas de desempenho
    tempo_total = time.time() - inicio_total

    # Memória da GPU
    memoria_alocada = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
    memoria_reservada = torch.cuda.memory_reserved() / (1024 ** 2)  # MB

    # 8. Mostrar resultados
    print(f"\n{' METRICAS GPU ':=^40}")
    print(f"Original: {original_size / 1024:.1f} KB")
    print(f"Comprimido: {compressed_size / 1024:.1f} KB")
    print(f"Taxa de compressão: {original_size / compressed_size:.1f}x")

    print(f"\n{' TEMPOS DE PROCESSAMENTO ':=^40}")
    print(f"- Carregamento (CPU): {tempo_carregamento:.4f} s")
    print(f"- Transferência CPU→GPU: {tempo_transferencia:.4f} s")
    print(f"- Cálculo PCA (GPU): {tempo_pca:.4f} s")
    print(f"- Reconstrução (GPU): {tempo_reconstrucao:.4f} s")
    print(f"- Salvamento (CPU): {tempo_salvamento + tempo_salvamento_img:.4f} s")
    print(f"→ TEMPO TOTAL: {tempo_total:.4f} s")

    print(f"\n{' USO DA GPU ':=^40}")
    print(f"- Memória alocada: {memoria_alocada:.2f} MB")
    print(f"- Memória reservada: {memoria_reservada:.2f} MB")
    print(f"- Dispositivo: {torch.cuda.get_device_name(0)}")

    # 9. Plotar comparação
    inicio_plot = time.time()
    eigenvalues = torch.sum(scores ** 2, dim=0) / (scores.shape[0] - 1)  # Autovalores aproximados

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed_img, cmap='gray')
    plt.title(f'GPU Comprimida (k={k_components})')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    explained_variance = torch.cumsum(eigenvalues, dim=0) / torch.sum(eigenvalues)
    plt.plot(explained_variance.cpu().numpy())
    plt.axvline(k_components, color='r', linestyle='--')
    plt.title('Variância Explicada (GPU)')
    plt.xlabel('Componentes Principais')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_gpu.png'))
    plt.show()
    tempo_plot = time.time() - inicio_plot

    return {
        'original': original_path,
        'compressed': compressed_path,
        'reconstructed': reconstructed_path,
        'metricas': {
            'tempo_total': tempo_total,
            'memoria_gpu_mb': memoria_alocada,
            'dispositivo': torch.cuda.get_device_name(0)
        }
    }


if __name__ == "__main__":
    # Verificação inicial do ambiente
    print("="*50)
    print(f"Configuração do ambiente:")
    print(f"- PyTorch versão: {torch.__version__}")
    print(f"- CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"- Dispositivo GPU: {torch.cuda.get_device_name(0)}")
        print(f"- Versão CUDA: {torch.version.cuda}")
    else:
        print("⚠️ GPU não detectada. Instale o CUDA Toolkit da NVIDIA")
        print("Download: https://developer.nvidia.com/cuda-downloads")
    print("="*50)

    # Configurações
    input_image = 'imgs/img_P.jpg'  # Substitua pelo seu caminho
    components = 10  # Número de componentes principais

    # Executar compressão
    print("\nIniciando compressão com PCA na GPU...")
    results = pca_gpu_compress(input_image, k_components=components)

    # Resultados
    print("\nArquivos salvos em:")
    print(f"- Original: {results['original']}")
    print(f"- Dados comprimidos: {results['compressed']}")
    print(f"- Reconstruída: {results['reconstructed']}")