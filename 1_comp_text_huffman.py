import tkinter as tk
import heapq


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
def construir_arvore_huffman(numeros):
    fila_prioridade = [NoHuffman(valor, valor) for valor in numeros]
    heapq.heapify(fila_prioridade)

    while len(fila_prioridade) > 1:
        esquerda = heapq.heappop(fila_prioridade)
        direita = heapq.heappop(fila_prioridade)
        combinado = NoHuffman(None, esquerda.frequencia + direita.frequencia)
        combinado.esquerda = esquerda
        combinado.direita = direita
        heapq.heappush(fila_prioridade, combinado)

    return fila_prioridade[0]


# Desenha a árvore de Huffman no Canvas
def desenhar_arvore(canvas, no, x, y, dx):
    if no is not None:
        raio = 25  # Aumentei o tamanho dos nós para melhorar a estética
        cor_fundo = "#add8e6"  # Azul claro mais agradável
        cor_borda = "#000000"  # Preto para melhor contraste

        # Desenha a conexão entre os nós (antes de desenhar os nós para que as linhas fiquem por trás)
        if no.esquerda:
            canvas.create_line(x, y + raio, x - dx, y + 80 - raio, width=2, fill="black")
            desenhar_arvore(canvas, no.esquerda, x - dx, y + 80, dx // 2)

        if no.direita:
            canvas.create_line(x, y + raio, x + dx, y + 80 - raio, width=2, fill="black")
            desenhar_arvore(canvas, no.direita, x + dx, y + 80, dx // 2)

        # Desenha os nós com estilo melhorado
        canvas.create_oval(x - raio, y - raio, x + raio, y + raio, fill=cor_fundo, outline=cor_borda, width=2)

        # Define o texto dentro do nó
        texto = f'{no.frequencia}' if no.valor is None else f'{no.valor}'
        canvas.create_text(x, y, text=texto, font=("Arial", 12, "bold"), fill="black")


# Processamento dos números fornecidos no próprio código
def processar_numeros():
    numeros = [45, 13, 12, 16, 9, 5]  # O usuário define os números aqui
    canvas.delete("all")  # Limpa o canvas
    arvore_huffman = construir_arvore_huffman(numeros)
    desenhar_arvore(canvas, arvore_huffman, 400, 50, 200)


# Configuração da interface gráfica
root = tk.Tk()
root.title("Árvore de Huffman")

# Configurações do Canvas
canvas = tk.Canvas(root, width=800, height=600, bg="white")
canvas.pack()

# Botão para gerar a árvore
botao = tk.Button(root, text="Gerar Árvore", command=processar_numeros, font=("Arial", 12, "bold"), bg="#4CAF50",
                  fg="white")
botao.pack(pady=10)

# Gera a árvore automaticamente ao iniciar
processar_numeros()

root.mainloop()
