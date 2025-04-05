import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import heapq
from collections import Counter


class Node:
    def __init__(self, char=None, freq=0):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


class ShannonFanoNode:
    def __init__(self, symbols, freq):
        self.symbols = symbols
        self.freq = freq
        self.left = None
        self.right = None


class CompressaoTextoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Compressão de Texto - Huffman e Shannon-Fano")
        self.root.geometry("800x600")

        self.setup_ui()

    def setup_ui(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        self.text_input = tk.Text(frame, height=10)
        self.text_input.pack(fill=tk.X)

        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="Compressão Huffman", command=self.compress_huffman).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Compressão Shannon-Fano", command=self.compress_shannon_fano).pack(side=tk.LEFT, padx=5)

        self.output_text = tk.Text(frame, height=10)
        self.output_text.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(frame, bg="white", height=300)
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=10)

    def compress_huffman(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Erro", "Texto vazio!")
            return

        freq = Counter(text)
        total_chars = sum(freq.values())
        heap = [Node(char, freq) for char, freq in freq.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = Node(freq=left.freq + right.freq)
            merged.left = left
            merged.right = right
            heapq.heappush(heap, merged)

        root = heap[0]
        codes = {}
        self.generate_huffman_codes(root, "", codes)

        compressed = ''.join(codes[char] for char in text)

        freq_output = "\n".join(f"'{char}': {freq[char]} ({freq[char] / total_chars:.2%})" for char in codes)

        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, f"Codificação Huffman:\n{codes}\n\nTexto Comprimido:\n{compressed}\n\nFrequências:\n{freq_output}")
        self.canvas.delete("all")
        self.draw_huffman_tree(self.canvas, root, 400, 20, 180, total_chars)

    def generate_huffman_codes(self, node, current_code, codes):
        if node is None:
            return
        if node.char is not None:
            codes[node.char] = current_code
            return
        self.generate_huffman_codes(node.left, current_code + "0", codes)
        self.generate_huffman_codes(node.right, current_code + "1", codes)

    def compress_shannon_fano(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Erro", "Texto vazio!")
            return

        freq = Counter(text)
        total_chars = sum(freq.values())
        symbols = sorted(freq.items(), key=lambda item: item[1], reverse=True)
        tree = self.build_shannon_fano_tree(symbols)
        codes = {}
        self.generate_shannon_fano_codes(tree, "", codes)

        compressed = ''.join(codes[char] for char in text)

        freq_output = "\n".join(f"'{char}': {freq[char]} ({freq[char] / total_chars:.2%})" for char in codes)

        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, f"Codificação Shannon-Fano:\n{codes}\n\nTexto Comprimido:\n{compressed}\n\nFrequências:\n{freq_output}")
        self.canvas.delete("all")
        self.draw_shannon_tree(self.canvas, tree, 400, 20, 180, total_chars)

    def build_shannon_fano_tree(self, symbols):
        if len(symbols) == 1:
            return ShannonFanoNode([symbols[0][0]], symbols[0][1])

        total = sum(freq for _, freq in symbols)
        acc = 0
        for i in range(len(symbols)):
            acc += symbols[i][1]
            if acc >= total / 2:
                break

        left = self.build_shannon_fano_tree(symbols[:i+1])
        right = self.build_shannon_fano_tree(symbols[i+1:])
        node = ShannonFanoNode([sym[0] for sym in symbols], total)
        node.left = left
        node.right = right
        return node

    def generate_shannon_fano_codes(self, node, code, codes):
        if node.left is None and node.right is None:
            for symbol in node.symbols:
                codes[symbol] = code
            return

        if node.left:
            self.generate_shannon_fano_codes(node.left, code + "0", codes)
        if node.right:
            self.generate_shannon_fano_codes(node.right, code + "1", codes)

    def draw_shannon_tree(self, canvas, node, x, y, spacing, total):
        if node is None:
            return
        symbols = ','.join(node.symbols)
        perc = node.freq / total
        label = f"{symbols} ({node.freq}, {perc:.2%})"
        canvas.create_text(x, y, text=label, font=("Arial", 10))

        if node.left:
            canvas.create_line(x, y + 10, x - spacing, y + 60)
            self.draw_shannon_tree(canvas, node.left, x - spacing, y + 60, spacing // 2, total)

        if node.right:
            canvas.create_line(x, y + 10, x + spacing, y + 60)
            self.draw_shannon_tree(canvas, node.right, x + spacing, y + 60, spacing // 2, total)

    def draw_huffman_tree(self, canvas, node, x, y, spacing, total):
        if node is None:
            return
        label = f"{node.char if node.char else ''} ({node.freq}, {node.freq / total:.2%})"
        canvas.create_text(x, y, text=label, font=("Arial", 10))

        if node.left:
            canvas.create_line(x, y + 10, x - spacing, y + 60)
            self.draw_huffman_tree(canvas, node.left, x - spacing, y + 60, spacing // 2, total)

        if node.right:
            canvas.create_line(x, y + 10, x + spacing, y + 60)
            self.draw_huffman_tree(canvas, node.right, x + spacing, y + 60, spacing // 2, total)


if __name__ == "__main__":
    root = tk.Tk()
    app = CompressaoTextoApp(root)
    root.mainloop()
