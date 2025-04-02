from collections import defaultdict


def shannon_fano(symbols):
    if len(symbols) == 1:
        return {symbols[0][0]: ''}

    # Ordena por frequência decrescente
    symbols.sort(key=lambda x: -x[1])

    # Encontra o ponto de divisão mais equilibrado
    total = sum(freq for _, freq in symbols)
    half = 0
    for i in range(len(symbols)):
        half += symbols[i][1]
        if half >= total / 2:
            break

    # Divide e atribui bits
    left = symbols[:i + 1]
    right = symbols[i + 1:]
    codes = {}
    for symbol, code in shannon_fano(left).items():
        codes[symbol] = '0' + code
    for symbol, code in shannon_fano(right).items():
        codes[symbol] = '1' + code
    return codes


# Exemplo:
data = "ABRACADABRA"
freq = defaultdict(int)
for char in data:
    freq[char] += 1
symbols = list(freq.items())

codes = shannon_fano(symbols)
print("Códigos de Shannon-Fano:", codes)