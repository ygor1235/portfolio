from bs4 import BeautifulSoup

# Simulando o código HTML que o robô baixou de um site de tecnologia
html_da_pagina = """
<html>
    <body>
        <div class="produto">
            <h1 class="titulo-produto">Placa de Vídeo RTX 4060</h1>
            <span class="preco-valor">2100.00</span>
        </div>
        <div class="produto">
            <h1 class="titulo-produto">Processador Ryzen 7</h1>
            <span class="preco-valor">1850.00</span>
        </div>
    </body>
</html>
"""


def raspar_dados(html):
    soup = BeautifulSoup(html, 'html.parser')
    produtos_encontrados = []

    # 1. Encontra todas as caixas de produtos pela classe "produto"
    caixas = soup.find_all('div', class_='produto')

    for caixa in caixas:
        # 2. Use caixa.find() para pegar o texto dentro da tag h1 (classe 'titulo-produto')
        nome = caixa.find('h1', class_='titulo-produto').text
        preco = caixa.find('span',class_='preco-valor').text
        produtos_encontrados.append({"nome": nome, "preco": float(preco)})

    return produtos_encontrados


print(raspar_dados(html_da_pagina))
