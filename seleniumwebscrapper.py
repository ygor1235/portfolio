from selenium import webdriver
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager  # Facilita a gestão do driver
from bs4 import BeautifulSoup
import time
import numpy as np
dicionariodeitens = {"titulo":[],"preco":[],"link":[]}
dicionariovalor = {"caro":{"titulo":[],"preco":[],"link":[]},"mediano":{"titulo":[],"preco":[],"link":[]},"promoçao":{"titulo":[],"preco":[],"link":[]}}
# 1. Configuração do Selenium para evitar detecção
chrome_options = Options()
# Descomente a linha abaixo se quiser que o navegador rode em segundo plano (oculto)
# chrome_options.add_argument("--headless")

# Desativa mensagens de log desnecessárias do Chrome
chrome_options.add_argument("--log-level=3")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")

# Inicializa o driver automaticamente
servico = Service(ChromeDriverManager().install())
navegador = webdriver.Chrome(service=servico, options=chrome_options)

url = "https://lista.mercadolivre.com.br/rtx-5060#D[A:rtx%205060,L:undefined]&origin=UNKNOWN&as.comp_t=SUG&as.comp_v=rtx&as.comp_id=HIS"

try:
    print("Abrindo o navegador e carregando a página...")
    navegador.get(url)

    # Aguarda alguns segundos para o JavaScript descriptografar e montar a página
    time.sleep(5)

    # 2. Captura o HTML completamente renderizado pelo navegador
    html_da_pagina = navegador.page_source

    # 3. Passa o HTML limpo para o BeautifulSoup trabalhar
    sopa = BeautifulSoup(html_da_pagina, "html.parser")

    # Busca todos os blocos de produtos (o container principal de cada item)
    # Nota: As classes do Mercado Livre mudam raramente, mas esta estrutura é a atual para listas
    itens = sopa.find_all("li", class_="ui-search-layout__item")

    print(f"\n--- Produtos Encontrados ({len(itens)}) ---")
    for item in itens:
        # Extrai o título do produto

        titulo_elemento = item.find("h3", class_="poly-component__title-wrapper")
        titulo = titulo_elemento.text.strip() if titulo_elemento else "Título não encontrado"

        # Extrai o preço (buscando a classe da fração inteira do valor)
        preco_elemento = item.find("span", class_="andes-money-amount__fraction")
        preco = preco_elemento.text.strip() if preco_elemento else "Sob consulta"

        # Extrai o link do produto
        link_elemento = item.find("a", class_="poly-component__title")
        link = link_elemento["href"] if link_elemento else "Link não encontrado"
        dicionariodeitens["titulo"].append(titulo)
        dicionariodeitens["preco"].append(float(preco.replace(',','').replace('.','')))
        dicionariodeitens["link"].append(link)
    mediavalor = np.mean(dicionariodeitens["preco"])
    for i,preco in enumerate(dicionariodeitens["preco"]):
        if preco > mediavalor:
            dicionariovalor["caro"]["titulo"].append(dicionariodeitens["titulo"][i])
            dicionariovalor["caro"]["preco"].append(dicionariodeitens["preco"][i])
            dicionariovalor["caro"]["link"].append(dicionariodeitens["link"][i])
        elif preco == mediavalor:
            dicionariovalor["mediano"]["titulo"].append(dicionariodeitens["titulo"][i])
            dicionariovalor["mediano"]["preco"].append(dicionariodeitens["preco"][i])
            dicionariovalor["mediano"]["link"].append(dicionariodeitens["link"][i])
        elif preco < mediavalor:
            dicionariovalor["promoçao"]["titulo"].append(dicionariodeitens["titulo"][i])
            dicionariovalor["promoçao"]["preco"].append(dicionariodeitens["preco"][i])
            dicionariovalor["promoçao"]["link"].append(dicionariodeitens["link"][i])
    resposta = input("qual quer ver? caro,mediano ou promoçao? escreva: ")
    for i in range(len(dicionariovalor[resposta]["preco"])):
        print(dicionariovalor[resposta]["titulo"][i],dicionariovalor[resposta]["preco"][i],dicionariovalor[resposta]["link"][i])
        print()
finally:
    # Garante que o navegador vai fechar mesmo se o código der erro no meio
    navegador.quit()
    print("Navegador finalizado.")
