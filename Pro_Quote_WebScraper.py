import logging
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",filename="log livros"
)
class QuotesScraper:
    def __init__(self,url:str):
        self.url = url
        self.urlcompleta = None
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
    def _extractsoup(self,url:str):
        try:
            resposta = self.session.get(url, timeout=10)
            resposta.raise_for_status()  # Dispara erro se o status for 4xx ou 5xx
            return BeautifulSoup(resposta.content, "lxml")
        except requests.RequestException as e:
            logging.error(f"Erro ao acessar {url}: {e}")
            return None
    def _passarpagina(self,sopa):
        parte1 = sopa.find("li",class_="next")
        parte2 = parte1.find("a")
        proximapagina1 = parte2.get("href")
        self.urlcompleta = urljoin(self.url,proximapagina1)
        logging.debug(f"passou de pagina para a pagina{self.urlcompleta}")

    def findinfo(self,num_page = 1):
        try:
            dadosparasalvar = []
            for i in range(num_page):
                if self.urlcompleta:
                    sopa = self._extractsoup(self.urlcompleta)
                else:
                    sopa = self._extractsoup(self.url)

                capsulainfo = sopa.find_all('div',class_="quote")


                for info in capsulainfo:
                    tagslist = info.find_all('meta',class_="keywords")
                    lista = [tag.get("content") for tag in tagslist if tag]
                    listaquotes = info.find_all('span',class_="text")
                    findauthor1 = info.find_all('small',class_='author')
                    autor = [au.text for au in findauthor1]
                    dadosparasalvar.append({"autor":autor,"quote":[quote.text for quote in listaquotes if quote],"tags":lista})
                self._passarpagina(sopa)
            return dadosparasalvar

        except Exception as e:
            logging.warning(f"Erro ao processar os dados desta frase específica: {e}")


if __name__ == "__main__":
    urlalvo = 'https://quotes.toscrape.com/'

    scraper = QuotesScraper(urlalvo)
    dadoscoletados = scraper.findinfo(3)
    df = pd.DataFrame(dadoscoletados).to_csv("quotes.csv",index=False,encoding="utf-8-sig")
else:
    logging.error("Nenhum dado foi coletado.")

