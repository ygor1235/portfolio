import numpy as np

# Esses são os dados que o seu robô de scraping coletou (já convertidos para float)
dados_raspados = [
    {"nome": "Placa de Vídeo RTX 4060", "preco": 2100.00},
    {"nome": "Processador Ryzen 7", "preco": 1850.00},
    {"nome": "Memória RAM 16GB", "preco": 350.00},
    {"nome": "SSD 1TB", "preco": 400.00}
]


class InteligenciaDePrecos:
    def __init__(self, produtos):
        self.produtos = produtos
        # 1. Cria uma lista contendo apenas os valores numéricos dos preços
        self.precos = [p["preco"] for p in produtos]
        # 2. Calcula a média de todos os preços usando numpy
        self.media_mercado = np.mean(self.precos)

    def classificar_oportunidades(self):
        relatorio = []

        for p in self.produtos:
            nome = p["nome"]
            preco = p["preco"]

            # Regra de classificação da IA:
            # Se o preço for menor que 80% da média do mercado -> "Super Promoção"
            # Se for maior que 120% da média do mercado -> "Caro"
            # Qualquer outra coisa -> "Preço Normal"

            if preco < (self.media_mercado * 0.8):
                status = "Super Promocao"
            elif preco > (self.media_mercado * 1.2):
                status = "Caro"
            else:
                status = "Preco Normal"

            relatorio.append({"nome": nome, "preco": preco, "status": status})

        return relatorio


# Executando a inteligência de mercado
ia_precos = InteligenciaDePrecos(dados_raspados)
print("Média do Mercado:", ia_precos.media_mercado)
print(ia_precos.classificar_oportunidades())
