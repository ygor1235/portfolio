import os

# Lista de arquivos que chegaram bagunçados
arquivos_recebidos = [
    "relatorio_financeiro.pdf",
    "dados_treinamento.csv",
    "foto_usuario.png",
    "fatura_setembro.pdf",
    "modelo_ia.h5"
]


def organizar_arquivos(lista):
    # O objetivo é retornar um dicionário onde a chave é o tipo e o valor é a lista de arquivos
    pastas = {
        "Documentos": [],
        "Dados": [],
        "Imagens": [],
        "Modelos_IA": []
    }

    for arquivo in lista:
       if arquivo.endswith('pdf'):
           pastas["Documentos"].append(arquivo)
       elif arquivo.endswith("csv"):
           pastas["Dados"].append(arquivo)
       elif arquivo.endswith('h5'):
           pastas["Modelos_IA"].append(arquivo)
       elif arquivo.endswith("png"):
           pastas["Imagens"].append(arquivo)

    return pastas


# Testando a automação
print(organizar_arquivos(arquivos_recebidos))
