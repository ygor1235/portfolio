texto_bruto = "A Inteligência Artificial é incrível, o PPO funcionou de primeira!"


def limpar_texto(texto):
    texto_minusculo = texto.lower()
    textosemsinais = texto_minusculo.replace(",","").replace("!","")
    split = textosemsinais.split(" ")
    print(split)


limpar_texto(texto_bruto)

