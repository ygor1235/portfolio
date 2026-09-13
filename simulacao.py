from rinhadoscara.Teste_rede import  Networkv3,PPOAgent,compute_gae
import pygame
import sys
import math
import random
import numpy as np
import pickle
from copy import deepcopy
import copy
import os
from collections import defaultdict
import threading
from estado import estadoserver,modelo_jogador
import socket
import json
import time
lockindividual = threading.Lock()
lockpopu = threading.Lock()
treinamento_rodando = False
circulos = []
jogadores = estadoserver["jogadores"]
armas_no_chao  = []
def dropar_arma(agente):
    arma = agente.get("arma")

    if arma is None:
        return

    armas_no_chao.append({
        "arma": arma,
        "pos": [
            agente["pos"][0],
            agente["pos"][1]
        ],
        "tempo": 600
    })
    agente['alcance'] = 100
    agente['dano'] = 10
    agente['recompensa_hit'] = 2
    agente["arma"] = None
def desenhar_tela_jogo(jogador):


    pygame.draw.circle(
        tela,
        jogador["cor"],
        jogador["pos"],
        jogador["tamanho"]
    )
def escolher_acao_com_planning(rede, z_t, outputs, n_steps=2, gamma=0.99):
    import numpy as np

    batch_size = z_t.shape[0]
    num_heads = len(outputs)

    actions_final = []

    for i in range(batch_size):

        z0 = z_t[i:i + 1]

        best_action_heads = []

        # 🔥 para cada head decide separadamente
        for h in range(num_heads):

            num_actions = outputs[h].shape[1]

            best_score = -1e9
            best_a = 0

            for a in range(num_actions):

                # =========================
                # monta ação one-hot COMPLETA
                # =========================
                action_vec_parts = []

                for h2 in range(num_heads):
                    probs = outputs[h2]

                    onehot = np.zeros((1, probs.shape[1]))

                    if h2 == h:
                        onehot[0, a] = 1.0
                    else:
                        # usa ação mais provável pras outras heads
                        best_other = np.argmax(probs[i])
                        onehot[0, best_other] = 1.0

                    action_vec_parts.append(onehot)

                action_vec = np.concatenate(action_vec_parts, axis=1)

                # =========================
                # rollout no latent
                # =========================
                z_sim = z0.copy()
                total_value = 0.0
                discount = 1.0

                for step in range(n_steps):

                    z_sim = rede.predict_next_latent_batch(z_sim, action_vec)

                    # ⚠️ NÃO usa forward completo
                    x_sim = z_sim
                    for layer in rede.layers:
                        x_sim = layer.activate_batch(x_sim)

                    v = rede.val_head.activate_batch_linear(x_sim)[0, 0]

                    total_value += discount * v
                    discount *= gamma

                if total_value > best_score:
                    best_score = total_value
                    best_a = a

            best_action_heads.append(best_a)

        actions_final.append(best_action_heads)
    if batch_size == 1:
        return actions_final[0]
# ---------------------------------------
# Função de loss MSE
# ---------------------------------------
def mse_loss(preds, targets):
    return np.mean([(p - t) ** 2 for p, t in zip(preds, targets)])


hitboxes_ataque = []
ANGULO_VISAO = math.radians(120)  # 120° → 2.094 rad
DISTANCIA_VISAO = 700  # até onde o agente enxerga


def spawnaleatorioobs(numerodeobstaculos=4):
    obstaculos = []

    while len(obstaculos) < numerodeobstaculos:
        tipo = "madeira" if len(obstaculos) % 2 == 0 else "pedra"

        largura = 50 if tipo == "madeira" else 100
        altura = 200 if tipo == "madeira" else 50

        x = random.randint(0, LARGURA - largura)
        y = random.randint(0, ALTURA - altura)

        novo_rect = pygame.Rect(x, y, largura, altura)

        # evita sobreposição entre obstáculos
        colidiu = False
        for obs in obstaculos:
            if novo_rect.colliderect(obs["rect"]):
                colidiu = True
                break

        if colidiu:
            continue

        obstaculos.append({
            "rect": novo_rect,
            "tipo": tipo,
            "qtd": 1,
            "cooldown": 0,
            "health": 50
        })

    return obstaculos
def debug_estado(estado, agente,limite=1.0):
    estado = np.array(estado)

    encontrou = False

    for b in range(estado.shape[0]):  # batch
        for i, v in enumerate(estado[b]):
            if v > limite or v < -limite:
                print(estado)
                print(f"batch {b}, indice {i}, valor: {v},agente:{agente["nome"]}")
                encontrou = True




def lidar(agente, obstaculos, rays=14):
    ax, ay = agente["pos"]
    base = agente["direcao"]
    half = agente["FOV"] / 2

    sensores = []

    for i in range(rays):
        ang = base - half + (i / (rays - 1)) * agente["FOV"]

        for d in range(0, agente["alcance_visao"], 5):
            x = ax + math.cos(ang) * d
            y = ay + math.sin(ang) * d

            # parede do mapa
            if x <= 0 or x >= LARGURA or y <= 0 or y >= ALTURA:
                sensores.append(d / agente["alcance_visao"])
                break

            # obstáculo
            bateu = False
            for obs in obstaculos:
                if obs["rect"].collidepoint(x, y):
                    sensores.append(d / agente["alcance_visao"])
                    bateu = True
                    break

            if bateu:
                break

        else:
            sensores.append(1.0)

    return sensores


def colide_circulo_rect(cx, cy, raio, rect):
    # ponto mais próximo do círculo dentro do retângulo
    mais_prox_x = max(rect.left, min(cx, rect.right))
    mais_prox_y = max(rect.top, min(cy, rect.bottom))

    dx = cx - mais_prox_x
    dy = cy - mais_prox_y

    return (dx * dx + dy * dy) < (raio * raio)


def colidiu_com_obstaculos(agente, obstaculos):
    cx, cy = agente["pos"]
    raio = agente["tamanho"]

    for obs in obstaculos:
        if colide_circulo_rect(cx, cy, raio, obs["rect"]):
            return True
    return False


def raycast_visao(agente, obstaculos, angulo, max_dist=DISTANCIA_VISAO, step=4):
    cx, cy = agente["pos"]

    for d in range(0, max_dist, step):
        x = cx + math.cos(angulo) * d
        y = cy + math.sin(angulo) * d

        # fora da tela = bloqueio
        if x < 0 or x > LARGURA or y < 0 or y > ALTURA:
            return x, y

        # colisão com obstáculo
        for obs in obstaculos:
            if not isinstance(obs["rect"], pygame.Rect):
                raise TypeError(f"Esperado pygame.Rect, veio {type(obs)} → {obs}")

            if obs["rect"].left <= x <= obs["rect"].right and obs["rect"].top <= y <= obs["rect"].bottom:
                return x, y

    return (
        cx + math.cos(angulo) * max_dist,
        cy + math.sin(angulo) * max_dist
    )


def desenhar_campo_visao(tela, agente, obstaculos, cor=(0, 255, 0)):
    x, y = agente["pos"]
    direcao = agente["direcao"]

    ang_inicio = direcao - ANGULO_VISAO / 2
    ang_fim = direcao + ANGULO_VISAO / 2

    pontos = [(x, y)]

    passos = 40  # ↑ mais suave e mais preciso
    for i in range(passos + 1):
        ang = ang_inicio + (ang_fim - ang_inicio) * i / passos
        px, py = raycast_visao(agente, obstaculos, ang)
        pontos.append((px, py))

    # superfície transparente
    s = pygame.Surface((LARGURA, ALTURA), pygame.SRCALPHA)
    pygame.draw.polygon(s, (*cor, 70), pontos)
    tela.blit(s, (0, 0))

    # linha da direção central (até o primeiro bloqueio)
    fx, fy = raycast_visao(agente, obstaculos, direcao)
    pygame.draw.line(tela, cor, (x, y), (fx, fy), 2)


def lidar_debug(agente, inimigos, obstaculos, rays=14):
    ax, ay = agente["pos"]
    base = agente["direcao"]
    half = agente["FOV"] / 2

    resultados = []  # (x, y, tipo)

    for i in range(rays):
        ang = base - half + (i / (rays - 1)) * agente["FOV"]

        for d in range(0, agente["alcance_visao"], 5):
            x = ax + math.cos(ang) * d
            y = ay + math.sin(ang) * d

            # ================= PAREDE DO MAPA =================
            if x <= 0 or x >= LARGURA or y <= 0 or y >= ALTURA:
                resultados.append((x, y, "parede"))
                break

            # ================= OBSTÁCULOS =================
            bateu = False
            for obs in obstaculos:
                if obs["rect"].collidepoint(x, y):
                    resultados.append((x, y, "parede"))
                    bateu = True
                    break
            if bateu:
                break

            # ================= INIMIGOS =================
            for inimigo in inimigos:
                if inimigo["vida"] <= 0:
                    continue

                dx = x - inimigo["pos"][0]
                dy = y - inimigo["pos"][1]
                dist = math.hypot(dx, dy)

                if dist <= inimigo["tamanho"]:
                    resultados.append((x, y, "inimigo"))
                    bateu = True
                    break
            if bateu:
                break

        else:
            resultados.append((x, y, "livre"))

    return resultados


def desenhar_campo_visao_lidar(tela, agente, agentes, obstaculos, cor=(0, 255, 0), rays=14):
    x, y = agente["pos"]

    # todos os outros agentes são alvos possíveis
    inimigos = [a for a in agentes if a is not agente and a["vida"] > 0]

    dados = lidar_debug(agente, inimigos, obstaculos, rays)

    pontos = [(px, py) for px, py, _ in dados]

    # ================= POLÍGONO =================
    s = pygame.Surface((LARGURA, ALTURA), pygame.SRCALPHA)
    pygame.draw.polygon(s, (*cor, 40), [(x, y)] + pontos)
    tela.blit(s, (0, 0))

    # ================= RAIOS =================
    for px, py, tipo in dados:
        pygame.draw.line(tela, cor, (x, y), (px, py), 1)

        if tipo == "parede":
            cor_ponto = (255, 0, 0)
        elif tipo == "inimigo":
            cor_ponto = (0, 150, 255)
        else:
            cor_ponto = (0, 255, 0)

        pygame.draw.circle(tela, cor_ponto, (int(px), int(py)), 3)


def desenhar_cone_ataque(tela, agente, cor=(255, 0, 0)):
    # 🔥 Só desenha se estiver atacando
    if agente["atacando"] == False:
        return

    x, y = agente["pos"]
    direcao = agente["direcao"]
    alcance = agente["alcance"]

    ANGULO_ATAQUE = math.pi / 3  # 60 graus

    ang_inicio = direcao - ANGULO_ATAQUE / 2
    ang_fim = direcao + ANGULO_ATAQUE / 2

    pontos = [(x, y)]

    passos = 15
    for i in range(passos + 1):
        ang = ang_inicio + (ang_fim - ang_inicio) * i / passos
        px = x + math.cos(ang) * alcance
        py = y + math.sin(ang) * alcance
        pontos.append((px, py))

    s = pygame.Surface((LARGURA, ALTURA), pygame.SRCALPHA)
    pygame.draw.polygon(s, (*cor, 80), pontos)
    tela.blit(s, (0, 0))


pygame.init()


def raycast_parede(agente, angulo, max_dist=1000, step=4):
    cx, cy = agente["pos"]

    for d in range(0, max_dist, step):
        x = cx + math.cos(angulo) * d
        y = cy + math.sin(angulo) * d

        # bateu na borda do mapa
        if x <= 0 or x >= LARGURA or y <= 0 or y >= ALTURA:
            return x, y

    # não achou borda (teoricamente impossível)
    return (
        cx + math.cos(angulo) * max_dist,
        cy + math.sin(angulo) * max_dist
    )


# Tamanho da tela
LARGURA = 800
ALTURA = 600
tela = pygame.display.set_mode((LARGURA, ALTURA))
pygame.display.set_caption("Jogo de Luta com IA")
obstaculos = spawnaleatorioobs(4)
# Cores
BRANCO = (255, 255, 255)
verde = (34, 139, 34)
VERMELHO = (255, 0, 0)
AZUL = (0, 0, 255)
AMARELO = (255, 255, 0)
listargb = []
imagem = pygame.image.load("soco.png").convert_alpha()
imagembastao = pygame.image.load("bastao.png").convert_alpha()
imagemcartola = pygame.image.load("cartola1.png").convert_alpha()
imagemcoroa = pygame.image.load('coroa_do_criador.png').convert_alpha()
imagemalabarda = pygame.image.load("alabarda.png").convert_alpha()
imagelanca = pygame.image.load("lancadropada.png").convert_alpha()
imagempistola = pygame.image.load("pistoladropada.png").convert_alpha()
imagemkatana = pygame.image.load('katana.png').convert_alpha()
def criar_repulsao(agente):
    pontos = agente['pontos']

    return {
        "tipo": "repulsao",
        "pos": agente['pos'],
        "raio": 0,
        "velocidade": 4,
        "max_raio": 200 + (pontos * 10)
    }
def criar_convergencia_portal(populacao):
    agente = obter_agente_amarelo(populacao)
    pontos = agente['pontos']

    max_raio = 200

    return {
        "tipo": "convergencia_portal",
        "pos": agente['pos'][:],  # 🔥 cópia (evita bug de referência)
        "raio": max_raio,
        "velocidade": 3,
        "max_raio": max_raio,

        "dono": "amarelo",

        # 🔫 sistema de tiro
        "balas": 2 + pontos,
        "cooldown": 0,
        "delay_tiro": 10
    }
def criar_convergencia(agente):
    pontos = agente['pontos']

    max_raio = 200 + (pontos * 10)

    return {
        "tipo": "convergencia",
        "pos": agente['pos'],
        "raio": max_raio,   # começa grande
        "velocidade": 3,
        "max_raio": max_raio
    }
def atualizar_efeitos(lista):
    global populacao
    dono = obter_agente_amarelo(populacao)
    for e in lista[:]:
        if e["tipo"] == "repulsao":
            e["raio"] += e["velocidade"]

            if e["raio"] >= e["max_raio"]:
                lista.remove(e)

        elif e["tipo"] == "convergencia":
            e["raio"] -= e["velocidade"]

            if e["raio"] <= 0:
                lista.remove(e)
        elif e["tipo"] == "convergencia_portal":
                e["raio"] -= e["velocidade"]

                if e["raio"] <= 0:
                    lista.remove(e)
                    continue

                # acabou munição
                if e["balas"] <= 0:
                    lista.remove(e)
                    continue

                # cooldown
                if e["cooldown"] > 0:
                    e["cooldown"] -= 1
                    continue



                # achar alvo (ignora dono)
                alvo = None
                menor_dist = float("inf")

                for a in populacao:
                    if a.get("nome") == e["dono"]:
                        continue

                    dx = a["pos"][0] - e["pos"][0]
                    dy = a["pos"][1] - e["pos"][1]
                    dist = dx * dx + dy * dy

                    if dist < menor_dist:
                        menor_dist = dist
                        alvo = a

                if alvo:
                    px, py = e["pos"]
                    ax, ay = alvo["pos"]

                    dx = ax - px
                    dy = ay - py
                    angulo = math.atan2(dy, dx)

                    # 🔥 dispara 2 tiros
                    disparar_projetil_angulo(e["pos"], angulo - 0.15, dono=dono)
                    disparar_projetil_angulo(e["pos"], angulo + 0.15, dono=dono)

                    e["balas"] -= 1  # 🔥 consome munição

                e["cooldown"] = e["delay_tiro"]
def aplicar_distorcao(screen, efeitos, intensidade=50, passo=8):
    largura, altura = screen.get_size()

    for e in efeitos:
        centro = e["pos"]
        raio = e["raio"]

        if raio <= 0:
            continue

        for x in range(0, largura, passo):
            for y in range(0, altura, passo):
                dx = x - centro[0]
                dy = y - centro[1]

                dist = (dx * dx + dy * dy) ** 0.5

                if dist > raio:
                    continue

                dist += 0.0001

                falloff = 1 - (dist / raio)

                if e["tipo"] == "repulsao":
                    fator = (intensidade * falloff) / dist
                else:  # convergencia
                    fator = -(intensidade * falloff) / dist

                src_x = int(x + dx * fator)
                src_y = int(y + dy * fator)

                rect = pygame.Rect(x, y, passo, passo)

                if 0 <= src_x < largura and 0 <= src_y < altura:
                    bloco = screen.subsurface(rect).copy()
                    screen.blit(bloco, (src_x, src_y))
def desenhar_cosmeticos(tela, populacao):
    # CARTOLA
    cartola1 = pygame.transform.scale(imagemcartola, (60, 60))
    cartola1 = pygame.transform.rotate(cartola1, 100)

    # COROA (usa imagem própria!)
    coroa = pygame.transform.scale(imagemcoroa, (60, 60))
    coroa = pygame.transform.rotate(coroa, 100)  # geralmente não precisa inclinar

    for agente in populacao:
        cosmetico = agente.get("cosmetico")

        if cosmetico in ["cartola1", "coroa"]:

            direcao = agente["direcao"]
            direcao_oposta = direcao + math.pi

            distancia = 40

            offset_x = math.cos(direcao_oposta) * distancia
            offset_y = math.sin(direcao_oposta) * distancia

            x = agente["pos"][0] + offset_x
            y = agente["pos"][1] + offset_y

            angulo = -math.degrees(direcao)

            if cosmetico == "cartola1":
                sprite = pygame.transform.rotate(cartola1, angulo)

            elif cosmetico == "coroa":
                sprite = pygame.transform.rotate(coroa, angulo)

            rect = sprite.get_rect(center=(x, y))
            tela.blit(sprite, rect)

def distancia(a, b):
    dx = a["pos"][0] - b["pos"][0]
    dy = a["pos"][1] - b["pos"][1]
    return math.hypot(dx, dy)


def spawnaleatorio(numerodeagentes=3, tamanho=30):
    listapos = []

    while len(listapos) < numerodeagentes:
        x = random.randint(tamanho, LARGURA - tamanho)
        y = random.randint(tamanho, ALTURA - tamanho)

        agente_fake = {"pos": [x, y], "tamanho": tamanho}

        # testa obstáculo
        if obstaculos and colidiu_com_obstaculos(agente_fake, obstaculos):
            continue

        # testa colisão com outros agentes
        colidiu = False
        for px, py in listapos:
            dist = math.hypot(px - x, py - y)
            if dist < tamanho * 2:
                colidiu = True
                break

        if colidiu:
            continue

        listapos.append([x, y])

    return listapos


spawn1, spawn2, spawn3 = spawnaleatorio(3)
outputs_config1 = (13,"softmax")
outputs_configazul = (12,"softmax")
outputs_config2 = (12,"softmax")
# Dicionários de atributos dos agentes
agente1 = {"confuso":0,"acao_pura_ppo":0,"morto":False,
           "treinando":False,
           "progresso_treino":0,
"paralisado":False,
    'paralisia_timer':0,
    "nome": "vermelho",
    "cosmetico": None,
    'distortioncd':0,
    "rede": PPOAgent(state_dim=213, hidden_sizes=[120, 120, 120, 120, 120, 120, 120, 120, 120, 120],
                                     action_dim=13),

    "target_rede": None,
    "cor": VERMELHO,
    "pos": spawn1,
    "velocidade": 5,
    'velocidade_max': 5,
"vel": [0.0, 0.0],      # velocidade atual (ESSENCIAL)
"acc": 1,             # aceleração por frame
"max_speed": 4,         # velocidade máxima (pode usar velocidade_max)
"friction": 0.85,       # atrito (quanto desacelera)
    "dano": 10,
    "tamanho": 30,
    "vida": 100,
    "vida_max": 100,
    "regeneracao": 0.02,
    "stamina": 100,
    "stamina_max": 100,
    "atacando": False,
    "cooldown_ataque": 0,
    "alcance": 100,
    "custo_ataque": 15,
    "dist_prev": None,
    "emote": None,  # emote atual
    "tempo_emote": 0,  # duração do emote
    "mensagem": None,  # vetor enviado para outros
    "arma": "lanca",
    "inventario": {},
    "stun": 0,
    "historico_loss": [],
    "inventariomax": 3,

    "pontos": 0,
    "recompensa_hit": 2,
    "memoria": [],
    "memoria_max": 1800,
    "tempo_ataque": 0,
    "DURACAO_ATAQUE": 30,  # 30 frames ≈ 0.5 segundos
    "recompensa_frame": 0,
    "recompensa_temp": 0,

    "direcao": 0.0,  # ângulo em radianos (0 = direita)
    "FOV": math.radians(120),  # campo de visão de 120 graus
    "alcance_visao": 700,  # distância máxima que enxerga
    "velocidade_rotacao": 0.12,  # quanto gira por frame
    "ultimos_estados": [],

}
agente2 = {"confuso":0,"treinando":False,"progresso_treino":0,"acao_pura_ppo":0,"morto":False,
    "nome": "azul",
    "cosmetico": 'coroa',

    "rede":  PPOAgent(state_dim=213, hidden_sizes=[120, 120, 120, 120, 120, 120, 120, 120, 120, 120],
                                     action_dim=12),
    "target_rede": None,
    "cor": AZUL,
    "pos": spawn2,
    "velocidade": 4,
    'velocidade_max': 4,
"vel": [0.0, 0.0],      # velocidade atual (ESSENCIAL)
"acc": 0.5,             # aceleração por frame
"max_speed": 5,         # velocidade máxima (pode usar velocidade_max)
"friction": 0.85,       # atrito (quanto desacelera)
    "dano": 12,
"coroa_timer":0,
    "tamanho": 30,
    "vida": 100,
    "vida_max": 100,
    "regeneracao": 0.01,
    "stamina": 100,
    "stamina_max": 100, "bloqueando": False,
    "historico_loss": [],
    "emote": None,  # emote atual
    "tempo_emote": 0,  # duração do emote
    "mensagem": None,  # vetor enviado para outros
    "arma": None,
    "inventario": {},
    "stun": 0,
    "recompensa_frame": 0,
    "recompensa_temp": 0,
    "inventariomax": 3,
    "dist_prev": None,
    "atacando": False,
    "cooldown_ataque": 0,
    "alcance": 100,
    "custo_ataque": 15,
    "pontos": 0,
    "recompensa_hit": 2,
    "memoria": [],
    "memoria_max": 1800,
    "tempo_ataque": 0,
    "DURACAO_ATAQUE": 30,  # 30 frames ≈ 0.5 segundos
    "direcao": 0.0,  # ângulo em radianos (0 = direita)
    "FOV": math.radians(120),  # campo de visão de 120 graus
    "alcance_visao": 700,  # distância máxima que enxerga
    "velocidade_rotacao": 0.12,  # quanto gira por frame
    "ultimos_estados": [],
"log_prob": None,
"value": None,
"paralisia_cd":0,
"paralisado":False
}
agente3 = {"confuso":0,"treinando":False,"progresso_treino":0,"acao_pura_ppo":0,"morto":False,
    "counter_cd": 0 ,
"counter_janela": 0 ,
    "paralisado":False,
    'paralisia_timer':0,
    "nome": "amarelo",
    "rede": PPOAgent(state_dim=216, hidden_sizes=[120, 120, 120, 120, 120, 120, 120, 120, 120, 120],
                                     action_dim=12),
    "target_rede": None,
    "cor": AMARELO,
    "pos": spawn3,
    "balas":6,
    'max_balas':6,
    'recarregando':False,
    'reload_timer':0,
    "velocidade": 4.5,
    'velocidade_max': 4.5,
"vel": [0.0, 0.0],      # velocidade atual (ESSENCIAL)
"acc": 0.5,             # aceleração por frame
"max_speed": 4,         # velocidade máxima (pode usar velocidade_max)
"friction": 0.85,       # atrito (quanto desacelera)
    "cosmetico":"cartola1",
    "dano": 12,
    "tamanho": 30,
    "vida": 100,
    "vida_max": 100,
    "regeneracao": 0.01,
    "stamina": 100,
    "stamina_max": 100, "bloqueando": False,
    "historico_loss": [],
    "emote": None,  # emote atual
    "tempo_emote": 0,  # duração do emote
    "mensagem": None,  # vetor enviado para outros
    "arma": "pistola",
    "inventario": {},
    "stun": 0,
    "recompensa_frame": 0,
    "recompensa_temp": 0,
    "inventariomax": 3,
    "dist_prev": None,
    "atacando": False,
    "cooldown_ataque": 0,
    "alcance": 100,
    "custo_ataque": 0,
    "pontos": 0,
    "recompensa_hit": 2,
    "memoria": [],
    "memoria_max": 1800,
    "tempo_ataque": 0,
    "DURACAO_ATAQUE": 60,  # 30 frames ≈ 0.5 segundos
    "direcao": 0.0,  # ângulo em radianos (0 = direita)
    "FOV": math.radians(120),  # campo de visão de 120 graus
    "alcance_visao": 700,  # distância máxima que enxerga
    "velocidade_rotacao": 0.12,  # quanto gira por frame
    "ultimos_estados": [],
"log_prob": None,
"value": None,
}
populacao = []
populacao.append(agente1)
populacao.append(agente2)
populacao.append(agente3)
historico = [[] for _ in populacao]
TIPOS_RECURSO = ["madeira", "pedra"]


def traduzir_acao_ppo(acao_unica, agente):
    """
    Traduz a ação discreta do PPO (0 a 12) na tupla antiga (mover, ataque, girar)

    mover:   0 = nada, 1 = cima, 2 = direita, 3 = esquerda, 4 = baixo
    girar:   0 = nada, 1 = esquerda, 2 = direita
    ataque:  0 = nada, 1 = Ataque/Tiro, 2 = Craft/Recarrega, 3 = Skill 1 (Distorção), 4 = Skill 2 (Repulsão)
    """
    mover = 0
    girar = 0
    ataque = 0

    # =========================================================================
    # MAPEAMENTO DOS 13 COMANDOS (0 a 12)
    # =========================================================================

    # 0 a 4: Movimentação pura (Incluindo ficar totalmente parado)
    if acao_unica == 0:
        mover = 0  # Ficar Parado
    elif acao_unica == 1:
        mover = 1  # Mover para Cima
    elif acao_unica == 2:
        mover = 2  # Mover para Direita
    elif acao_unica == 3:
        mover = 3  # Mover para Esquerda
    elif acao_unica == 4:
        mover = 4  # Mover para Baixo

    # 5 a 6: Rotação pura da visão
    elif acao_unica == 5:
        girar = 1  # Girar para a Esquerda
    elif acao_unica == 6:
        girar = 2  # Girar para a Direita

    # 7: Comando básico de ataque/disparo parado
    elif acao_unica == 7:
        ataque = 1  # Ataca/Atira onde está olhando

    # 8 a 9: Ataques em movimento (Avançando ou Recuando) para dar fluidez ao combate
    elif acao_unica == 8:
        mover = 1; ataque = 1  # Anda para cima atacando
    elif acao_unica == 9:
        mover = 4; ataque = 1  # Anda para baixo atacando (Kiting)

    # 10: Gerenciamento de Recursos / Utilidade
    elif acao_unica == 10:
        ataque = 2  # Tenta fazer Crafting ou Recarregar Arma

    # 11 a 12: Habilidades Especiais Ativas (Distorção, Repulsão, Coroa)
    elif acao_unica == 11:
        ataque = 3  # Usa a Skill 1 (Distorção / Círculo da Coroa)
    elif acao_unica == 12:
        ataque = 4  # Usa a Skill 2 (Repulsão)

    return mover, ataque, girar


def girar_visao_por_acao(agente, acao):
    """
    acao:
    0 = não gira
    1 = gira para esquerda
    2 = gira para direita
    """

    CUSTO_GIRO = 0.05  # custo por frame de rotação

    if acao == 1:  # gira esquerda
        if agente["stamina"] > CUSTO_GIRO:
            agente["direcao"] -= agente["velocidade_rotacao"]
            agente["stamina"] -= CUSTO_GIRO

    elif acao == 2:  # gira direita
        if agente["stamina"] > CUSTO_GIRO:
            agente["direcao"] += agente["velocidade_rotacao"]
            agente["stamina"] -= CUSTO_GIRO

    # Mantém ângulo sempre entre -pi e pi
    if agente["direcao"] > math.pi:
        agente["direcao"] -= 2 * math.pi
    elif agente["direcao"] < -math.pi:
        agente["direcao"] += 2 * math.pi


def nome_arquivo_agente(agente):
    r, g, b = agente["cor"]
    return f"rede_{r}_{g}_{b}.pkl"


def salvar_rede(agente, nome_arquivo):
    # 1. Cria um nome temporário na mesma pasta (ex: .nome_arquivo.tmp)
    diretorio = os.path.dirname(nome_arquivo) or "."
    nome_base = os.path.basename(nome_arquivo)
    nome_temp = os.path.join(diretorio, f".{nome_base}.tmp")

    try:
        # 2. Grava os dados no arquivo temporário
        with open(nome_temp, "wb") as f:
            pickle.dump(agente["rede"], f)
            # Força o Windows a gravar fisicamente no disco AGORA, tirando da memória RAM
            f.flush()
            os.fsync(f.fileno())

        # 3. Substitui o arquivo antigo pelo novo de forma atômica
        # Se a tela azul der antes daqui, o arquivo antigo continua intacto!
        if os.path.exists(nome_arquivo):
            os.replace(nome_temp, nome_arquivo)
        else:
            os.rename(nome_temp, nome_arquivo)

        print(f"Rede salva com total segurança em {nome_arquivo}")

    except Exception as e:
        # Se faltar memória no meio do processo, limpa o arquivo temporário corrompido
        if os.path.exists(nome_temp):
            os.remove(nome_temp)
        print(f"Erro ao salvar a rede (prevenido corrupção): {e}")


def carregar_rede(agente, nome_arquivo):
    if os.path.exists(nome_arquivo):
        with open(nome_arquivo, "rb") as f:
            agente["rede"] = pickle.load(f)
        print(f"Rede carregada de {nome_arquivo}")
    else:
        print(f"Nenhuma rede encontrada em {nome_arquivo}, começando do zero.")


for agente in populacao:
    arquivo = nome_arquivo_agente(agente)
    carregar_rede(agente, arquivo)


def inimigo_visivel(agente, inimigo, obstaculos, retornar_angulo=False):
    ax, ay = agente["pos"]
    ix, iy = inimigo["pos"]

    dx = ix - ax
    dy = iy - ay
    dist = math.hypot(dx, dy)

    # fora do alcance
    if dist > agente["alcance_visao"]:
        return (False, 0.0) if retornar_angulo else False

    angulo_inimigo = math.atan2(dy, dx)

    # ângulo relativo [-pi, pi]
    diff = (angulo_inimigo - agente["direcao"] + math.pi) % (2 * math.pi) - math.pi

    # fora do FOV
    if abs(diff) > agente["FOV"] / 2:
        return (False, diff) if retornar_angulo else False

    # 🔥 OCLUSÃO USANDO O MESMO RAYCAST DO DEBUG
    px, py = raycast_visao(
        agente,
        obstaculos,
        angulo_inimigo,
        max_dist=agente["alcance_visao"]
    )

    dist_raycast = math.hypot(px - ax, py - ay)

    # se o raio bateu antes de alcançar o inimigo → bloqueado
    if dist_raycast + 1e-6 < dist:
        return (False, diff) if retornar_angulo else False

    return (True, diff) if retornar_angulo else True


def dentro_cone_ataque(atacante, alvo):
    dx = alvo["pos"][0] - atacante["pos"][0]
    dy = alvo["pos"][1] - atacante["pos"][1]
    dist = math.hypot(dx, dy)

    if dist > atacante["alcance"]:
        return False

    angulo_alvo = math.atan2(dy, dx)
    diff = (angulo_alvo - atacante["direcao"] + math.pi) % (2 * math.pi) - math.pi

    # Cone de ataque mais estreito (ex: 60 graus)
    return abs(diff) <= math.radians(30)


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def sensor_obstaculo(agente, obstaculos, angulo, max_dist=300, step=5):
    cx, cy = agente["pos"]
    raio = agente["tamanho"]

    for d in range(raio, max_dist, step):
        x = cx + math.cos(angulo) * d
        y = cy + math.sin(angulo) * d

        # 🚧 colisão com parede do mapa
        if x <= 0 or x >= LARGURA or y <= 0 or y >= ALTURA:
            return d / max_dist

        # 🧱 colisão com obstáculos
        for obs in obstaculos:
            if isinstance(obs, pygame.Rect) and obs.collidepoint(x, y):
                return d / max_dist

    return 1.0  # nada detectado


def sensores_obstaculos(agente, obstaculos):
    base = agente["direcao"]

    frente = sensor_obstaculo(agente, obstaculos, base)
    tras = sensor_obstaculo(agente, obstaculos, base + math.pi)
    esquerda = sensor_obstaculo(agente, obstaculos, base - math.pi / 2)
    direita = sensor_obstaculo(agente, obstaculos, base + math.pi / 2)

    return frente, tras, esquerda, direita


def relativo_ao_agente(dx, dy, direcao):
    cos_d = math.cos(-direcao)
    sin_d = math.sin(-direcao)

    rx = dx * cos_d - dy * sin_d
    ry = dx * sin_d + dy * cos_d

    return rx, ry


def sensor_distancia(agente, obstaculos, angulo):
    ax, ay = agente["pos"]
    px, py = raycast_visao(agente, obstaculos, angulo)

    dist = math.hypot(px - ax, py - ay)
    return dist / agente["alcance_visao"]


def lidar2(agente, obstaculos, rays=16):
    base = agente["direcao"]
    half = agente["FOV"] / 2

    sensores = []
    angulos = []

    for i in range(rays):
        ang = base - half + (i / (rays - 1)) * agente["FOV"]
        px, py = raycast_visao(agente, obstaculos, ang)
        sensores.append((px, py))
        angulos.append(ang)

    return sensores, angulos


def ultima_action_one_hot(agente, head_sizes=(5, 3, 3)):
    if len(agente["memoria"]) == 0:
        return np.zeros(sum(head_sizes), dtype=np.float32)

    _, action, _, _, _,_ = agente["memoria"][-1]

    vecs = []
    for a, n in zip(action, head_sizes):
        v = np.zeros(n, dtype=np.float32)
        v[a] = 1.0
        vecs.append(v)

    return np.concatenate(vecs)


def obstaculo_visivel(agente, obstaculos):
    ax, ay = agente["pos"]

    melhor = None
    melhor_dist = 1e9
    melhor_diff = 0

    for o in obstaculos:
        rect = o["rect"]
        ox, oy = rect.centerx, rect.centery

        dx = ox - ax
        dy = oy - ay
        dist = math.hypot(dx, dy)

        if dist > agente["alcance_visao"]:
            continue

        ang = math.atan2(dy, dx)
        diff = (ang - agente["direcao"] + math.pi) % (2 * math.pi) - math.pi

        if abs(diff) > agente["FOV"] / 2:
            continue

        # raycast oclusão
        px, py = raycast_visao(
            agente,
            obstaculos,
            ang,
            max_dist=agente["alcance_visao"]
        )

        dist_raycast = math.hypot(px - ax, py - ay)

        if dist_raycast + 1e-6 < dist:
            continue  # bloqueado

        # pega o mais próximo visível
        if dist < melhor_dist:
            melhor = o
            melhor_dist = dist
            melhor_diff = diff

    return melhor, melhor_dist, melhor_diff


def projetil_visivel(agente, projeteis):
    ax, ay = agente["pos"]

    melhor = None
    melhor_dist = 1e9
    melhor_diff = 0

    for p in projeteis:

        # ignora projétil do próprio agente
        if p["dono"] == agente:
            continue

        px, py = p["pos"]

        dx = px - ax
        dy = py - ay
        dist = math.hypot(dx, dy)

        if dist > agente["alcance_visao"]:
            continue

        ang = math.atan2(dy, dx)
        diff = (ang - agente["direcao"] + math.pi) % (2 * math.pi) - math.pi

        if abs(diff) > agente["FOV"] / 2:
            continue

        # pega o mais próximo visível
        if dist < melhor_dist:
            melhor = p
            melhor_dist = dist
            melhor_diff = diff

    return melhor, melhor_dist, melhor_diff


def vetor_obstaculo(agente, obstaculos, projeteis):
    o, dist_o, diff_o = obstaculo_visivel(agente, obstaculos)
    p, dist_p, diff_p = projetil_visivel(agente, projeteis)

    # escolhe o objeto mais próximo visível
    alvo = None

    if o is None and p is None:
        return [0, 0, 0, 0, 0]

    if o is not None and (p is None or dist_o <= dist_p):
        alvo = "obstaculo"
    else:
        alvo = "projetil"

    # -------- OBSTÁCULO --------
    if alvo == "obstaculo":
        vida = o["health"] / 50
        dist_norm = dist_o / agente["alcance_visao"]
        ang_norm = diff_o / math.pi

        tipo = o["tipo"]
        madeira = 1.0 if tipo == "madeira" else 0.0
        pedra = 1.0 if tipo == "pedra" else 0.0

        return [vida, dist_norm, ang_norm, madeira, pedra]

    # -------- PROJÉTIL --------
    else:
        # ameaça = projétil perigoso
        vida = 1.0
        dist_norm = dist_p / agente["alcance_visao"]
        ang_norm = diff_p / math.pi

        # projétil não gera material

        madeira = 0.0
        pedra = 0.0

        return [vida, dist_norm, ang_norm, madeira, pedra]

ordem_armas = {
    None: 0,
    "bastao": 1,
    "lanca": 2,
    "funda": 3,
    "katana": 5,
    "alabarda": 4,
    "pistola": 6
}

def vetor_armafunc(agente):
    arma = agente.get("arma")

    indice = ordem_armas.get(arma, 0)

    max_indice = max(ordem_armas.values())
    max_indice = max(max_indice, 1)

    return [indice / max_indice]

def codificar_item(item):
    itens = {
        None: 0,
        "madeira": 1,
        "pedra": 2,
        "ferro": 3
    }

    return itens.get(item, 0) / 3
def vetor_inventario(agente):

    inv = agente["inventario"]

    slots = list(inv.items())[:3]

    vetor = []

    for i in range(3):

        if i < len(slots):

            item, qtd = slots[i]

            vetor.extend([
                codificar_item(item),
                min(qtd, 10) / 10
            ])

        else:
            vetor.extend([
                0.0,
                0.0
            ])

    return np.array(vetor, dtype=np.float32)

def vetor_armas_chao(agente):

    armas_visiveis = []

    for item in armas_no_chao:

        dx = item["pos"][0] - agente["pos"][0]
        dy = item["pos"][1] - agente["pos"][1]

        dist = math.hypot(dx, dy)

        if dist > agente["alcance_visao"]:
            continue

        armas_visiveis.append({
            "dx": dx,
            "dy": dy,
            "dist": dist,
            "arma": item["arma"]
        })

    armas_visiveis.sort(key=lambda x: x["dist"])

    vetor = []

    for i in range(3):

        if i < len(armas_visiveis):

            a = armas_visiveis[i]

            vetor.extend([
                a["dx"] / (agente["alcance_visao"] + 1e-8),
                a["dy"] / (agente["alcance_visao"] + 1e-8),
                a["dist"] / (agente["alcance_visao"] + 1e-8),
            ] + vetor_armafunc(a))

        else:
            vetor.extend([0.0] * 4)

    return np.array(vetor, dtype=np.float32)
def obter_estado(agente, populacao, jogadores, obstaculos, rays, tempo_rodada, projeteis):

    # =========================
    # SAFE MAX NORMALIZATION
    # =========================
    max_abs = max(abs(a["pontos"]) for a in populacao)
    max_abs = max_abs if max_abs != 0 else 1.0

    sensores_lidar = np.array(lidar(agente, obstaculos, rays))

    # =========================
    # INIMIGOS (IA + JOGADORES)
    # =========================
    alvos = []

    # IA
    for outro in populacao:
        if outro is agente:
            continue
        alvos.append(outro)

    # PLAYERS (conversão leve)
    for j in jogadores.values():
        alvos.append({
            "pos": j["pos"],
            "vel": j.get("vel", [0.0, 0.0]),
            "vida": j["vida"],
            "vida_max": j["vida_max"],
            "atacando": j["atacando"],
            "max_speed": j.get("max_speed", 1.0),
            "alcance_visao": j.get("alcance_visao", 500),
            "nome": j["nick"]
        })

    # =========================
    # FILTRO VISUAL (TOP 3)
    # =========================
    inimigos_visiveis = []

    for outro in alvos:

        if outro is agente:
            continue

        visivel, ang_rel = inimigo_visivel(
            agente,
            outro,
            obstaculos,
            retornar_angulo=True
        )

        if not visivel:
            continue

        dx = outro["pos"][0] - agente["pos"][0]
        dy = outro["pos"][1] - agente["pos"][1]
        dist = math.hypot(dx, dy)

        inimigos_visiveis.append({
            "dx": dx,
            "dy": dy,
            "dist": dist,
            "ang": ang_rel,
            "vida": outro["vida"],
            "velx": outro["vel"][0],
            "vely": outro["vel"][1],
            "atacando": outro["atacando"]
        })

    inimigos_visiveis = sorted(inimigos_visiveis, key=lambda x: x["dist"])[:3]

    # =========================
    # BASE FEATURES
    # =========================
    direcao_agent = wrap_pi(agente['direcao']) / math.pi

    vida_norm = agente["vida"] / (agente["vida_max"] + 1e-8)
    stamina_norm = agente["stamina"] / (agente["stamina_max"] + 1e-8)

    posx_norm = agente["pos"][0] / LARGURA
    posy_norm = agente["pos"][1] / ALTURA

    pontos_norm = agente["pontos"] / (max_abs + 1e-8)

    # =========================
    # INIMIGOS EMPACOTADOS (3x)
    # =========================
    inimigos_vetor = []

    for i in range(3):
        if i < len(inimigos_visiveis):

            e = inimigos_visiveis[i]

            inimigos_vetor.extend([
                                      e["dx"] / (agente["alcance_visao"] + 1e-8),
                                      e["dy"] / (agente["alcance_visao"] + 1e-8),
                                      e["dist"] / (agente["alcance_visao"] + 1e-8),
                                      e["ang"] / math.pi,
                                      e["vida"] / (agente["vida_max"] + 1e-8),
                                      e["velx"] / (agente["max_speed"] + 1e-8),
                                      e["vely"] / (agente["max_speed"] + 1e-8),
                                      1.0 if e["atacando"] else 0.0
                                  ] + vetor_armafunc(e))
        else:
            inimigos_vetor.extend([0.0] * 9)

    # =========================
    # OWN WEAPON
    # =========================
    vetor_arma = vetor_armafunc(agente)

    # =========================
    # BASE STATE
    # =========================
    if agente['nome'] == 'vermelho':
        estado_base = np.array([
            vida_norm,
            stamina_norm,
            direcao_agent,
            posx_norm,
            posy_norm,
            pontos_norm,
        ], dtype=np.float32)

    elif agente['nome'] == 'azul':
        estado_base = np.array([
            vida_norm,
            stamina_norm,
            direcao_agent,
            posx_norm,
            posy_norm,
            pontos_norm,
        ], dtype=np.float32)

    elif agente['nome'] == 'amarelo':
        balas = agente.get("balas", 0)
        balas_max = agente.get("balas_max", 6)
        balas_norm = balas / (balas_max + 1e-8)

        estado_base = np.array([
            vida_norm,
            stamina_norm,
            direcao_agent,
            posx_norm,
            posy_norm,
            pontos_norm,
            balas_norm
        ], dtype=np.float32)

    # =========================
    # FINAL
    # =========================
    estado = np.concatenate([
        sensores_lidar,
        estado_base,
        vetor_inventario(agente),
        vetor_obstaculo(agente, obstaculos, projeteis),
        vetor_arma,
        vetor_armas_chao(agente),
        np.array(inimigos_vetor, dtype=np.float32)
    ], axis=0)

    return estado
def ejetar_para_fora(agente, obstaculos, raio_max=20):
    if not colidiu_com_obstaculos(agente, obstaculos):
        return

    x0, y0 = agente["pos"]

    for r in range(1, raio_max):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                agente["pos"][0] = x0 + dx
                agente["pos"][1] = y0 + dy

                if not colidiu_com_obstaculos(agente, obstaculos):
                    return


def aplicar_forca_efeitos(agente, listaefeitos):
    # 🔴 ignora agente vermelho
    if agente.get("nome") == "vermelho":
        return

    for e in listaefeitos:
        cx, cy = e["pos"]
        ax, ay = agente["pos"]

        dx = ax - cx
        dy = ay - cy

        dist = math.hypot(dx, dy)

        if dist == 0:
            continue

        if dist > e["raio"]:
            continue

        nx = dx / dist
        ny = dy / dist

        falloff = 1 - (dist / (e["raio"] + 1e-6))
        forca = 0.8 * falloff

        if e["tipo"] == "repulsao":
            fx = nx * forca
            fy = ny * forca
        else:
            fx = -nx * forca
            fy = -ny * forca

        agente["vel"][0] += fx
        agente["vel"][1] += fy
def obter_acao_jogador_rede(comandos):

    if comandos["esquerda"]:
        return 1

    if comandos["direita"]:
        return 2

    if comandos["cima"]:
        return 3

    if comandos["baixo"]:
        return 4

    return 0


def mover_agente_por_acao(agente, acao, obstaculos, listaefeitos):
    custostamina = 0.05
    custo_stamina_lunge = 15.0

    # 🔥 inicialização
    if "vel" not in agente:
        agente["vel"] = [0.0, 0.0]
    if "acc" not in agente:
        agente["acc"] = 0.5
    if "max_speed" not in agente:
        agente["max_speed"] = agente.get("velocidade", 3)
    if "friction" not in agente:
        agente["friction"] = 0.85

    # 🔥 corrige spawn bugado
    if colidiu_com_obstaculos(agente, obstaculos):
        ejetar_para_fora(agente, obstaculos)

    # Variável local temporária para facilitar a leitura do estado atual
    esta_no_lunge = agente.get("atacando") == True and agente.get("arma") == "lanca" and agente.get("tempo_ataque", 0) > 0

    # ========================================================
    # ⚡ SE ESTIVER ATACANDO DE LANÇA (LUNGE ATIVO)
    # ========================================================
    if esta_no_lunge:
        # CORREÇÃO: Removeu a trava de velocidade antiga.
        # Se tem stamina, consome e define a velocidade cravada no teto máximo.
        if agente.get("stamina", 0) >= custo_stamina_lunge:
            agente["stamina"] -= custo_stamina_lunge
            agente["vel"][0] = math.cos(agente["direcao"]) * agente["max_speed"]
            agente["vel"][1] = math.sin(agente["direcao"]) * agente["max_speed"]

    # ========================================================
    # 🎮 MOVIMENTAÇÃO NORMAL (SÓ ENTRA SE NÃO ESTIVER EM LUNGE)
    # ========================================================
    else:
        dx = 0
        dy = 0

        if acao == 1:
            dx = -1
        elif acao == 2:
            dx = 1
        elif acao == 3:
            dy = -1
        elif acao == 4:
            dy = 1

        if dx != 0 or dy != 0:
            if agente["stamina"] >= custostamina:
                agente["stamina"] -= custostamina
            else:
                dx = 0
                dy = 0

        agente["vel"][0] += dx * agente["acc"]
        agente["vel"][1] += dy * agente["acc"]

        aplicar_forca_efeitos(agente, listaefeitos)

    # ========================================================
    # 🧱 LIMITADOR GERAL E ATRITO (Garante a normalização da IA)
    # ========================================================
    vx, vy = agente["vel"]
    speed = math.hypot(vx, vy)

    # CORREÇÃO CRÍTICA: Se passou do max_speed por qualquer motivo (fim do lunge, empurrão, etc), traz de volta ao limite.
    if speed > agente["max_speed"]:
        scale = agente["max_speed"] / (speed + 1e-6)
        agente["vel"][0] *= scale
        agente["vel"][1] *= scale

    # 🔥 CORREÇÃO DO ATRITO: Quando o ataque zera/termina, o atrito volta a frear o agente imediatamente.
    if not esta_no_lunge:
        agente["vel"][0] *= agente["friction"]
        agente["vel"][1] *= agente["friction"]

    # ========================================================
    # 🧱 ATUALIZAÇÃO DE POSIÇÃO E COLISÕES
    # ========================================================
    antiga_x = agente["pos"][0]
    agente["pos"][0] += agente["vel"][0]

    if colidiu_com_obstaculos(agente, obstaculos):
        agente["pos"][0] = antiga_x
        agente["vel"][0] = 0

    antiga_y = agente["pos"][1]
    agente["pos"][1] += agente["vel"][1]

    if colidiu_com_obstaculos(agente, obstaculos):
        agente["pos"][1] = antiga_y
        agente["vel"][1] = 0

    # Limites do mapa
    if agente["pos"][0] < agente["tamanho"]:
        agente["pos"][0] = agente["tamanho"]
        agente["vel"][0] = 0
    elif agente["pos"][0] > LARGURA - agente["tamanho"]:
        agente["pos"][0] = LARGURA - agente["tamanho"]
        agente["vel"][0] = 0

    if agente["pos"][1] < agente["tamanho"]:
        agente["pos"][1] = agente["tamanho"]
        agente["vel"][1] = 0
    elif agente["pos"][1] > ALTURA - agente["tamanho"]:
        agente["pos"][1] = ALTURA - agente["tamanho"]
        agente["vel"][1] = 0



def adicionar_item(agente, item, qtd=1):
    inv = agente["inventario"]

    # Pedra tem 40% de chance de virar ferro
    if item == "pedra":
        if random.random() < 0.4:
            item = "ferro"

    if len(inv) >= agente["inventariomax"] and item not in inv:
        return False

    inv[item] = inv.get(item, 0) + qtd
    return True


receitas = {
    "bastao": {
        "custo": {"madeira": 1},
        "atributos": {
            "dano": 15,
            "stun_chance": 0.35,
            "recompensa_hit": 2.5
        }
    },
"funda": {
    "custo": {"pedra": 1},
    "atributos": {
        "dano": 18,
        "alcance": 320,
        "recompensa_hit": 3.0,

        "nocaute_chance": 0.15,
        "critico_chance": 0.25,
        "desarme_chance": 0.15,
        "funda_cooldown":0,
    }
},
    "lanca": {
        "custo": {"madeira": 1, "pedra": 1},
        "atributos": {
            "dano": 22,
            "alcance": 180,
            "recompensa_hit": 2.7
        }
    },

    "alabarda": {
        "custo": {"madeira": 1, "ferro": 1},
        "atributos": {
            "dano": 30,
            "alcance": 200,
            "recompensa_hit": 3.0
        }
    },
"katana": {
        "custo": {"madeira": 1, "ferro": 2},
        "atributos": {
            "dano": 24,
            "alcance": 130,
            "recompensa_hit": 2.8,
            "corta_projeteis": True
        }
    }
}

def pegar_armas_chao(agente):
    global receitas
    for item in armas_no_chao[:]:  # cópia para poder remover

        dx = agente["pos"][0] - item["pos"][0]
        dy = agente["pos"][1] - item["pos"][1]

        distancia = dx * dx + dy * dy

        # raio de coleta
        if distancia <= 200:

            # só pega se estiver sem arma
            if agente.get("arma") is None and agente['confuso'] <= 0 :

                agente["arma"] = item["arma"]
                arma = item["arma"]
                dados = receitas[arma]["atributos"]
                for atributo, valor in dados.items():
                    agente[atributo] = valor
                armas_no_chao.remove(item)

                return True

    return False


def tentar_crafting(agente):
    inv = agente["inventario"]
    arma_atual = agente.get("arma")

    for item, data in sorted(receitas.items(),
                             key=lambda x: -sum(x[1]["custo"].values())):

        # 🔴 não crafta downgrade
        if ordem_armas[item] <= ordem_armas[arma_atual]:
            continue

        custo = data["custo"]

        pode = True
        for r, q in custo.items():
            if inv.get(r, 0) < q:
                pode = False
                break

        if not pode:
            continue

        # paga custo
        for r, q in custo.items():
            inv[r] -= q
            if inv[r] <= 0:
                del inv[r]

        # equipa arma
        agente["arma"] = item

        # aplica atributos
        for k, v in data["atributos"].items():
            agente[k] = v

        return item

    return None


projeteis = []  # lista global


def disparar_projetil(atacante):
    # ❌ não pode atirar recarregando
    if atacante.get('arma') == "pistola":
        if atacante["recarregando"]:
            return
        if atacante['balas'] > 0:
            velocidade = 8
            direcao = atacante["direcao"]
            atacante["balas"] -= 1
            vx = math.cos(direcao) * velocidade
            vy = math.sin(direcao) * velocidade

            projeteis.append({
                "pos": atacante["pos"][:],
                "vel": [vx, vy],
                "raio": 15,
                "dano": atacante.get("dano", 10),
                "dono": atacante,
                "vida": 60
            })
    elif atacante.get("arma") == "funda":
        velocidade = 8
        direcao = atacante["direcao"]
        vx = math.cos(direcao) * velocidade
        vy = math.sin(direcao) * velocidade

        projeteis.append({
            "pos": atacante["pos"][:],
            "vel": [vx, vy],
            "raio": 15,
            "dano": atacante.get("dano", 10),
            "dono": atacante,
            "vida": 60
        })


import random

def atualizar_projeteis(obstaculos: list, jogadores):
    remover = []

    for p in projeteis:

        # 🔥 guarda posição anterior (anti-ghost)
        prev_x, prev_y = p["pos"][0], p["pos"][1]

        # mover
        p["pos"][0] += p["vel"][0]
        p["pos"][1] += p["vel"][1]
        p["vida"] -= 1

        if p["vida"] <= 0:
            remover.append(p)
            continue


        # 🔥 colisão com obstáculo (interpolação)
        steps = 3
        colidiu_obs = False

        for i in range(steps):

            t = i / steps

            x = prev_x + (p["pos"][0] - prev_x) * t
            y = prev_y + (p["pos"][1] - prev_y) * t

            for obs in obstaculos:

                if obs["rect"].collidepoint(int(x), int(y)):

                    obs["health"] -= p["dano"]

                    remover.append(p)
                    colidiu_obs = True
                    break

            if colidiu_obs:
                break


        if colidiu_obs:
            continue



        # 🔥 colisão contra agentes E jogadores
        entidades = populacao + list(jogadores.values())

        for defensor in entidades:

            if defensor is p["dono"]:
                continue

            if defensor["vida"] <= 0:
                continue


            dx = defensor["pos"][0] - p["pos"][0]
            dy = defensor["pos"][1] - p["pos"][1]

            r = p["raio"] + defensor["tamanho"]


            if dx * dx + dy * dy <= r * r:

                dono = p["dono"]


                # =========================
                # EFEITOS DO PROJÉTIL
                # =========================

                efeitos = p.get("efeitos", {})


                dano = p["dano"]


                # 💥 crítico
                if random.random() < efeitos.get("critico_chance", 0):

                    dano *= 2

                    dono["recompensa_frame"] += 1.5



                defensor["vida"] -= dano



                # 💤 nocaute
                if random.random() < efeitos.get("nocaute_chance", 0):

                    defensor["stun"] = 60

                    dono["recompensa_frame"] += 2.0



                # 🔥 desarme
                if random.random() < efeitos.get("desarme_chance", 0):

                    if defensor.get("arma") is not None:
                        defensor['confuso'] = 300
                        dropar_arma(defensor)

                        dono["recompensa_frame"] += 2.0
                        defensor["recompensa_frame"] -= 2.0



                # =========================
                # RECOMPENSAS
                # =========================

                dono["recompensa_frame"] += p.get(
                    "recompensa_hit",
                    0.5
                )

                defensor["recompensa_frame"] -= 0.5


                dono["pontos"] += 1
                defensor["pontos"] -= 1



                if defensor["vida"] <= 0:

                    dono["recompensa_frame"] += 10.0
                    defensor["recompensa_frame"] -= 10.0

                    dono["pontos"] += 1



                remover.append(p)
                break



    # remover depois do loop
    for p in remover:

        if p in projeteis:
            projeteis.remove(p)
def obter_agente_amarelo(populacao):
    for a in populacao:
        if a.get("nome") == "amarelo":
            return a
    return None
def disparar_projetil_angulo(pos, angulo, velocidade=8, dano=10, dono=None):
    vx = math.cos(angulo) * velocidade
    vy = math.sin(angulo) * velocidade
    projeteis.append({
        "pos": list(pos),
        "vel": [vx, vy],
        "raio": 15,
        "dano": dano,
        "dono": dono,   # pode ser None ou portal/agente
        "vida": 60
    })
def distorcer_leve(screen, centro, intensidade, raio=100, passo=8):
    largura, altura = screen.get_size()

    for x in range(0, largura, passo):
        for y in range(0, altura, passo):
            dx = x - centro[0]
            dy = y - centro[1]

            dist = (dx * dx + dy * dy) ** 0.5

            # 🔥 LIMITADOR DE ALCANCE
            if dist > raio:
                continue

            # evita divisão por zero
            dist += 0.0001

            fator = intensidade / dist

            src_x = int(x + dx * fator)
            src_y = int(y + dy * fator)

            rect = pygame.Rect(x, y, passo, passo)

            if 0 <= src_x < largura and 0 <= src_y < altura:
                bloco = screen.subsurface(rect).copy()
                screen.blit(bloco, (src_x, src_y))
def distorcao_convergindo(screen, centro, intensidade, raio_max, tempo, duracao, passo=8):
    # progresso de 0 → 1
    t = min(tempo / duracao, 1.0)

    # raio diminui
    raio = raio_max * (1 - t)

    largura, altura = screen.get_size()

    for x in range(0, largura, passo):
        for y in range(0, altura, passo):
            dx = x - centro[0]
            dy = y - centro[1]

            dist = (dx * dx + dy * dy) ** 0.5

            if dist > raio:
                continue

            dist += 0.0001

            falloff = 1 - (dist / raio)

            # 🔥 negativo = puxa pro centro
            fator = -(intensidade * falloff) / dist

            src_x = int(x + dx * fator)
            src_y = int(y + dy * fator)

            rect = pygame.Rect(x, y, passo, passo)

            if 0 <= src_x < largura and 0 <= src_y < altura:
                bloco = screen.subsurface(rect).copy()
                screen.blit(bloco, (src_x, src_y))
def distorcao_expandindo(screen, centro, intensidade, raio_max, tempo, duracao, passo=8):
    # progresso de 0 → 1
    t = min(tempo / duracao, 1.0)

    # raio cresce
    raio = raio_max * t

    largura, altura = screen.get_size()

    for x in range(0, largura, passo):
        for y in range(0, altura, passo):
            dx = x - centro[0]
            dy = y - centro[1]

            dist = (dx * dx + dy * dy) ** 0.5

            if dist > raio:
                continue

            dist += 0.0001

            # falloff suave (fica bonito)
            falloff = 1 - (dist / raio)

            fator = (intensidade * falloff) / dist

            src_x = int(x + dx * fator)
            src_y = int(y + dy * fator)

            rect = pygame.Rect(x, y, passo, passo)

            if 0 <= src_x < largura and 0 <= src_y < altura:
                bloco = screen.subsurface(rect).copy()
                screen.blit(bloco, (src_x, src_y))
def processar_combate(atacante, defensor, acao):
    global hitboxes_ataque, obstaculos,circulos,projeteis
    recompensaframe = atacante.get("recompensa_frame", 0)
    punicaoframe = defensor.get("recompensa_frame", 0)
    arma_pistola = atacante.get("arma") == "pistola"
    # 1. Busca os valores com segurança. Se a chave não existir, assume False por padrão.
    está_treinando = defensor.get('treinando', False)
    está_morto = defensor.get('morto', False)
    está_vivo = defensor.get('vivo', True)  # Assume True por padrão, já que não está morto

    # 2. Executa a lógica de parada de forma limpa
    if está_treinando or está_morto:
        return

    if not está_vivo:
        return

    dano = atacante.get("dano", 10)
    alcance = atacante.get("alcance", 100)

    # ================= ATAQUE CORPO-A-CORPO EM ANDAMENTO =================
    if atacante.get("tempo_ataque", 0) > 0 and not arma_pistola:
        atacante["tempo_ataque"] -= 1
        atacante["atacando"] = True

        pos = atacante["pos"][:]
        direcao = atacante["direcao"]

        hitboxes_ataque.append((pos, alcance))

        # ================= ATAQUE CORPO-A-CORPO EM ANDAMENTO =================
        if atacante.get("tempo_ataque", 0) > 0 and not arma_pistola:

            atacante["tempo_ataque"] -= 1
            atacante["atacando"] = True

            pos = atacante["pos"][:]
            direcao = atacante["direcao"]

            dir_x = math.cos(direcao)
            dir_y = math.sin(direcao)

            hitboxes_ataque.append((pos, alcance))

            # ================= CORTE DE PROJÉTEIS (KATANA) =================
            if atacante.get("arma") == "katana":

                for p in projeteis[:]:

                    px = p["pos"][0]
                    py = p["pos"][1]

                    dxp = px - pos[0]
                    dyp = py - pos[1]

                    distancia_p = math.hypot(dxp, dyp)

                    if distancia_p <= alcance:

                        nxp = dxp / max(distancia_p, 1e-6)
                        nyp = dyp / max(distancia_p, 1e-6)

                        dot_proj = dir_x * nxp + dir_y * nyp

                        # arco de 90 graus na frente
                        if dot_proj >= math.cos(math.radians(45)):
                            projeteis.remove(p)

                            recompensaframe += 0.2

            # -------- dano no defensor --------
            if not atacante.get("acertou_este_ataque", False):

                dx = defensor["pos"][0] - pos[0]
                dy = defensor["pos"][1] - pos[1]

                distancia = math.hypot(dx, dy)

                if 1e-6 < distancia <= alcance:

                    nx = dx / distancia
                    ny = dy / distancia

                    dot = dir_x * nx + dir_y * ny

                    if dot >= math.cos(math.radians(45)):

                        # 🟡 COUNTER DO AMARELO
                        if defensor.get("nome") == "amarelo" and defensor.get("counter_janela", 0) > 0:

                            listargb.append(criar_convergencia_portal(populacao))

                            defensor["counter_janela"] = 0

                            defensor["recompensa_frame"] += 0.5
                            atacante["recompensa_frame"] -= 0.5


                        else:

                            # ================= DANO NORMAL =================

                            defensor["vida"] -= dano

                            # efeito do bastão
                            if atacante.get("arma") == "bastao":

                                if random.random() < atacante.get("stun_chance", 0):
                                    defensor["stun"] = 120

                            defensor["vida"] = max(0, defensor["vida"])

                            if defensor["vida"] == 0:
                                punicaoframe -= 10.0
                                recompensaframe += 10.0

                            atacante["pontos"] += 1
                            defensor["pontos"] -= 1

                            recompensaframe += 0.5 + (atacante["recompensa_hit"] - 2)
                            punicaoframe -= 0.5 + (atacante["recompensa_hit"] - 2)

                        atacante["acertou_este_ataque"] = True

            # -------- dano em obstáculos --------
            for o in obstaculos[:]:

                if o["qtd"] <= 0:
                    continue

                rect = o["rect"]

                closest_x = max(rect.left, min(pos[0], rect.right))
                closest_y = max(rect.top, min(pos[1], rect.bottom))

                dx = closest_x - pos[0]
                dy = closest_y - pos[1]

                distancia = math.hypot(dx, dy)

                if 1e-6 < distancia <= alcance:

                    nx = dx / distancia
                    ny = dy / distancia

                    dot = dir_x * nx + dir_y * ny

                    if dot >= math.cos(math.radians(45)):

                        o["health"] -= dano

                        recompensaframe += 0.5

                        if o["health"] <= 0:
                            adicionar_item(atacante, o["tipo"], o["qtd"])

                            recompensaframe += 1

                            obstaculos.remove(o)

                            break
    # ================= RESET MELEE =================
    if atacante.get("tempo_ataque", 0) <= 0 and not arma_pistola:
        atacante["atacando"] = False
        atacante["acertou_este_ataque"] = False
    if  atacante['cosmetico'] == 'coroa':
        if atacante['paralisia_cd'] > 0:
            atacante['paralisia_cd'] -= 1
        if acao == 3:
            if atacante['paralisia_cd'] <= 0:
                circulo = criar_circulo(atacante)
                circulos.append(circulo)

                atacante['paralisia_cd'] = 900  # ou o valor que você quiser
    if atacante['nome'] == 'vermelho' and acao == 3:
        if atacante['distortioncd'] > 0:
            atacante['distortioncd'] -= 1
        if atacante['stamina'] > 0 and atacante['distortioncd'] == 0:
            atacante['stamina'] -= 10
            atacante['distortioncd'] = 60*5
            listargb.append(criar_convergencia(atacante))
    elif atacante['nome'] == 'vermelho' and acao == 4:
        if atacante['distortioncd'] > 0:
            atacante['distortioncd'] -= 1
        if atacante['stamina'] > 0 and atacante['distortioncd'] == 0:
            atacante['stamina'] -= 10
            atacante['distortioncd'] = 60 * 10
            listargb.append(criar_repulsao(atacante))

    if acao == 1:

        # -------- PISTOLA --------
        if arma_pistola:
            if atacante["stamina"] >= atacante["custo_ataque"] and atacante["cooldown_ataque"] <= 0:
                atacante["stamina"] -= atacante["custo_ataque"]
                atacante["cooldown_ataque"] = 10
                atacante["atacando"] = True
                disparar_projetil(atacante)
        elif atacante.get("arma") == "funda":

            if atacante["stamina"] >= atacante["custo_ataque"] and atacante["cooldown_ataque"] <= 0:
                atacante["stamina"] -= atacante["custo_ataque"]
                atacante['tempo_ataque'] = 90
                atacante["cooldown_ataque"] = 20

                atacante["atacando"] = True

                disparar_projetil(atacante)

        # -------- MELEE --------
        else:
            if atacante["stamina"] >= atacante["custo_ataque"] and atacante["cooldown_ataque"] <= 0:
                atacante["stamina"] -= atacante["custo_ataque"]
                atacante["cooldown_ataque"] = 60
                atacante["tempo_ataque"] = atacante["DURACAO_ATAQUE"]
                atacante["atacando"] = True
                atacante["acertou_este_ataque"] = False


    # ================= CRAFT =================
    elif acao == 2 and atacante["cooldown_ataque"] <= 0:

        arma = tentar_crafting(atacante)
        atacante["cooldown_ataque"] = 20

        if arma is None:
            return

        recompensa = 0
        if arma == "bastao":
            recompensa = 1
        elif arma == "lanca":
            recompensa = 2
        elif arma == 'alabarda':
            recompensa = 3
        elif arma == 'funda':
            recompensa = 1
        elif arma == 'katana':
            recompensa = 3

        recompensaframe += recompensa



    # 🔫 ação de recarregar (separada!)
    if arma_pistola:
        if acao == 2 and not atacante["recarregando"]:
            if atacante["balas"] < atacante["max_balas"]:
                atacante["recarregando"] = True
                atacante["reload_timer"] = 60
        # 🔥 COUNTER DO AMARELO (pistola)
        if acao == 3:
            if atacante.get("counter_cd", 0) <= 0 and atacante["stamina"] >= 5:
                atacante["stamina"] -= 5
                atacante["counter_janela"] = 10  # frames ativos
                atacante["counter_cd"] = 60  # cooldown

        # ================= SALVAR REWARD =================
    atacante["recompensa_frame"] = recompensaframe
    defensor["recompensa_frame"] = punicaoframe



def desenhar_agente(agente, agentes):
    cor = agente["cor"]

    # Corpo do agente
    pygame.draw.circle(
        tela,
        cor,
        agente["pos"],
        agente["tamanho"]
    )

    # Barras
    desenhar_barra_vida(agente)
    desenhar_barra_stamina(agente)

    # =========================
    # PROGRESSO DE TREINO
    # =========================
    if agente["treinando"]:

        largura = 40
        altura = 6

        x = agente["pos"][0] - largura // 2
        y = agente["pos"][1] - agente["tamanho"] - 25

        progresso = max(
            0.0,
            min(1.0, agente["progresso_treino"]/100)
        )

        # Fundo
        pygame.draw.rect(
            tela,
            (60, 60, 60),
            (x, y, largura, altura)
        )

        # Preenchimento
        pygame.draw.rect(
            tela,
            (0, 255, 255),
            (
                x,
                y,
                int(largura * progresso),
                altura
            )
        )

        # Borda
        pygame.draw.rect(
            tela,
            (255, 255, 255),
            (x, y, largura, altura),
            1
        )

    # 🔥 DEBUG DE VISÃO
    for inimigo in agentes:

        if inimigo is agente:
            continue

        visivel = inimigo_visivel(
            agente,
            inimigo,
            obstaculos
        )

        if visivel:
            pygame.draw.circle(
                tela,
                (255, 255, 0),
                inimigo["pos"],
                6
            )
def desenhar_barra_stamina(agente):
    largura_barra = agente["tamanho"] * 2
    altura_barra = 5

    # Um pouco abaixo da barra de vida
    x = agente["pos"][0] - largura_barra // 2
    y = agente["pos"][1] - agente["tamanho"] - 7

    proporcao = agente["stamina"] / agente["stamina_max"]
    largura_stamina = int(largura_barra * proporcao)

    # Fundo
    pygame.draw.rect(tela, (50, 50, 50), (x, y, largura_barra, altura_barra))

    # Stamina (azul)
    pygame.draw.rect(tela, (0, 150, 255), (x, y, largura_stamina, altura_barra))

    # Borda
    pygame.draw.rect(tela, (0, 0, 0), (x, y, largura_barra, altura_barra), 1)


def extrair_batch_por_episodios(memoria, max_batch, n=8, max_len=None):
    # 🔹 1. separar por episódios
    episodios = []
    atual = []

    for exp in memoria:
        atual.append(exp)
        if exp[3]:  # done
            episodios.append(atual)
            atual = []

    if atual:
        episodios.append(atual)

    if len(episodios) == 0:
        return []

    # 🔹 2. embaralhar
    random.shuffle(episodios)

    # 🔹 3. filtrar episódios muito grandes
    filtrados = []
    for ep in episodios:
        if len(ep) > max_batch:
            continue
        filtrados.append(ep)

    if len(filtrados) == 0:
        filtrados = episodios  # fallback

    # 🔹 4. escolher N episódios
    if len(filtrados) >= n:
        selecionados = filtrados[:n]
    else:
        # 🔥 fallback: sample com repetição
        selecionados = [random.choice(filtrados) for _ in range(n)]

    # 🔹 5. limitar tamanho (padding-free approach)
    if max_len is not None:
        selecionados = [ep[:max_len] for ep in selecionados]

    return selecionados
def salvar_experiencia(
    agente,
    estado,
    acao_ppo,       # Agora recebe o ID único de 0 a 12 (agente["acao_pura_ppo"])
    recompensa,
    done,
    log_prob,
    value
):
    # Garante a inicialização da lista caso não exista
    if "memoria" not in agente:
        agente["memoria"] = []

    # Estrutura a tupla exatamente como o PPO puro precisa
    experiencia = (
        np.asarray(estado, dtype=np.float32),   # Estado (71,)
        acao_ppo,                                # Ação discreta única (int)
        float(recompensa),                       # Recompensa do frame (float)
        bool(done),                              # Indicador de fim de jogo (bool)
        float(np.asarray(log_prob).squeeze()),   # Probabilidade logarítmica (float)
        float(np.asarray(value).squeeze())       # Valor estimado pelo crítico (float)
    )

    agente["memoria"].append(experiencia)





def preparar_batch(agente, max_batch, batch_size):

    memoria = agente["memoria"]

    if len(memoria) < batch_size:
        return None

    episodios = extrair_batch_por_episodios(
        memoria,
        max_batch,
        n=batch_size
    )

    if len(episodios) == 0:
        return None

    max_len = max(len(ep) for ep in episodios)

    state_dim = len(episodios[0][0][0])

    return episodios, max_len, state_dim
def executar_forward_batch(
    rede,
    episodios,
    max_len,
    batch_size,
    state_dim
):
    import numpy as np

    values = np.zeros(
        (max_len, batch_size),
        dtype=np.float32
    )

    rewards = np.zeros(
        (max_len, batch_size),
        dtype=np.float32
    )

    dones = np.zeros(
        (max_len, batch_size),
        dtype=bool
    )

    old_log_probs = np.zeros(
        (max_len, batch_size),
        dtype=np.float32
    )

    mask = np.zeros(
        (max_len, batch_size),
        dtype=np.float32
    )

    outputs_cache = []

    rede.gru.reset_state()

    for t in range(max_len):

        states = np.zeros(
            (batch_size, state_dim),
            dtype=np.float32
        )

        actions_batch = []
        h_states = []
        z_batch = []
        plan_prob_batch = []
        current_dones = np.zeros(batch_size, dtype=bool)

        for i, ep in enumerate(episodios):

            if t < len(ep):

                (
                    s,
                    a,
                    r,
                    d,
                    logp,
                    value,
                    h_state,
                    z_saved,
                    plan_prob_saved
                ) = ep[t]

                states[i] = s
                rewards[t, i] = r
                current_dones[i] = bool(d)
                dones[t, i] = bool(d)
                old_log_probs[t, i] = float(logp)
                values[t, i] = float(value)

                mask[t, i] = 1.0
                actions_batch.append(a)

                h_states.append(
                    h_state.squeeze()
                )
                z_batch.append(z_saved)
                plan_prob_batch.append(
                    plan_prob_saved
                )

            else:
                # Fallback seguro para padding
                sample_action = episodios[0][0][1]
                actions_batch.append(
                    [0] * len(sample_action) if isinstance(sample_action, (list, np.ndarray)) else 0
                )

                h_states.append(
                    np.zeros_like(
                        episodios[0][0][6].squeeze()
                    )
                )
                z_batch.append(
                    np.zeros_like(
                        episodios[0][0][7]
                    )
                )
                plan_prob_batch.append(
                    0.0
                )

        z_batch = np.array(
            z_batch,
            dtype=np.float32
        )

        plan_prob_batch = np.array(
            plan_prob_batch,
            dtype=np.float32
        ).reshape(-1, 1)

        # Atualiza o hidden state da GRU carregado do buffer antes do forward
        rede.gru.h = np.array(
            h_states,
            dtype=np.float32
        )

        # CORREÇÃO CRUCIAL: Zera o hidden state ANTES do forward se o passo anterior/atual marcou done
        if np.any(current_dones):
            rede.gru.h[current_dones] = 0.0

        if rede.gru.h is not None:
            rede.gru.h *= mask[t].reshape(
                -1,
                1
            )

        # Executa o forward com a hidden state devidamente limpa para quem terminou
        outputs, values_pred, _, _ = rede.forward(
            states
        )

        outputs_cache.append(
            (
                outputs,
                actions_batch,
                z_batch,
                plan_prob_batch,
                values_pred
            )
        )

    return {
        "values": values,
        "rewards": rewards,
        "dones": dones,
        "old_log_probs": old_log_probs,
        "mask": mask,
        "outputs_cache": outputs_cache
    }


def calcular_prediction_error(
    rede,
    outputs_cache,
    rewards,
    mask,
    batch_size,
    max_len
):
    import numpy as np

    z_cache = [
        item[2]
        for item in outputs_cache
    ]

    # Desloca o latente para frente para representar o z_{t+1} real
    z_next_cache = z_cache[1:] + [
        np.zeros_like(z_cache[0])
    ]

    pred_error = np.zeros_like(
        rewards
    )

    # Executa por toda a sequência até max_len
    for t in range(max_len):
        # Se a máscara inteira do batch for 0 neste passo, pula
        current_mask = mask[t].squeeze() if mask[t].ndim > 1 else mask[t]
        if current_mask.sum() == 0:
            continue

        (
            outputs,
            actions_batch,
            z_t,
            _,
            _
        ) = outputs_cache[t]

        z_next = z_next_cache[t]

        action_vec = []

        for h, probs in enumerate(outputs):
            actions = np.array(
                [a[h] for a in actions_batch]
            )

            onehot = np.zeros_like(
                probs
            )

            onehot[
                np.arange(batch_size),
                actions
            ] = 1

            action_vec.append(
                onehot
            )

        action_vec = np.concatenate(
            action_vec,
            axis=1
        )

        # Prediz o próximo estado latente via modelo do agente
        z_pred = (
            rede.predict_next_latent_batch(
                z_t,
                action_vec
            )
        )

        # Erro quadrático médio por elemento do batch
        err = np.mean(
            (z_pred - z_next) ** 2,
            axis=1
        )

        # Aplicação segura da máscara com o formato correto
        pred_error[t] = err * current_mask

    return (
        pred_error,
        z_next_cache
    )


def calcular_GAE(
        rewards,
        values,
        dones,
        mask,
        gamma,
        lam=0.95
):
    import numpy as np

    max_len = len(values)
    batch_size = values.shape[1]

    advantages = np.zeros_like(values)
    returns_seq = np.zeros_like(values)  # Criamos uma matriz para os retornos estáveis

    last_gae = np.zeros(batch_size, dtype=np.float32)
    last_return = np.zeros(batch_size, dtype=np.float32)  # Alvo do crítico acumulado

    not_dones = 1.0 - dones.astype(np.float32)

    for t in reversed(range(max_len)):
        next_value = (
            values[t + 1]
            if t < max_len - 1
            else np.zeros(batch_size, dtype=np.float32)
        )

        # Delta TD bruto
        delta = (
                rewards[t]
                + gamma * next_value * not_dones[t]
                - values[t]
        )
        delta *= mask[t]

        # Acumulação temporal do GAE amortecido
        last_gae = (
                delta
                + gamma * lam * not_dones[t] * last_gae
        )
        last_gae *= mask[t]
        advantages[t] = last_gae

        # 🔥 CÁLCULO ESTÁVEL DE RETORNO (Alvo do Crítico puro, sem depender do vício de advantages)
        last_return = rewards[t] + gamma * not_dones[t] * last_return
        last_return *= mask[t]
        returns_seq[t] = last_return

    # --- NORMALIZAÇÃO RESTRITA ÀS ÁREAS VÁLIDAS ---
    mask_bool = mask.astype(bool)

    if np.sum(mask_bool) > 1:
        valid_advantages = advantages[mask_bool]
        mean = valid_advantages.mean()
        std = valid_advantages.std()

        # Aplica a normalização mantendo fora da máscara zerado
        advantages = np.where(mask_bool, (advantages - mean) / (std + 1e-8), 0.0)

    # ⚠️ Clip de segurança nos retornos para impedir que bônus cumulativos passem de limites saudáveis
    returns_seq = np.clip(returns_seq, -50.0, 50.0)

    return advantages, returns_seq


def calcular_ratio_ppo(
    outputs,
    actions_batch,
    old_logp,
    batch_size,
    clip_eps,
    advantages_t
):
    new_logp = np.zeros(batch_size, dtype=np.float32)

    for h, probs in enumerate(outputs):
        probs = np.clip(probs, 1e-8, 1.0)
        actions = np.array([a[h] for a in actions_batch])
        new_logp += np.log(probs[np.arange(batch_size), actions])

    ratio = np.exp(new_logp - old_logp)
    clipped = np.clip(ratio, 1 - clip_eps, 1 + clip_eps)

    # O PPO busca MAXIMIZAR o objetivo substituto.
    # Para o otimizador que MINIMIZA a perda no backward, usamos o sinal negativo (-).
    policy_weight = -np.minimum(ratio * advantages_t, clipped * advantages_t).reshape(-1, 1)

    return ratio, policy_weight


def calcular_policy_grads(
    outputs,
    actions_batch,
    batch_size,
    policy_weight,
    entropy_beta,
    valid_mask
):
    douts = []
    action_vec = []

    for h, probs in enumerate(outputs):
        probs = np.clip(probs, 1e-8, 1.0)
        actions = np.array([a[h] for a in actions_batch])

        # 1. Gradiente da Política (PPO) em relação aos logits
        # dL/dlogits = (probs - target) * policy_weight
        grad_ppo = probs.copy()
        grad_ppo[np.arange(batch_size), actions] -= 1.0
        grad_ppo = grad_ppo * policy_weight

        # 2. Gradiente Correto da Entropia em relação aos logits
        # H = -sum(p * log(p))
        # dH/dlogits = probs * (log(probs) - E[log(probs)])
        # Queremos MAXIMIZAR a entropia, ou seja, MINIMIZAR (-entropy_beta * H)
        log_probs = np.log(probs)
        mean_log_probs = np.sum(probs * log_probs, axis=1, keepdims=True)
        grad_entropy = -entropy_beta * probs * (log_probs - mean_log_probs)

        # Somamos os gradientes (ambos já ajustados para minimização no backward)
        grad = grad_ppo + grad_entropy
        grad *= valid_mask

        douts.append(grad)

        # Construção do vetor one-hot original
        onehot = np.zeros_like(probs)
        onehot[np.arange(batch_size), actions] = 1
        action_vec.append(onehot)

    action_vec = np.concatenate(action_vec, axis=1)

    return douts, action_vec


def calcular_value_loss(
        values_pred,
        old_v,
        target,
        clip_eps,
        valid_mask
):
    import numpy as np
    values_pred_flat = values_pred.reshape(-1)

    # Clipping do valor previsto em relação ao valor antigo (old_v)
    v_clipped = old_v + np.clip(
        values_pred_flat - old_v,
        -clip_eps,
        clip_eps
    )

    loss_unclipped = (values_pred_flat - target) ** 2
    loss_clipped = (v_clipped - target) ** 2

    # PPO value clipping loss (pega o máximo entre os dois para ser pessimista)
    loss_clipped_max = np.maximum(loss_unclipped, loss_clipped)

    # Gradiente em relação à predição de valor (derivada de 0.5*(pred - target)^2 é pred - target)
    # Selecionamos o valor que gerou o loss de acordo com o clipping do PPO
    use_clipped = (loss_clipped > loss_unclipped)
    final_values = np.where(use_clipped, v_clipped, values_pred_flat)

    dval = (final_values - target).reshape(-1, 1)
    dval = np.clip(dval, -10.0, 10.0)

    dval *= valid_mask

    # Média do loss restrita apenas aos elementos válidos da máscara
    mask_flat = valid_mask.reshape(-1)
    if mask_flat.sum() > 0:
        value_loss = np.sum(loss_clipped_max * mask_flat) / (mask_flat.sum() + 1e-8)
    else:
        value_loss = 0.0

    return dval, value_loss


def calcular_plan_target(
    rede,
    z_t,
    action_vec,
    values_t,
    pred_error_t,
    mask_t
):
    import numpy as np
    z_sim = rede.predict_next_latent_batch(
        z_t,
        action_vec
    )

    x_sim = z_sim

    for layer in rede.layers:
        x_sim = layer.activate_batch(x_sim)

    v_sim = rede.val_head.activate_batch_linear(x_sim).reshape(-1)

    # Garante compatibilidade de formas (1D) para as máscaras e vantagens
    v_flat = values_t.reshape(-1)
    mask_flat = mask_t.reshape(-1)
    err_flat = pred_error_t.reshape(-1) if hasattr(pred_error_t, 'reshape') else pred_error_t

    adv_plan = (v_sim - v_flat) * mask_flat

    score = (
        rede.plan_alpha * adv_plan
        + rede.plan_beta * err_flat
    )

    score = np.clip(score, -5, 5)

    return (
        1 / (1 + np.exp(-score))
    ).reshape(-1, 1)


def treinar_agente(agente, gamma=0.99, lam=0.95, epochs=10):
    with lockindividual:
        """
        Função de treino para o novo PPOAgent Denso e Vetorizado.
        USA A SUA FUNÇÃO COMPUTE_GAE ORIGINAL.
        """
        if "memoria" not in agente or len(agente["memoria"]) == 0:
            return 0.0

        ppo_agent = agente['rede']
        memoria = agente['memoria']

        agente['progresso_treino'] = 0

        # =========================================================================
        # 1. DESEMPACOTAMENTO DA MEMÓRIA (Mesmo formato da sua tupla de 6 elementos)
        # =========================================================================
        states = np.array([m[0] for m in memoria]).T  # Formato: (71, batch_size)
        actions = np.array([m[1] for m in memoria])  # (batch_size,)
        rewards = np.array([m[2] for m in memoria])  # (batch_size,)
        dones = np.array([m[3] for m in memoria])  # (batch_size,)
        old_log_probs = np.array([m[4] for m in memoria])  # (batch_size,)
        values = [m[5] for m in memoria]  # Lista para append do bootstrap

        agente['progresso_treino'] += 20

        # Converte os 'dones' do jogo nas 'masks' (1.0 ativo, 0.0 morto) que seu compute_gae pede
        masks = [0.0 if d else 1.0 for d in dones]

        # =========================================================================
        # 2. BOOTSTRAP DO CRÍTICO (ÚLTIMO ESTADO)
        # =========================================================================
        # Pegamos o último estado registrado para prever o V(s_prime) do passo final
        ultimo_estado_bruto = memoria[-1][0]
        _, _, ultimo_valor = ppo_agent.get_action(ultimo_estado_bruto)
        values.append(float(ultimo_valor))

        agente['progresso_treino'] += 20

        # =========================================================================
        # 3. CHAMADA DA SUA FUNÇÃO COMPUTE_GAE
        # =========================================================================
        returns, advantages = compute_gae(rewards, values, masks, gamma, lam)

        # Redimensiona para matrizes de 2D (1, batch_size) exigidas pelo train_step vetorizado
        advantages = advantages.reshape(1, -1)
        returns = returns.reshape(1, -1)

        agente['progresso_treino'] += 20

        # Métricas de perda do Crítico antes de atualizar os pesos
        values_before = ppo_agent.critic.forward(states)
        loss_before = np.mean((values_before - returns) ** 2)

        # =========================================================================
        # 4. LOOP DE ÉPOCAS DO PPO (Treina Ator e Crítico)
        # =========================================================================
        for epoch in range(epochs):
            ppo_agent.train_step(states, actions, old_log_probs, returns, advantages)

        agente['progresso_treino'] += 20

        # Métricas de perda depois do treino
        values_after = ppo_agent.critic.forward(states)
        loss_after = np.mean((values_after - returns) ** 2)

        total_reward = float(np.sum(rewards))
        agente['progresso_treino'] += 10

        # =========================================================================
        # 5. RELATÓRIO DE TREINAMENTO E LIMPEZA
        # =========================================================================
        print("\n================ LOG DE TREINO PPO ================")
        print(f"Value Loss (Antes):  {loss_before:.6f}")
        print(f"Value Loss (Depois): {loss_after:.6f}")
        print(f"📊 Reward acumulado no episódio: {total_reward:.3f}")
        print(f"🚀 Total de frames processados:  {states.shape[1]}")
        print("🔥 TREINO DO PPO DENSO CONCLUÍDO COM SUCESSO")
        print("===================================================\n")

        # Reseta o ambiente de treino do agente
        agente['memoria'].clear()
        agente['progresso_treino'] = 100
        for agente in populacao:
            print("Salvando redes neurais...")

            arquivo = nome_arquivo_agente(agente)
            salvar_rede(agente, arquivo)
        return total_reward
def desenhar_barra_vida(agente):
    # Posição da barra (um pouco acima da cabeça)
    largura_barra = agente["tamanho"] * 2
    altura_barra = 6

    x = agente["pos"][0] - largura_barra // 2
    y = agente["pos"][1] - agente["tamanho"] - 15

    # Porcentagem de vida
    proporcao = agente["vida"] / agente["vida_max"]
    largura_vida = int(largura_barra * proporcao)

    # Cor da vida (verde → amarelo → vermelho)
    if proporcao > 0.6:
        cor = (0, 200, 0)  # verde
    elif proporcao > 0.3:
        cor = (255, 200, 0)  # amarelo
    else:
        cor = (255, 0, 0)  # vermelho

    # Fundo da barra (cinza escuro)
    pygame.draw.rect(tela, (50, 50, 50), (x, y, largura_barra, altura_barra))

    # Vida atual
    pygame.draw.rect(tela, cor, (x, y, largura_vida, altura_barra))

    # Borda preta
    pygame.draw.rect(tela, (0, 0, 0), (x, y, largura_barra, altura_barra), 1)

def desenhar_funda(tela, agente):

    x, y = agente["pos"]
    direcao = agente["direcao"]
    tempo_ataque = agente.get("tempo_ataque", 0)

    # posição da mão
    mao_x = x + math.cos(direcao) * 20
    mao_y = y + math.sin(direcao) * 20


    # animação girando
    rotacao = 0

    if tempo_ataque > 0:
        rotacao = 90 - tempo_ataque * 8


    angulo = direcao + math.radians(rotacao)


    # tamanho da funda
    comprimento = 40


    pedra_x = mao_x + math.cos(angulo) * comprimento
    pedra_y = mao_y + math.sin(angulo) * comprimento


    # corda principal
    pygame.draw.line(
        tela,
        (90, 50, 20),
        (mao_x, mao_y),
        (pedra_x, pedra_y),
        4
    )


    # segunda corda
    angulo2 = direcao - 0.6

    ponta2_x = mao_x + math.cos(angulo2) * 30
    ponta2_y = mao_y + math.sin(angulo2) * 30

    pygame.draw.line(
        tela,
        (90, 50, 20),
        (mao_x, mao_y),
        (ponta2_x, ponta2_y),
        4
    )


    # bolsa da funda
    pygame.draw.circle(
        tela,
        (120, 70, 30),
        (int(pedra_x), int(pedra_y)),
        9
    )


    # pedra
    pygame.draw.circle(
        tela,
        (120, 120, 120),
        (int(pedra_x), int(pedra_y)),
        7
    )
def desenhar_lanca(tela, agente):
    direcao = agente["direcao"]
    comprimento = agente["alcance"]

    x, y = agente["pos"]

    ponta_x = x + math.cos(direcao) * comprimento
    ponta_y = y + math.sin(direcao) * comprimento

    # haste
    pygame.draw.line(
        tela,
        (160, 110, 60),
        (x, y),
        (ponta_x, ponta_y),
        4
    )

    tam_ponta = 20
    largura = 8

    # vetor direção normalizado
    dx = math.cos(direcao)
    dy = math.sin(direcao)

    # vetor perpendicular
    px = -dy
    py = dx

    # ponto da frente
    frente = (
        ponta_x + dx * tam_ponta,
        ponta_y + dy * tam_ponta
    )

    # base da ponta (um pouco atrás da ponta da haste)
    base_x = ponta_x - dx * 4
    base_y = ponta_y - dy * 4

    esquerda = (
        base_x + px * largura,
        base_y + py * largura
    )

    direita = (
        base_x - px * largura,
        base_y - py * largura
    )

    pygame.draw.polygon(tela, (220, 40, 40), [esquerda, direita, frente])
def desenhar_armas_chao(tela):

    for item in armas_no_chao:

        arma = item["arma"]

        if arma == "bastao":
            sprite = imagembastao

        elif arma == "alabarda":
            sprite = imagemalabarda

        elif arma == "lanca":
            sprite = imagelanca

        elif arma == "pistola":
            sprite = imagempistola

        else:
            continue

        rect = sprite.get_rect(
            center=(
                item["pos"][0],
                item["pos"][1]
            )
        )

        tela.blit(sprite, rect)
def desenhar_arma(tela, agente):



    arma = agente.get("arma")
    if arma == "funda":
        desenhar_funda(tela,agente)
        return
    if not agente["atacando"]:
        return
    if arma == "bastao":
        sprite_base = imagembastao
        alcance = agente["alcance"] * 0.6


    elif arma == "alabarda":

        sprite_base = pygame.transform.scale(

            imagemalabarda,

            (

                int(imagemalabarda.get_width() * 0.3),

                int(imagemalabarda.get_height() * 0.3)

            )

        )

        alcance = agente["alcance"] * 0.8

    elif arma == "lanca":
        desenhar_lanca(tela, agente)
        return
    elif arma == 'katana':
        sprite_base = imagemkatana
        alcance = agente["alcance"] * 0.6
    elif arma is None:
        sprite_base = imagem
        alcance = agente["alcance"] * 0.6

    direcao = agente["direcao"]

    offset_x = math.cos(direcao) * alcance
    offset_y = math.sin(direcao) * alcance

    x = agente["pos"][0] + offset_x
    y = agente["pos"][1] + offset_y

    angulo = -math.degrees(direcao) - 90

    sprite = pygame.transform.rotate(sprite_base, angulo)

    rect = sprite.get_rect(center=(x, y))
    tela.blit(sprite, rect)
def atualizar_status_agente(agente):
    # duração da paralisia
    if agente['cosmetico'] != "coroa":
        if agente["paralisia_timer"] > 0:
            agente["paralisia_timer"] -= 1
            agente["paralisado"] = True
        else:
            agente["paralisado"] = False

    # cooldown só importa se ele tiver
    if "paralisia_cd" in agente and agente["paralisia_cd"] > 0:
        agente["paralisia_cd"] -= 1

def criar_circulo(agente):
    pontos = agente['pontos']

    return {
        "ativo":True,
        "pos": agente['pos'],
        "raio": 0,
        "velocidade": 3,
        "max_raio": 120+(pontos*10)
    }
def atualizar_circulos(lista_circulos):
    for circulo in lista_circulos[:]:  # cópia da lista (IMPORTANTE)
        if not circulo["ativo"]:
            lista_circulos.remove(circulo)
            continue

        circulo["raio"] += circulo["velocidade"]

        if circulo["raio"] >= circulo["max_raio"]:
            lista_circulos.remove(circulo)
def desenhar_circulos(tela, lista_circulos):
    global jogadores
    for jogador in jogadores.values():
        desenhar_tela_jogo(jogador)
    for circulo in lista_circulos:
        pygame.draw.circle(
            tela,
            (120, 200, 255),
            (int(circulo["pos"][0]), int(circulo["pos"][1])),
            int(circulo["raio"]),
            3
        )
def obter_dono_coroa(populacao):
    for agente in populacao:
        if agente.get("cosmetico") == "coroa":
            return agente
    return None
def aplicar_efeito_circulos(circulos, populacao):
    dono = obter_dono_coroa(populacao)

    if dono is None:
        return

    for circulo in circulos:
        for agente in populacao:

            # 👑 imunidade do dono
            if agente is dono:
                continue

            dx = agente["pos"][0] - circulo["pos"][0]
            dy = agente["pos"][1] - circulo["pos"][1]
            dist = (dx**2 + dy**2) ** 0.5

            if dist <= circulo["raio"]:
                agente["paralisia_timer"] = 60
def media_movel(dados, janela=10):
    medias = []
    for i in range(len(dados)):
        inicio = max(0, i - janela + 1)
        trecho = dados[inicio:i + 1]
        medias.append(sum(trecho) / len(trecho))
    return medias


clientes = {}
lock = threading.Lock()
def login(cliente, nick):

    if nick not in estadoserver["jogadores"]:

        estadoserver["jogadores"][nick] = copy.deepcopy(
            modelo_jogador
        )

        estadoserver["jogadores"][nick]["nick"] = nick

    clientes[nick] = cliente

    print(f"{nick} entrou no jogo")
def tratar_cliente(cliente, endereco):
    print("Cliente conectado:", endereco)

    buffer = ""

    try:
        while True:
            dados = cliente.recv(5000)
            if not dados:
                break

            buffer += dados.decode()

            while "\n" in buffer:
                mensagem_str, buffer = buffer.split("\n", 1)

                try:
                    mensagem = json.loads(mensagem_str)
                except:
                    continue

                tipo = mensagem.get("tipo")

                # ======================
                # LOGIN
                # ======================
                if tipo == "login":
                    nick = mensagem.get("nick")
                    login(cliente, nick)
                    continue

                # ======================
                # INPUT
                # ======================
                if tipo != "input":
                    continue

                nick = mensagem.get("nick")

                with lock:
                    if nick not in estadoserver["jogadores"]:
                        continue

                    jogador = estadoserver["jogadores"][nick]

                    comandos = mensagem.get("comandos", {})

                    jogador["comandos"] = {
                        "cima": comandos.get("cima", False),
                        "baixo": comandos.get("baixo", False),
                        "esquerda": comandos.get("esquerda", False),
                        "direita": comandos.get("direita", False),
                        "atacar": comandos.get("atacar", False),
                        "olhando_esquerda": comandos.get("olhando_esquerda", False),
                        "olhando_direita": comandos.get("olhando_direita", False),
                        "craftar": comandos.get("craftar", False),
                    }
    except Exception as e:
        print("Erro cliente:", endereco, e)

    finally:
        nick_remover = None

        with lock:
            for nick, sock in clientes.items():
                if sock == cliente:
                    nick_remover = nick
                    break

            if nick_remover:
                clientes.pop(nick_remover, None)
                estadoserver["jogadores"].pop(nick_remover, None)

        cliente.close()
        print(f"{nick_remover} saiu")
def montar_estado():

    with lock:

        obstaculos_rede = [
            {
                "x": obs["rect"].x,
                "y": obs["rect"].y,
                "w": obs["rect"].width,
                "h": obs["rect"].height,
                "tipo": obs["tipo"],
                "health": obs["health"]
            }
            for obs in obstaculos
        ]
        projeteis_rede = [
            {
                "pos": p["pos"],
                "raio": p["raio"]
            }
            for p in projeteis
        ]
        agentes_rede = {
            str(i): {
                "pos": agente["pos"],
                "cor": agente["cor"],
                "tamanho": agente["tamanho"],
                "vida": agente["vida"],
                'atacando':agente["atacando"],
                'direcao':agente["direcao"],
                'arma':agente["arma"],
                'alcance':agente["alcance"],
                'FOV':agente["FOV"],
                'vida_max':agente["vida_max"],
                'stamina':agente["stamina"],
                'stamina_max':agente["stamina_max"],
                "progresso_treino":agente["progresso_treino"],
                "treinando":agente["treinando"]
            }
            for i, agente in enumerate(populacao)
        }

        estado_envio = {
            "jogadores": estadoserver["jogadores"],
            "obstaculos": obstaculos_rede,
            "agentes": agentes_rede,
            "projeteis_rede":projeteis_rede
        }

        return (json.dumps(estado_envio) + "\n").encode()

def loop_broadcast():
    while True:

        estado_json = montar_estado()

        for sock in list(clientes.values()):
            try:
                sock.send(estado_json)
            except:
                pass

        time.sleep(0.05)
cores = ["red", "blue", "yellow"]  # cores para cada agente
rewards_geracao = []
historico = [[] for _ in populacao]
epsilon = 0.1
clock = pygame.time.Clock()
rodando = True
fonte = pygame.font.SysFont(None, 28)
tempo_rodada = 0
TEMPO_MAX_RODADA = 60 * 30  # 60 segundos (ajuste depois)

rodadas = 0
ultimaupdate = 0

tempo = 0
estado_dia = "Dia"

def atualizar_status_entidades(populacao, jogadores):

    entidades = list(populacao) + list(jogadores.values())
    atualizar_projeteis(obstaculos, jogadores)

    for a in entidades:
        pegar_armas_chao(a)
    for jogador in jogadores.values():
        if jogador['vida'] > 0:
            if jogador['tempo_respawn'] > 0:
                jogador['tempo_respawn'] -= 1
        if jogador['tempo_respawn'] <= 0:
            jogador['vivo'] = True
    for entidade in entidades:
        if entidade['confuso']:
            if entidade['confuso'] > 0:
                entidade['confuso'] -= 1
        arma_pistola = entidade.get("arma") == "pistola"
        if entidade['arma'] == 'funda':
            entidade['funda_cooldown'] -= 1

        if arma_pistola:
            # ================= COUNTER TIMERS =================
            if entidade.get("counter_cd", 0) > 0:
                entidade["counter_cd"] -= 1

            if entidade.get("counter_janela", 0) > 0:
                entidade["counter_janela"] -= 1
            # 🔄 atualiza reload só se estiver recarregando
            if entidade["recarregando"]:
                entidade["reload_timer"] -= 1

                if entidade["reload_timer"] <= 0:
                    entidade["balas"] = entidade["max_balas"]
                    entidade["recarregando"] = False
        # ================= COOLDOWN / STUN =================
        if entidade["cooldown_ataque"] > 0:
            entidade["cooldown_ataque"] -= 1

        # ================= TEMPO DE ATAQUE =================
        if not arma_pistola:
            if entidade.get("tempo_ataque", 0) > 0:
                entidade["tempo_ataque"] -= 1
                entidade["atacando"] = True
            else:
                entidade["atacando"] = False
                entidade["acertou_este_ataque"] = False

        # ================= LIMPA ESTADO PISTOLA =================
        if arma_pistola and entidade["cooldown_ataque"] == 0:
            entidade["atacando"] = False

        if entidade.get("stun", 0) > 0:
            entidade["stun"] -= 1

        entidade["vida"] = min(
            entidade["vida_max"],
            entidade["vida"] + entidade["regeneracao"]
        )

        entidade["stamina"] = min(
            entidade["stamina_max"],
            entidade["stamina"] + 0.25
        )

        entidade["velocidade"] = (
            0 if entidade.get("stun", 0) > 0
            else entidade["velocidade_max"]
        )
def treinar_populacao():
    global treinamento_rodando,tempo_rodada,estado_dia
    with lockpopu:
        rewards_geracao = []

        for i, agente in enumerate(populacao):
            agente["treinando"] = True

            rewardmedio = treinar_agente(
                agente=agente

            )

            historico[i].append(rewardmedio)
            rewards_geracao.append(rewardmedio)

        print("Treinamento concluído.")
        for a in populacao:
            a["treinando"] = False
            a["progresso_treino"] = 0

        estado_dia = "Dia"
        treinamento_rodando = False
def funcaosocket():

    HOST = "0.0.0.0"
    PORTA = 5000

    server = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )

    server.bind((HOST, PORTA))
    server.listen()

    print("Servidor aguardando conexões...")

    # 🔥 INICIA BROADCAST UMA VEZ AQUI
    threading.Thread(
        target=loop_broadcast,
        daemon=True
    ).start()

    while True:

        cliente, endereco = server.accept()

        threading.Thread(
            target=tratar_cliente,
            args=(cliente, endereco),
            daemon=True
        ).start()
def tratar_comandos(jogadores):
    for jogador in jogadores.values():
        if jogador["stun"] > 0:
            continue
        if jogador["vida"] <= 0:
            jogador['vivo'] = False
            jogador["tempo_respawn"] = 900
        if jogador["vivo"] == False:
            continue
        comandos = jogador["comandos"]

        if comandos["esquerda"]:
            acao = 1

        elif comandos["direita"]:
            acao = 2

        elif comandos["cima"]:
            acao = 3

        elif comandos["baixo"]:
            acao = 4

        else:
            acao = 0
        if comandos["atacar"]:
            atacante = jogador

            todos = list(populacao) + list(estadoserver["jogadores"].values())

            alvos = [a for a in todos if a is not jogador and a["vida"] > 0]

            if alvos:
                alvo = min(
                    alvos,
                    key=lambda a: math.hypot(
                        a["pos"][0] - atacante["pos"][0],
                        a["pos"][1] - atacante["pos"][1]
                    )
                )

                processar_combate(atacante, alvo, 1)

        mover_agente_por_acao(
            jogador,
            acao,
            obstaculos,
            listargb
        )
        acaogirar = 0
        if comandos['olhando_esquerda'] == True:
            acaogirar = 1
        elif comandos['olhando_direita'] == True:
            acaogirar = 2
        elif comandos['olhando_esquerda'] == True and comandos['olhando_direita'] == True:
            acaogirar = 0
        girar_visao_por_acao(jogador, acaogirar)
        print(comandos)
        if comandos['craftar']==True:
            tentar_crafting(jogador)
        if comandos["atacar"]==True:
            jogadores1 = list(estadoserver["jogadores"].values())

            inimigos_ia = [a for a in populacao if a is not agente and a["vida"] > 0]

            alvos = inimigos_ia + [j for j in jogadores1 if j["vida"] > 0]

            if alvos:
                alvo = min(
                    alvos,
                    key=lambda a: math.hypot(
                        a["pos"][0] - agente["pos"][0],
                        a["pos"][1] - agente["pos"][1]
                    )
                )
                processar_combate(jogador,alvo,1)

def rodar_treino(
    rodando,
    populacao,
    epsilon,
    clock,
    fonte,
    tela,
    tempo_rodada,
    ultimaupdate,
    TEMPO_MAX_RODADA,
    historico,
    obstaculos,
    projeteis,
    rodadas
):
    global rewards_geracao,linha_suave,tempo,treinamento_rodando,estado_dia,jogadores
    threading.Thread(
        target=funcaosocket,
        daemon=True
    ).start()
    while rodando:
        clock.tick(60)
        if estado_dia == "Dia":
            tempo_rodada += 1
        elif estado_dia == "Noite":
            tempo_rodada = 0
        fim_rodada = False

        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:


                rodando = False
        atualizar_circulos(circulos)
        aplicar_efeito_circulos(circulos, populacao)

        # =========================
        # BUILD STATE BATCH (ONCE)
        # =========================
        for agente in populacao:
            estado = obter_estado(
                agente,
                populacao,
                estadoserver["jogadores"],  # 👈 AQUI
                obstaculos,
                14,
                tempo_rodada,
                projeteis
            )
            if len(agente['ultimos_estados']) == 0:
                agente['ultimos_estados'] = [estado, estado, estado]
            else:
                # Se o jogo já estava rodando, segue o seu fluxo padrão de append
                agente['ultimos_estados'].append(estado)
            if len(agente['ultimos_estados']) > 3:
                agente['ultimos_estados'].pop(0)
            vetor_empilhado = np.concatenate(agente['ultimos_estados'])

            agente["estado"] = vetor_empilhado.reshape(-1, 1)


        # =========================
        # FORWARD PASS
        # =========================
        for agente in populacao:
            if agente["treinando"] ==  True:
                continue
            atualizar_status_agente(agente)

            debug_estado(
                agente['estado'],
                agente
            )

            # =====================================
            # FORWARD
            # =====================================

            # =====================================
            # GET ACTION (NOVO PPOAGENT DENSO)
            # =====================================
            # O método já faz o forward interno, amostra a ação (0 a 12), calcula o log_prob e retorna o value
            actions_policy, log_prob, value = agente["rede"].get_action(agente["estado"])

            # batch -> single agent
            action = actions_policy
            agente["acao_pura_ppo"] = action

            # =====================================
            # SAVE
            # =====================================
            # Salvamos os dados necessários para o cálculo do GAE e do train_step do PPO
            # 4. GAMBIARRA: Traduz o ID (0-12) na tupla antiga (mover, ataque, girar)
            agente["acao"] = traduzir_acao_ppo(action, agente)
            agente["log_prob"] = float(
                np.asarray(log_prob).squeeze()
            )

            agente["value"] = float(
                np.asarray(value).squeeze()
            )



        # =========================
        # COMBATE
        # =========================
        for agente in populacao:

            if agente.get("treinando", False):
                continue

            jogadores1 = list(estadoserver["jogadores"].values())

            inimigos_ia = [a for a in populacao if a is not agente and a["vida"] > 0]

            alvos = inimigos_ia + [j for j in jogadores1 if j["vida"] > 0]

            if alvos:
                alvo = min(
                    alvos,
                    key=lambda a: math.hypot(
                        a["pos"][0] - agente["pos"][0],
                        a["pos"][1] - agente["pos"][1]
                    )
                )

                _, ataque, _ = agente["acao"]
                processar_combate(agente, alvo, ataque)



        # =========================
        # MOVIMENTO
        # =========================

        for agente in populacao:

            if agente["treinando"] ==  True:
                continue
            mover, _, girar = agente["acao"]
            if agente["paralisado"] == False:
                mover_agente_por_acao(agente, mover, obstaculos,listargb)
                girar_visao_por_acao(agente, girar)

        tratar_comandos(jogadores)

        # Verifica se sofreu Game Over (Morte)
        for agente in populacao:
            if agente["vida"] <= 0 and agente['morto'] == False:
                # 1. Salva a experiência final de morte
                agente['morto'] = True
                salvar_experiencia(
                    agente=agente,
                    estado=agente["estado"].squeeze(),
                    acao_ppo=agente["acao_pura_ppo"],
                    recompensa=-10 + agente['pontos'],
                    done=True,
                    log_prob=agente["log_prob"],
                    value=agente["value"]
                )

                # 2. Roda o treino IMEDIATAMENTE com os dados que ele coletou até morrer
                print(f"🤖 Agente {agente.get('nome', '')} morreu! Iniciando treino...")
                # 3. Agora sim, ativa a flag para o jogo saber que ele está pronto para reviver ou resetar
                agente["treinando"] = True
                threadtreinoid = threading.Thread(target=treinar_agente, args=(agente,), daemon=True)
                threadtreinoid.start()

            # Salvamento padrão do frame ativo
        for agente in populacao:
            if treinamento_rodando == False:
                salvar_experiencia(
                    agente=agente,
                    estado=agente["estado"].squeeze(),
                    acao_ppo=agente["acao_pura_ppo"],  # Salvamos o ID numérico 0-12
                    recompensa=agente['recompensa_frame'],
                    done=False,
                    log_prob=agente["log_prob"],
                    value=agente["value"]
                )
            else:
                continue
            agente["recompensa_frame"] = 0
            atualizar_efeitos(listargb)
        # =========================
        # END OF EPISODE
        # =========================
        if tempo_rodada >= TEMPO_MAX_RODADA:
            tempo_rodada = 0

            rodadas += 1
            fim_rodada = True
            estado_dia = "Noite"

            ranking = sorted(
                populacao,
                key=lambda a: a["pontos"],
                reverse=True
            )

            for a in populacao:

                if a["pontos"] > 0:
                    a["bonus_final"] = a["pontos"]

                elif a["pontos"] == 0:
                    a["bonus_final"] = 0

                else:
                    a["bonus_final"] = a["pontos"]

            print("\nRecompensas finais da rodada:")

            for a in ranking:
                print(a["bonus_final"])

            for i, agente in enumerate(populacao):
                recompensa_final = agente["bonus_final"]

                print(
                    f"Agente {i + 1} | bônus: {agente['bonus_final']} | "
                    f"temporária: {agente.get('recompensa_temp', 0):.2f} | "
                    f"FINAL: {recompensa_final + agente.get('recompensa_temp', 0):.2f}"
                )

                # =========================================================================
                # EXPERIÊNCIA DE FIM DE EPISÓDIO / FIM DE RODADA
                # =========================================================================
                salvar_experiencia(
                    agente=agente,
                    estado=agente["estado"].squeeze(),
                    acao_ppo=agente["acao_pura_ppo"],  # Salvamos o ID numérico puro 0-12
                    recompensa=recompensa_final,
                    done=fim_rodada,  # Passa se a rodada de fato acabou
                    log_prob=agente["log_prob"],
                    value=agente["value"]
                )



            # ================= TRAINING =================

            for agente in populacao:
                agente["treinando"] = True

            if not treinamento_rodando:
                treinamento_rodando = True
                treinopopu = threading.Thread(target=treinar_populacao,daemon=True)
                treinopopu.start()
            print("Treinamento iniciado em background.")
            obstaculos.extend(
                spawnaleatorioobs(4)
            )


            spawns = spawnaleatorio(3)

            for i, agente in enumerate(populacao):
                agente["vida"] = agente["vida_max"]

                if agente["arma"] != "pistola":
                    agente["arma"] = None

                agente["inventario"] = {}
                agente["recompensa_hit"] = 2
                agente["cooldown_ataque"] = 0
                agente["dano"] = 10
                agente["atacando"] = False
                agente["stamina"] = agente["stamina_max"]
                agente["pontos"] = 0
                agente["recompensa_temp"] = 0
                agente["direcao"] = 0.0
                agente['morto'] = False
                if i < len(spawns):
                    agente["pos"] = spawns[i]


        atualizar_status_entidades(populacao, jogadores)
        # =========================
        # RENDER
        # =========================
        tela.fill(verde)

        for p in projeteis:
            pygame.draw.circle(
                tela,
                (255, 50, 50),
                (int(p["pos"][0]), int(p["pos"][1])),
                p["raio"]
            )


        ranking = sorted(populacao, key=lambda a: a["pontos"], reverse=True)

        y = 10
        for i, agente in enumerate(ranking, start=1):
            texto = fonte.render(
                f"{i}º - {agente['nome']} : {agente['pontos']} pts",
                True,
                agente["cor"]
            )
            tela.blit(texto, (10, y))
            y += 25
        desenhar_armas_chao(tela)
        for a in list(populacao) + list(jogadores.values()):
            a["recompensa_frame"] = 0

            if a["arma"] != "pistola":
                desenhar_arma(tela, a)
        for a in populacao:
            desenhar_agente(a, populacao)
            desenhar_campo_visao_lidar(tela, a, populacao, obstaculos)

        desenhar_cosmeticos(tela, populacao)

        for obs in obstaculos:
            pygame.draw.rect(tela, (200, 60, 60), obs["rect"])

        texto = fonte.render(f"Rodadas: {rodadas}", True, (0, 0, 0))
        rect_texto = texto.get_rect()
        rect_texto.midtop = (LARGURA // 2, 10)
        tela.blit(texto, rect_texto)
        desenhar_circulos(tela,circulos)
        for agente in populacao:
            if agente["nome"] == "vermelho":
                aplicar_distorcao(tela, listargb)

        pygame.display.flip()

    pygame.quit()
    sys.exit()
rodar_treino(rodando,populacao,epsilon,clock,fonte,tela,tempo_rodada,ultimaupdate,TEMPO_MAX_RODADA,historico,obstaculos,projeteis,rodadas)
