import random
import numpy as np
import networkx as nx

"""
Esqueleto Estruturado: Minimização de Fake News por Algoritmos Evolucionários (Python)

Este projeto demonstra duas abordagens principais citadas na literatura:
1. Abordagem de Classificação (Otimização de Pesos em Atributos Textuais).
2. Abordagem de Grafo (Seleção Estratégica de Observadores Vacinados em Redes Sociais).
"""

# ==========================================
# PARÂMETROS GERAIS DO PROJETO
# ==========================================
SEED = 42
POPULATION_SIZE = 50     # Tamanho da população (50 a 100 recomendado)
GENERATIONS = 30         # Número de gerações para o ciclo evolutivo
CX_PROB = 0.8            # Probabilidade de cruzamento (crossover)
MUT_PROB = 0.2           # Probabilidade de mutação
TOURNAMENT_SIZE = 3      # Tamanho do torneio para seleção

random.seed(SEED)
np.random.seed(SEED)


# ==============================================================================
# ABORDAGEM 1: OTIMIZAÇÃO DE PESOS PARA CLASSIFICAÇÃO DE CONTEÚDO
# ==============================================================================

def gerar_dataset_prototipo(num_noticias=500, num_atributos=10):
    """
    Cria uma matriz de atributos sintética (500 instâncias x 10 atributos)
    simulando TF-IDF, estatísticas de pontuação e sentimento.
    """
    X = np.random.rand(num_noticias, num_atributos)
    # Rótulo binário: 1 (Fake News), 0 (Notícia Verdadeira)
    y = np.random.randint(0, 2, size=num_noticias)
    return X, y


def calcular_fitness_classificador(individuo, X, y):
    """
    Avalia a acurácia de um vetor de pesos (individuo) ao classificar as notícias.
    """
    pesos = np.array(individuo)
    pontuacoes = np.dot(X, pesos)
    predicoes = (pontuacoes > 0.0).astype(int)
    acuracia = np.mean(predicoes == y)
    return acuracia


def criar_populacao_pesos(tam_pop, num_atributos):
    """Cria a população inicial de vetores de pesos reais (-1.0 a 1.0)."""
    return [[random.uniform(-1.0, 1.0) for _ in range(num_atributos)] for _ in range(tam_pop)]


def selecao_torneio(populacao, fitnesses, k=TOURNAMENT_SIZE):
    """Seleção por torneio entre k indivíduos aleatórios."""
    selecionados = random.sample(list(zip(populacao, fitnesses)), k)
    selecionados.sort(key=lambda x: x[1], reverse=True)
    return [gene for gene in selecionados[0][0]]


def cruzamento_ponto_duplo(pai1, pai2):
    """Realiza o cruzamento de dois pontos entre dois indivíduos."""
    if len(pai1) < 2 or random.random() > CX_PROB:
        return pai1.copy(), pai2.copy()
    pt1, pt2 = sorted(random.sample(range(len(pai1)), 2))
    filho1 = pai1[:pt1] + pai2[pt1:pt2] + pai1[pt2:]
    filho2 = pai2[:pt1] + pai1[pt1:pt2] + pai2[pt2:]
    return filho1, filho2


def mutacao_gaussiana(individuo, mut_rate=MUT_PROB, sigma=0.2):
    """Aplica mutação gaussiana nos genes do indivíduo."""
    for i in range(len(individuo)):
        if random.random() < mut_rate:
            individuo[i] += random.gauss(0, sigma)
            # Manter valores dentro dos limites [-1.0, 1.0]
            individuo[i] = max(-1.0, min(1.0, individuo[i]))
    return individuo


def rodar_algoritmo_classificacao(X, y):
    """Executa o ciclo evolucionário completo para o classificador de notícias."""
    num_atributos = X.shape[1]
    populacao = criar_populacao_pesos(POPULATION_SIZE, num_atributos)
    
    melhor_fitness_historico = []
    melhor_individuo = None
    melhor_fitness = -1.0
    
    print("\n[MÓDULO 1] Executando Evolução para Otimização de Pesos...")
    for gen in range(GENERATIONS):
        fitnesses = [calcular_fitness_classificador(ind, X, y) for ind in populacao]
        
        # Atualizar o melhor indivíduo
        max_idx = np.argmax(fitnesses)
        if fitnesses[max_idx] > melhor_fitness:
            melhor_fitness = fitnesses[max_idx]
            melhor_individuo = populacao[max_idx].copy()
            
        melhor_fitness_historico.append(melhor_fitness)
        
        # Gerar nova população
        nova_populacao = []
        # Elitismo: preserva o melhor indivíduo para a próxima geração
        nova_populacao.append(melhor_individuo.copy())
        
        while len(nova_populacao) < POPULATION_SIZE:
            pai1 = selecao_torneio(populacao, fitnesses)
            pai2 = selecao_torneio(populacao, fitnesses)
            f1, f2 = cruzamento_ponto_duplo(pai1, pai2)
            nova_populacao.append(mutacao_gaussiana(f1))
            if len(nova_populacao) < POPULATION_SIZE:
                nova_populacao.append(mutacao_gaussiana(f2))
                
        populacao = nova_populacao
        if (gen + 1) % 10 == 0 or gen == 0:
            print(f"  Geração {gen+1:02d}/{GENERATIONS} | Melhor Acurácia (Fitness): {melhor_fitness:.4f}")
            
    return melhor_individuo, melhor_fitness


# ==============================================================================
# ABORDAGEM 2: SELEÇÃO DE OBSERVADORES EM GRAFOS (PROPAGAÇÃO DE REDE)
# ==============================================================================

def criar_grafo_rede_social(num_nos=100):
    """
    Cria um grafo de rede livre de escala (Barabási-Albert) simulando conexões sociais.
    """
    grafo = nx.barabasi_albert_graph(n=num_nos, m=3, seed=SEED)
    return grafo


def simular_propagacao_si(grafo, no_semente=0, observadores=set(), passos_tempo=10):
    """
    Simula o modelo epidemiológico SI (Susceptible-Infected) em um grafo com observadores vacinados.
    Observadores vacinados bloqueiam a retransmissão da fake news.
    Retorna a taxa final de nós infectados (quanto menor, melhor a contenção).
    """
    infectados = {no_semente}
    suscetiveis = set(grafo.nodes()) - infectados
    
    for _ in range(passos_tempo):
        novos_infectados = set()
        for no in infectados:
            # Se o nó for um observador vacinado, ele interrompe a propagação aos vizinhos
            if no in observadores:
                continue
            for vizinho in grafo.neighbors(no):
                if vizinho in suscetiveis and random.random() < 0.3: # probabilidade de infecção
                    novos_infectados.add(vizinho)
        if not novos_infectados:
            break
        infectados.update(novos_infectados)
        suscetiveis.difference_update(novos_infectados)
        
    taxa_infeccao = len(infectados) / len(grafo.nodes())
    return taxa_infeccao


def calcular_fitness_observadores(individuo, grafo, k_observadores=10):
    """
    Aptidão para minimização de contágio em redes:
    Maximiza a contenção (1 - taxa_infeccao).
    """
    observadores = set(individuo[:k_observadores])
    taxa_infeccao = simular_propagacao_si(grafo, no_semente=0, observadores=observadores, passos_tempo=8)
    fitness_contencao = 1.0 - taxa_infeccao
    return fitness_contencao


def rodar_algoritmo_observadores(grafo, k_observadores=10):
    """Evolui uma população de conjuntos de nós observadores para conter a desinformação."""
    nos_disponiveis = list(grafo.nodes())
    populacao = [random.sample(nos_disponiveis, k_observadores) for _ in range(POPULATION_SIZE)]
    
    melhor_fitness = -1.0
    melhores_observadores = None
    
    print("\n[MÓDULO 2] Executando Evolução para Seleção de Observadores no Grafo...")
    for gen in range(GENERATIONS):
        fitnesses = [calcular_fitness_observadores(ind, grafo, k_observadores) for ind in populacao]
        
        max_idx = np.argmax(fitnesses)
        if fitnesses[max_idx] > melhor_fitness:
            melhor_fitness = fitnesses[max_idx]
            melhores_observadores = populacao[max_idx].copy()
            
        nova_populacao = [melhores_observadores.copy()]
        
        while len(nova_populacao) < POPULATION_SIZE:
            pai1 = selecao_torneio(populacao, fitnesses)
            pai2 = selecao_torneio(populacao, fitnesses)
            
            # Cruzamento por combinação de conjuntos de nós
            corte = random.randint(1, k_observadores - 1)
            filho = list(set(pai1[:corte] + pai2[corte:]))
            
            # Preencher se houver duplicadas
            while len(filho) < k_observadores:
                novo_no = random.choice(nos_disponiveis)
                if novo_no not in filho:
                    filho.append(novo_no)
            
            # Mutação: substitui um nó observador aleatoriamente
            if random.random() < MUT_PROB:
                idx_sub = random.randint(0, k_observadores - 1)
                novo_no = random.choice(nos_disponiveis)
                filho[idx_sub] = novo_no
                
            nova_populacao.append(filho[:k_observadores])
            
        populacao = nova_populacao
        if (gen + 1) % 10 == 0 or gen == 0:
            print(f"  Geração {gen+1:02d}/{GENERATIONS} | Taxa de Contenção (Fitness): {melhor_fitness*100:.2f}%")
            
    return melhores_observadores, melhor_fitness


# ==============================================================================
# BLOCO PRINCIPAL DE EXECUÇÃO
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  MINIMIZAÇÃO DE FAKE NEWS POR ALGORITMOS EVOLUCIONÁRIOS EM PYTHON")
    print("=" * 70)
    
    # Executar Módulo 1: Classificação de Conteúdo
    X, y = gerar_dataset_prototipo(num_noticias=500, num_atributos=10)
    pesos_otimizados, acuracia_final = rodar_algoritmo_classificacao(X, y)
    
    # Executar Módulo 2: Contenção em Grafo de Rede Social
    grafo = criar_grafo_rede_social(num_nos=100)
    observadores_otimizados, contencao_final = rodar_algoritmo_observadores(grafo, k_observadores=10)
    
    print("\n" + "=" * 70)
    print("  RESUMO DOS RESULTADOS DA PROTOTIPAGEM")
    print("=" * 70)
    print(f"✔ Classificação Textual: Acurácia máxima otimizada = {acuracia_final * 100:.2f}%")
    print(f"✔ Pesos Finais Otimizados: {np.round(pesos_otimizados[:5], 3)} ... (total: {len(pesos_otimizados)} atributos)")
    print(f"✔ Contenção em Grafo: Taxa de contenção de propagação = {contencao_final * 100:.2f}%")
    print(f"✔ Observadores Estratégicos Selecionados (Nós do Grafo): {observadores_otimizados}")
    print("=" * 70)
