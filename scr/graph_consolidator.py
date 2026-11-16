import os
import re
import networkx as nx

# Importa a função de extração criada no módulo anterior
try:
    from graph_extractor import extrair_grafo_semantico
except ImportError:
    # Definição auxiliar se importado isoladamente
    def extrair_grafo_semantico(texto):
        G = nx.DiGraph()
        texto_limpo = re.sub(r'[^\w\s]', '', texto.lower())
        palavras = [p for p in texto_limpo.split() if len(p) > 2]
        for i in range(len(palavras) - 2):
            s, r, o = palavras[i], palavras[i+1], palavras[i+2]
            G.add_edge(s, o, relacao=r)
        return G


def consolidar_grafos(lista_textos, min_peso=1):
    """
    Recebe uma lista de textos de notícias e os consolida em um único 
    Grafo Conceitual de Referência (NetworkX DiGraph).
    
    Cada aresta acumula um 'peso' (frequência de ocorrência da mesma relação entre nós).
    """
    G_consolidado = nx.DiGraph()

    for idx, texto in enumerate(lista_textos):
        g_sub = extrair_grafo_semantico(texto)
        
        for u, v, data in g_sub.edges(data=True):
            relacao = data.get('relacao', 'relacionado')
            
            if G_consolidado.has_edge(u, v):
                # Incrementa peso da relação existente
                G_consolidado[u][v]['peso'] += 1
                if relacao not in G_consolidado[u][v]['relacoes']:
                    G_consolidado[u][v]['relacoes'].append(relacao)
            else:
                # Cria nova aresta com peso 1
                G_consolidado.add_edge(u, v, peso=1, relacoes=[relacao])

    # Filtragem opcional de ruído baseada no peso mínimo
    if min_peso > 1:
        arestas_para_remover = [(u, v) for u, v, d in G_consolidado.edges(data=True) if d['peso'] < min_peso]
        G_consolidado.remove_edges_from(arestas_para_remover)
        # Remove nós isolados resultantes da filtragem
        G_consolidado.remove_nodes_from(list(nx.isolates(G_consolidado)))

    return G_consolidado


def salvar_grafo_consolidado(G, filepath="data/processed/grafo_referencia.graphml"):
    """
    Salva o grafo consolidado em formato GraphML (compatível com Gephi, NetworkX, etc).
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Prepara atributos para serialização em GraphML (listas -> strings)
    G_export = G.copy()
    for u, v, d in G_export.edges(data=True):
        if 'relacoes' in d and isinstance(d['relacoes'], list):
            d['relacoes'] = ", ".join(d['relacoes'])
            
    nx.write_graphml(G_export, filepath)
    print(f"Grafo consolidado salvo em: {filepath}")


def obter_estatisticas_grafo(G):
    """
    Retorna métricas principais do grafo conceitual consolidado.
    """
    return {
        "total_nos": G.number_of_nodes(),
        "total_arestas": G.number_of_edges(),
        "nos_centrais": sorted(G.degree, key=lambda x: x[1], reverse=True)[:5],
        "densidade": nx.density(G)
    }


if __name__ == "__main__":
    # Teste de demonstração da consolidação
    noticias_fake_exemplo = [
        "Vacinas alteram o DNA humano e causam doenças graves segundo especialistas.",
        "Estudo revela que vacina's alteram o sistema imunológico causando doenças ocultas.",
        "Novas vacinas causam danos ao DNA de pacientes em testes recentes."
    ]

    print("=== CONSTRUINDO GRAFO CONSOLIDADO DE REFERÊNCIA ===")
    g_ref = consolidar_grafos(noticias_fake_exemplo)
    stats = obter_estatisticas_grafo(g_ref)
    
    print(f"Total de Nós (Conceitos): {stats['total_nos']}")
    print(f"Total de Arestas (Relações): {stats['total_arestas']}")
    print(f"Nós Mais Conectados (Hubs): {stats['nos_centrais']}")
    print(f"Densidade da Rede: {stats['densidade']:.4f}")
