import re
import networkx as nx
import spacy

# Carrega o sentencizador para Português
nlp = spacy.blank("pt")
nlp.add_pipe("sentencizer")

ARTIGOS_STOPWORDS = {"o", "a", "os", "as", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas"}
VERBOS_IGNORAR = {"é", "são", "foi", "foram", "ser", "estar", "tem", "têm", "haver", "há", "segundo"}

def limpar_termo(termo):
    """Limpa pontuações, artigos e converte para minúsculas para padronizar o nó do grafo."""
    termo = re.sub(r'[^\w\s]', '', termo).lower().strip()
    palavras = termo.split()
    while palavras and palavras[0] in ARTIGOS_STOPWORDS:
        palavras.pop(0)
    return " ".join(palavras)

def extrair_triplas_texto(texto):
    """
    Extrai triplas semânticas (Sujeito, Relação/Verbo, Objeto) de frases em português.
    """
    doc = nlp(texto)
    triplas = []

    for sentena in doc.sents:
        frase = sentena.text.strip()
        # Divide orações compostas ligadas por vírgulas ou 'e'
        subfrases = re.split(r'[,;]|\s+e\s+|\s+mas\s+', frase)

        for sub in subfrases:
            tokens = [t.strip() for t in sub.split() if t.strip()]
            if len(tokens) < 3:
                continue

            # Busca por padrões de Verbos de ação comuns no Português
            match = re.search(r'(.+?)\s+([a-zà-ú]+(?:am|em|ou|ar|er|ir|am|ram|am))\s+(.+)', sub, re.IGNORECASE)
            if match:
                suj_raw, rel_raw, obj_raw = match.groups()
                suj = limpar_termo(suj_raw)
                rel = rel_raw.lower().strip()
                obj = limpar_termo(obj_raw)

                if rel not in VERBOS_IGNORAR and len(suj) > 1 and len(obj) > 1:
                    triplas.append((suj, rel, obj))

    return triplas

def texto_para_grafo_semantico(texto, id_noticia="noticia_01"):
    """
    Converte o texto de uma notícia em um Grafo Dirigido (networkx.DiGraph).
    """
    G = nx.DiGraph(id=id_noticia)
    triplas = extrair_triplas_texto(texto)

    for suj, rel, obj in triplas:
        G.add_node(suj, tipo="entidade")
        G.add_node(obj, tipo="entidade")
        G.add_edge(suj, obj, relacao=rel)

    return G

def calcular_similaridade_grafos(G1, G2):
    """
    Calcula a similaridade estrutural e de conteúdo entre dois grafos semânticos.
    Retorna uma métrica composta entre 0.0 e 1.0.
    """
    nos_g1 = set(G1.nodes())
    nos_g2 = set(G2.nodes())

    if not nos_g1 or not nos_g2:
        return 0.0

    intersecao_nos = nos_g1.intersection(nos_g2)
    uniao_nos = nos_g1.union(nos_g2)
    sim_nos = len(intersecao_nos) / len(uniao_nos)

    arestas_g1 = set((u, v, d.get('relacao', '')) for u, v, d in G1.edges(data=True))
    arestas_g2 = set((u, v, d.get('relacao', '')) for u, v, d in G2.edges(data=True))

    intersecao_arestas = arestas_g1.intersection(arestas_g2)
    uniao_arestas = arestas_g1.union(arestas_g2)
    sim_arestas = len(intersecao_arestas) / len(uniao_arestas) if uniao_arestas else 0.0

    return round((0.6 * sim_nos) + (0.4 * sim_arestas), 4)

if __name__ == "__main__":
    exemplo_fake = "Vacinas alteram o DNA humano. Notícias falsas causam doenças graves."
    exemplo_ref = "Vacinas estimulam o sistema imunológico. Vacinas previnem doenças graves."

    g_fake = texto_para_grafo_semantico(exemplo_fake, "fake_01")
    g_ref = texto_para_grafo_semantico(exemplo_ref, "ref_01")

    print("=== TESTE DO MÓDULO GRAPH EXTRACTOR ===")
    print("Grafos construídos com sucesso!")
    print(f"Grafo Fake: {len(g_fake.nodes())} nós, {len(g_fake.edges())} arestas")
    print(f"Grafo Ref:  {len(g_ref.nodes())} nós, {len(g_ref.edges())} arestas")
    print(f"Similaridade Estrutural: {calcular_similaridade_grafos(g_fake, g_ref)}")
