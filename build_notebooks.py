import json
import os

def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }

# --- NOTEBOOK 01 ---
nb1_cells = [
    md_cell("# Análise Exploratória: Perfil e Dores do Eleitorado\n\nNeste notebook, realizamos a Análise Exploratória de Dados (EDA) primária do dataset de respostas da Bússola Eleitoral.\n\n## Objetivos:\n1. Compreender a distribuição demográfica da amostra.\n2. Identificar correlações iniciais entre clivagens (Renda, Religião, Idade) e os blocos temáticos.\n3. Estabelecer as bases para a Ponderação Amostral (Post-Stratification)."),
    code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\n# Configurações estéticas do Seaborn\nsns.set_theme(style='whitegrid', palette='muted')\nplt.rcParams['figure.figsize'] = (10, 6)"),
    md_cell("## 1. Carregamento dos Dados\nImportamos os dados sintéticos gerados pelo módulo de mock_data."),
    code_cell("df = pd.read_csv('../data/raw/mock_voters.csv')\ndisplay(df.head())\nprint(f'Tamanho da amostra: {df.shape[0]} eleitores')"),
    md_cell("## 2. Clivagens Sociais: Demografia\nA teoria das clivagens de Lipset e Rokkan argumenta que traços estruturais moldam o voto. Vamos analisar a composição da nossa amostra."),
    code_cell("fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n\nsns.countplot(data=df, x='idade', order=['16-24', '25-34', '35-44', '45-59', '60+'], ax=axes[0])\naxes[0].set_title('Distribuição por Idade')\n\nsns.countplot(data=df, x='faixa_renda', order=['Ate 2SM', '2 a 5SM', '5 a 10SM', 'Mais de 10SM'], ax=axes[1])\naxes[1].set_title('Distribuição por Renda')\n\nsns.countplot(data=df, x='religiao', ax=axes[2])\naxes[2].set_title('Distribuição por Religião')\n\nplt.tight_layout()\nplt.show()"),
    md_cell("## 3. Distribuição de Opiniões (Eixos Econômico e Valores)\nAnalisamos agora as respostas (em escala Likert de 1 a 5) sobre o papel do Estado (Bloco D) e Valores/Punitivismo (Bloco E)."),
    code_cell("cols_opiniao = ['bloco_d_estatais', 'bloco_d_tabelamento', 'bloco_e_punitivismo', 'bloco_e_educacao']\n\nfig, axes = plt.subplots(2, 2, figsize=(14, 10))\naxes = axes.flatten()\n\nfor i, col in enumerate(cols_opiniao):\n    sns.histplot(df[col], bins=5, kde=True, ax=axes[i], color='royalblue')\n    axes[i].set_title(col)\n    axes[i].set_xticks([1, 2, 3, 4, 5])\n\nplt.tight_layout()\nplt.show()"),
    md_cell("## 4. Ponderação Amostral (Post-Stratification)\n**Teoria**: Pesquisas de internet possuem viés de seleção. Normalmente amostras online têm excesso de jovens e alta renda em comparação ao eleitorado real aferido pelo TSE.\n\nPara corrigir isso, atribuímos \"pesos\" a cada respondente. Se um jovem responde, seu peso será menor (pois já temos muitos); se um idoso responde, seu peso será maior (para equilibrar a balança)."),
    code_cell("# Exemplo simulado de cálculo de peso amostral baseado em idade.\n# Supondo que a população real (TSE) seja 20% jovens (16-24), mas nossa amostra tem 30%.\n\n# Distribuição da Amostra\namostra_idade = df['idade'].value_counts(normalize=True)\n\n# Distribuição Fictícia do TSE\ntse_idade = {'16-24': 0.15, '25-34': 0.20, '35-44': 0.25, '45-59': 0.25, '60+': 0.15}\n\n# Calculando pesos (TSE / Amostra)\npesos_idade = {k: tse_idade[k] / amostra_idade[k] for k in tse_idade.keys()}\n\ndf['peso_amostral'] = df['idade'].map(pesos_idade)\nprint('Pesos calculados por faixa etária:')\nprint(pesos_idade)\ndisplay(df[['idade', 'peso_amostral']].head())")
]

# --- NOTEBOOK 02 ---
nb2_cells = [
    md_cell("# Cruzamento Espacial do Voto (VAA)\n\nNeste notebook, aplicamos a Teoria Espacial do Voto de Anthony Downs (1957). Mapeamos eleitores e candidatos como pontos em um espaço multidimensional e calculamos as distâncias matemáticas entre eles.\n\n## Objetivos\n1. Definir o \"Gabarito\" dos pré-candidatos.\n2. Calcular Distância Euclidiana e Similaridade por Cosseno.\n3. Projetar os dados em 2D usando PCA."),
    code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.metrics.pairwise import euclidean_distances, cosine_similarity\nfrom sklearn.decomposition import PCA\n\nsns.set_theme(style='whitegrid')"),
    md_cell("## 1. Definindo o Perfil dos Pré-Candidatos\nPara fins matemáticos, convertemos as posições públicas de candidatos em coordenadas nos nossos blocos D (Economia) e E (Valores).\n\nA classificação utiliza uma **Escala Likert de 1 a 5** definida da seguinte forma:\n\n**Bloco D: O Papel do Estado e a Economia**\n- `bloco_d_estatais`: 1 = Estado Forte (Discorda da venda) | 5 = Privatização Total\n- `bloco_d_tabelamento`: 1 = Livre Mercado | 5 = Intervenção Estatal (Controle de preços)\n\n**Bloco E: Valores Morais e Punitivismo**\n- `bloco_e_punitivismo`: 1 = Garantismo (Foco em ressocialização) | 5 = Punitivismo (Leis mais duras)\n- `bloco_e_educacao`: 1 = Progressista | 5 = Conservador"),
    code_cell("candidatos_data = {\n    'Candidato': ['Lula', 'Flavio Bolsonaro', 'Romeu Zema', 'Ciro Gomes'],\n    'bloco_d_estatais': [1, 5, 5, 2],    # 1=Discorda(Estado forte), 5=Concorda(Privatização)\n    'bloco_d_tabelamento': [4, 1, 1, 3], # 1=Livre Mercado, 5=Intervenção Estatal\n    'bloco_e_punitivismo': [2, 5, 4, 3], # 1=Garantismo, 5=Punitivismo\n    'bloco_e_educacao': [2, 5, 4, 2]     # 1=Progressista, 5=Conservador\n}\ndf_candidatos = pd.DataFrame(candidatos_data).set_index('Candidato')\ndisplay(df_candidatos)"),
    md_cell("## 2. Vetorizando os Eleitores e Calculando Proximidade\nVamos importar nosso dataset e extrair apenas as colunas numéricas de opinião para formar o vetor do eleitor. Faremos o cálculo para os 5 primeiros eleitores como demonstração."),
    code_cell("df_eleitores = pd.read_csv('../data/raw/mock_voters.csv')\ncols_opiniao = ['bloco_d_estatais', 'bloco_d_tabelamento', 'bloco_e_punitivismo', 'bloco_e_educacao']\n\nvetores_eleitores = df_eleitores[cols_opiniao].head(5).values\nvetores_candidatos = df_candidatos.values\n\n# --- 2.1 Distância Euclidiana ---\n# Quanto MENOR, mais próximo\ndist_euclidiana = euclidean_distances(vetores_eleitores, vetores_candidatos)\n\n# --- 2.2 Similaridade Cosseno ---\n# Quanto MAIOR (perto de 1), mais similar a direção\nsim_cosseno = cosine_similarity(vetores_eleitores, vetores_candidatos)\n\nprint('Distância Euclidiana (Eleitor 0 aos Candidatos):')\nprint(pd.Series(dist_euclidiana[0], index=df_candidatos.index).sort_values())\n\nprint('\\nSimilaridade Cosseno (Eleitor 0 aos Candidatos):')\nprint(pd.Series(sim_cosseno[0], index=df_candidatos.index).sort_values(ascending=False))"),
    md_cell("## 3. Redução de Dimensionalidade (PCA)\nNosso espaço tem 4 dimensões (as 4 perguntas). Para visualizar num gráfico 2D, aplicamos o Principal Component Analysis (PCA) para comprimir as informações preservando a maior variância possível."),
    code_cell("pca = PCA(n_components=2)\n\n# Concatenar candidatos e todos os eleitores para treinar o mesmo espaço dimensional\nvetores_totais = np.vstack([df_candidatos.values, df_eleitores[cols_opiniao].values])\nproj_2d = pca.fit_transform(vetores_totais)\n\n# Separar as projeções\nproj_cand = proj_2d[:len(df_candidatos)]\nproj_eleit = proj_2d[len(df_candidatos):]\n\nplt.figure(figsize=(10, 8))\n# Plotar amostra de eleitores (fundo)\nplt.scatter(proj_eleit[:, 0], proj_eleit[:, 1], alpha=0.15, color='gray', label='Eleitores')\n\n# Plotar candidatos (destaque)\ncores_cand = ['red', 'blue', 'orange', 'purple']\nfor i, (idx, row) in enumerate(df_candidatos.iterrows()):\n    plt.scatter(proj_cand[i, 0], proj_cand[i, 1], color=cores_cand[i], s=200, edgecolors='black', label=idx)\n\nplt.title('Bússola Eleitoral 2026: Projeção 2D (PCA) de Candidatos e Eleitores')\nplt.xlabel('Componente Principal 1 (Econômico predominante)')\nplt.ylabel('Componente Principal 2 (Valores predominante)')\nplt.legend()\nplt.show()")
]

# --- NOTEBOOK 03 ---
nb3_cells = [
    md_cell("# Análise de Rejeição e Dissonância Cognitiva\n\nEste notebook explora o conceito de **Polarização Afetiva**. Em democracias polarizadas, o \"voto negativo\" (rejeição absoluta a um candidato ou partido) frequentemente supera a convergência ideológica.\n\n## Objetivos\n1. Mapear o índice de Rejeição Absoluta.\n2. Aplicar um \"Filtro de Veto\" ao modelo matemático de recomendação.\n3. Identificar Dissonância Cognitiva (quando a recomendação ideal sofre veto do eleitor)."),
    code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.metrics.pairwise import euclidean_distances\n\nsns.set_theme(style='whitegrid')"),
    md_cell("## 1. Mapeamento da Rejeição Absoluta\nA pergunta \"Quem você considera uma ameaça e NÃO votaria sob nenhuma hipótese?\" captura essa identidade negativa extrema."),
    code_cell("df = pd.read_csv('../data/raw/mock_voters.csv')\n\nplt.figure(figsize=(10, 5))\nsns.countplot(data=df, y='rejeicao_absoluta', order=df['rejeicao_absoluta'].value_counts().index, palette='Reds_r')\nplt.title('Índice de Rejeição Absoluta dos Eleitores')\nplt.xlabel('Quantidade de Eleitores')\nplt.ylabel('Candidato Rejeitado')\nplt.show()"),
    md_cell("## 2. O Paradoxo da Polarização (Dissonância Cognitiva)\nVamos simular o cenário onde identificamos o \"Candidato Ideal\" de um eleitor matematicamente (menor Distância Euclidiana). A Dissonância Cognitiva ocorre se esse Candidato Ideal for EXATAMENTE a pessoa que o eleitor declarou \"Rejeição Absoluta\"."),
    code_cell("candidatos_data = {\n    'Candidato': ['Lula', 'Flavio Bolsonaro', 'Romeu Zema', 'Ciro Gomes'],\n    'bloco_d_estatais': [1, 5, 5, 2],\n    'bloco_d_tabelamento': [4, 1, 1, 3],\n    'bloco_e_punitivismo': [2, 5, 4, 3],\n    'bloco_e_educacao': [2, 5, 4, 2]\n}\ndf_cand = pd.DataFrame(candidatos_data).set_index('Candidato')\ncols_opiniao = ['bloco_d_estatais', 'bloco_d_tabelamento', 'bloco_e_punitivismo', 'bloco_e_educacao']\n\nvetores_cand = df_cand.values\n\n# Calcular a recomendação puramente matemática (Sem Filtro) para toda a base\ndist_matriz = euclidean_distances(df[cols_opiniao].values, vetores_cand)\ndf['candidato_matematico_ideal'] = [df_cand.index[i] for i in np.argmin(dist_matriz, axis=1)]\n\ndisplay(df[['candidato_matematico_ideal', 'rejeicao_absoluta']].head(10))"),
    md_cell("## 3. Aplicando o Filtro de Veto\nSe o candidato_matematico_ideal == rejeicao_absoluta, o algoritmo VAA não pode recomendá-lo. Ele deve aplicar uma penalidade (distância infinita) e buscar a 2ª melhor opção."),
    code_cell("def recomendacao_com_veto(row, dist_row):\n    rejeicao = row['rejeicao_absoluta']\n    \n    # Cria uma cópia das distâncias do eleitor para os 4 candidatos\n    dist_corrigida = dist_row.copy()\n    \n    # Se o candidato rejeitado estiver na matriz, aplicamos veto (distância infinita)\n    if rejeicao in df_cand.index:\n        idx_rejeitado = df_cand.index.get_loc(rejeicao)\n        dist_corrigida[idx_rejeitado] = np.inf\n        \n    # Retorna o índice do menor valor APÓS o veto\n    idx_recomendado = np.argmin(dist_corrigida)\n    return df_cand.index[idx_recomendado]\n\n# Aplicando a lógica de veto\ndf['candidato_recomendado_final'] = [recomendacao_com_veto(row, dist_matriz[i]) for i, row in df.iterrows()]\n\n# Calculando quantos sofreram dissonância cognitiva\ndf['dissonancia'] = df['candidato_matematico_ideal'] != df['candidato_recomendado_final']\n\ntaxa_dissonancia = df['dissonancia'].mean() * 100\nprint(f'{taxa_dissonancia:.2f}% do eleitorado possui uma inclinação política inconsciente para o candidato que eles mesmos rejeitam.')\n\nplt.figure(figsize=(6, 6))\nplt.pie(df['dissonancia'].value_counts(), labels=['Voto Consistente', 'Dissonância / Veto Aplicado'], autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])\nplt.title('Impacto da Polarização Afetiva no Recomendador')\nplt.show()")
]

if __name__ == "__main__":
    base_dir = r"c:/Users/marce/Documents/Analises/Caderno-de-Analises/Tema_01_Comportamento_Eleitoral_2026/notebooks"
    
    with open(os.path.join(base_dir, "01_exploracao_perfil_e_dores.ipynb"), "w", encoding="utf-8") as f:
        json.dump(create_notebook(nb1_cells), f, ensure_ascii=False, indent=2)
        
    with open(os.path.join(base_dir, "02_cruzamento_espacial_voto.ipynb"), "w", encoding="utf-8") as f:
        json.dump(create_notebook(nb2_cells), f, ensure_ascii=False, indent=2)
        
    with open(os.path.join(base_dir, "03_analise_rejeicao_engajamento.ipynb"), "w", encoding="utf-8") as f:
        json.dump(create_notebook(nb3_cells), f, ensure_ascii=False, indent=2)
    
    print("Notebooks gerados com sucesso!")
