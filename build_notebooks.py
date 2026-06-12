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
    lines = [line + "\n" for line in code.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

# --- NOTEBOOK 01 ---
nb1_cells = [
    md_cell("# Caderno 01: Análise Exploratória e Clivagens Sociais\n\nNeste notebook, realizamos a Análise Exploratória de Dados (EDA) primária do dataset de respostas da Bússola Eleitoral.\n\n## Referencial Teórico: Seymour Martin Lipset e Stein Rokkan (1967)\nA teoria das **Clivagens Sociais** argumenta que o comportamento eleitoral é moldado por fraturas estruturais profundas na sociedade (ex: Centro vs Periferia, Estado vs Igreja, Capital vs Trabalho). Antes de olharmos para a opinião do indivíduo, precisamos olhar para as \"caixas\" demográficas em que ele está inserido. A demografia frequentemente antecede a ideologia.\n\nNossa análise focará em desvendar essas fraturas (Renda, Religião, Idade) dentro da nossa amostra."),
    code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\n# Configurações estéticas do Seaborn\nsns.set_theme(style='whitegrid', palette='muted')\nplt.rcParams['figure.figsize'] = (10, 6)"),
    md_cell("## 1. Carregamento da Amostra Bruta"),
    code_cell("df = pd.read_csv('../data/raw/mock_voters.csv')\ndisplay(df.head())\nprint(f'Tamanho da amostra: {df.shape[0]} eleitores')"),
    md_cell("## 2. Radiografia Demográfica\nAnalisamos a composição da nossa base. Desvios muito grandes em relação à sociedade real (excesso de jovens ou alta renda) exigirão ponderação estatística."),
    code_cell("fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n\nsns.countplot(data=df, x='idade', order=['16-24', '25-34', '35-44', '45-59', '60+'], ax=axes[0])\naxes[0].set_title('Distribuição por Idade')\n\nsns.countplot(data=df, x='faixa_renda', order=['Ate 2SM', '2 a 5SM', '5 a 10SM', 'Mais de 10SM'], ax=axes[1])\naxes[1].set_title('Distribuição por Renda')\n\nsns.countplot(data=df, x='religiao', ax=axes[2])\naxes[2].set_title('Distribuição por Religião')\n\nplt.tight_layout()\nplt.show()"),
    md_cell("## 3. Ponderação Amostral (Post-Stratification)\n**Problema**: Pesquisas de internet (opt-in) possuem **viés de seleção**. Normalmente amostras online super-representam jovens escolarizados.\n\n**Solução**: Atribuímos \"pesos amostrais\" a cada respondente. Se um jovem responde, seu peso será < 1; se um idoso de baixa escolaridade responde, seu peso será > 1, corrigindo matematicamente a pirâmide para que ela reflita os dados reais do TSE e IBGE."),
    code_cell("# Exemplo simulado de cálculo de peso amostral baseado em idade.\n# Supondo que a população real (TSE) seja 20% jovens (16-24), mas nossa amostra tem 30%.\n\namostra_idade = df['idade'].value_counts(normalize=True)\ntse_idade = {'16-24': 0.15, '25-34': 0.20, '35-44': 0.25, '45-59': 0.25, '60+': 0.15}\n\npesos_idade = {k: tse_idade[k] / amostra_idade[k] for k in tse_idade.keys()}\ndf['peso_amostral'] = df['idade'].map(pesos_idade)\nprint('Pesos calculados por faixa etária:')\nprint(pesos_idade)")
]

# --- NOTEBOOK 02 ---
nb2_cells = [
    md_cell("# Caderno 02: Cruzamento Espacial Estático e Gabaritos\n\n## Referencial Teórico: Anthony Downs (1957)\nNo livro clássico *\"An Economic Theory of Democracy\"*, Downs introduziu a **Teoria Espacial do Voto**. Ele sugere que eleitores e partidos podem ser alocados em um espaço ideológico (tradicionalmente um eixo linear Esquerda-Direita, mas expansível para N-dimensões).\n\nO eleitor racional votará no candidato que estiver **geometricamente mais próximo** de si neste espaço. Se o eleitor está no ponto X, e os candidatos estão nos pontos A e B, o cálculo da distância determinará o voto.\n\nNeste caderno, faremos exatamente isso usando vetores matemáticos e Distância Euclidiana em todo o nosso DataSet bruto."),
    code_cell("import pandas as pd\nimport numpy as np\nfrom sklearn.metrics.pairwise import euclidean_distances, cosine_similarity\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.decomposition import PCA\nsns.set_theme(style='whitegrid')"),
    md_cell("## 1. O Mapa de Ideologias (Partidos vs Políticos)\nO sistema lida com o fenômeno em que as ações de um político podem desviar do estatuto original de seu partido. Por isso, vetorizamos ambas as entidades em 7 dimensões."),
    code_cell("partidos_data = {\n    'Partido': ['PT', 'PL', 'NOVO', 'PDT'],\n    'ideologia': ['Social-Democracia', 'Centrão/Fisiológico', 'Liberalismo Clássico', 'Trabalhismo'],\n    'justificativa': ['Foco no estado de bem-estar social', 'Foco na governabilidade', 'Foco em estado mínimo', 'Defesa da indústria nacional'],\n    'bloco_d_estatais': [2, 4, 5, 2],\n    'bloco_d_tabelamento': [3, 2, 1, 3],\n    'bloco_e_punitivismo': [2, 4, 3, 3],\n    'bloco_e_educacao': [2, 4, 3, 2],\n    'bloco_g_corrupcao': [2, 4, 5, 2],\n    'bloco_g_pesquisa': [4, 2, 1, 5],\n    'bloco_g_politica_externa': [4, 2, 1, 4]\n}\ndf_partidos = pd.DataFrame(partidos_data).set_index('Partido')\n\npoliticos_data = {\n    'Político': ['Lula', 'Flavio Bolsonaro', 'Romeu Zema', 'Ciro Gomes'],\n    'ideologia': ['Pragmatismo Progressista', 'Direita Radical', 'Neoliberalismo', 'Nacional-Desenvolvimentismo'],\n    'justificativa': ['Conciliação de pautas', 'Pauta de costumes', 'Livre mercado', 'Estado forte na economia'],\n    'bloco_d_estatais': [1, 5, 5, 2],\n    'bloco_d_tabelamento': [4, 1, 1, 3],\n    'bloco_e_punitivismo': [2, 5, 4, 3],\n    'bloco_e_educacao': [2, 5, 4, 2],\n    'bloco_g_corrupcao': [1, 5, 5, 2],\n    'bloco_g_pesquisa': [5, 1, 2, 5],\n    'bloco_g_politica_externa': [5, 1, 2, 4]\n}\ndf_politicos = pd.DataFrame(politicos_data).set_index('Político')\n\ncols_opiniao = ['bloco_d_estatais', 'bloco_d_tabelamento', 'bloco_e_punitivismo', 'bloco_e_educacao', 'bloco_g_corrupcao', 'bloco_g_pesquisa', 'bloco_g_politica_externa']\ndisplay(df_partidos[cols_opiniao])\ndisplay(df_politicos[cols_opiniao])"),
    md_cell("## 2. A Matemática: Distância Euclidiana\nA fórmula da distância euclidiana mede o comprimento do segmento de reta entre o eleitor e o candidato no hiper-espaço de 7 dimensões. Quanto **menor** a distância, maior a chance de voto."),
    code_cell("df_eleitores = pd.read_csv('../data/raw/mock_voters.csv')\nvetores_eleitores = df_eleitores[cols_opiniao].head(5).values\n\ndist_pol = euclidean_distances(vetores_eleitores, df_politicos[cols_opiniao].values)\ndist_part = euclidean_distances(vetores_eleitores, df_partidos[cols_opiniao].values)\n\nprint('Distância Euclidiana (Amostra de 5 Eleitores x Políticos):')\nprint(pd.DataFrame(dist_pol, columns=df_politicos.index))\n\n# O Argmin revela a menor distância (A recomendação de fato)\neleitores_recomendados = [df_politicos.index[i] for i in np.argmin(dist_pol, axis=1)]\nprint('\\nRecomendação Puramente Matemática para os 5 primeiros:', eleitores_recomendados)")
]

# --- NOTEBOOK 03 ---
nb3_cells = [
    md_cell("# Caderno 03: A Bússola Eleitoral Interativa (VAA)\n\nAs *Voting Advice Applications* (VAA), como a Bússola Eleitoral holandesa ou o Wahl-O-Mat alemão, são ferramentas digitais que automatizam a lógica espacial para o eleitor em tempo real.\n\nNeste caderno, implementamos a versão interativa da teoria apresentada no Caderno 02. Preencha os formulários e descubra o seu alinhamento individual."),
    code_cell("import pandas as pd\nimport numpy as np\nfrom sklearn.metrics.pairwise import euclidean_distances\nfrom sklearn.decomposition import PCA\nimport matplotlib.pyplot as plt\nimport ipywidgets as widgets\nfrom IPython.display import display, HTML"),
    code_cell(
        "partidos_data = {\n"
        "    'Partido': ['PT', 'PL', 'NOVO', 'PDT'],\n"
        "    'ideologia': ['Social-Democracia', 'Centrão/Fisiológico', 'Liberalismo Clássico', 'Trabalhismo'],\n"
        "    'justificativa': ['Partido de massas com foco no estado de bem-estar social', 'Foco na governabilidade', 'Foco em desregulamentação e estado mínimo', 'Defesa da indústria nacional e do trabalhador'],\n"
        "    'bloco_d_estatais': [2, 4, 5, 2],\n"
        "    'bloco_d_tabelamento': [3, 2, 1, 3],\n"
        "    'bloco_e_punitivismo': [2, 4, 3, 3],\n"
        "    'bloco_e_educacao': [2, 4, 3, 2],\n"
        "    'bloco_g_corrupcao': [2, 4, 5, 2],\n"
        "    'bloco_g_pesquisa': [4, 2, 1, 5],\n"
        "    'bloco_g_politica_externa': [4, 2, 1, 4]\n"
        "}\n"
        "df_partidos = pd.DataFrame(partidos_data).set_index('Partido')\n\n"
        "politicos_data = {\n"
        "    'Político': ['Lula', 'Flavio Bolsonaro', 'Romeu Zema', 'Ciro Gomes'],\n"
        "    'ideologia': ['Pragmatismo Progressista', 'Direita Radical', 'Neoliberalismo', 'Nacional-Desenvolvimentismo'],\n"
        "    'justificativa': ['Conciliação de classes e pautas sociais', 'Pauta de costumes conservadora e punitivismo', 'Livre mercado estrito e alianças de conveniência', 'Estado forte na economia e crítica ao rentismo'],\n"
        "    'bloco_d_estatais': [1, 5, 5, 2],\n"
        "    'bloco_d_tabelamento': [4, 1, 1, 3],\n"
        "    'bloco_e_punitivismo': [2, 5, 4, 3],\n"
        "    'bloco_e_educacao': [2, 5, 4, 2],\n"
        "    'bloco_g_corrupcao': [1, 5, 5, 2],\n"
        "    'bloco_g_pesquisa': [5, 1, 2, 5],\n"
        "    'bloco_g_politica_externa': [5, 1, 2, 4]\n"
        "}\n"
        "df_politicos = pd.DataFrame(politicos_data).set_index('Político')\n"
        "cols_opiniao = ['bloco_d_estatais', 'bloco_d_tabelamento', 'bloco_e_punitivismo', 'bloco_e_educacao', 'bloco_g_corrupcao', 'bloco_g_pesquisa', 'bloco_g_politica_externa']\n"
    ),
    code_cell(
        "# Interface (Widgets)\n"
        "style = {'description_width': 'initial'}\n"
        "nome_widget = widgets.Text(value='Eleitor', description='Nome:', style=style)\n"
        "estatais_widget = widgets.IntSlider(value=3, min=1, max=5, description='Privatização de Estatais (1=Estado Forte, 5=Privatização Total):', style=style, layout=widgets.Layout(width='80%'))\n"
        "tabelamento_widget = widgets.IntSlider(value=3, min=1, max=5, description='Controle de Preços (1=Livre Mercado, 5=Intervenção Estatal):', style=style, layout=widgets.Layout(width='80%'))\n"
        "punitivismo_widget = widgets.IntSlider(value=3, min=1, max=5, description='Segurança Pública (1=Ressocialização, 5=Punição Severa):', style=style, layout=widgets.Layout(width='80%'))\n"
        "educacao_widget = widgets.IntSlider(value=3, min=1, max=5, description='Educação (1=Progressista, 5=Conservadora):', style=style, layout=widgets.Layout(width='80%'))\n"
        "corrupcao_widget = widgets.IntSlider(value=3, min=1, max=5, description='Corrupção (1=Sistêmica, 5=Punitiva):', style=style, layout=widgets.Layout(width='80%'))\n"
        "pesquisa_widget = widgets.IntSlider(value=3, min=1, max=5, description='Pesquisa (1=Privada, 5=Estado):', style=style, layout=widgets.Layout(width='80%'))\n"
        "politica_externa_widget = widgets.IntSlider(value=3, min=1, max=5, description='Pol. Externa (1=EUA/Ocid., 5=Sul Global):', style=style, layout=widgets.Layout(width='80%'))\n"
        "rejeicao_widget = widgets.Dropdown(options=['Nenhum'] + list(df_politicos.index), value='Nenhum', description='Rejeição Absoluta:', style=style)\n"
        "display(widgets.VBox([nome_widget, estatais_widget, tabelamento_widget, punitivismo_widget, educacao_widget, corrupcao_widget, pesquisa_widget, politica_externa_widget, rejeicao_widget]))\n"
    ),
    md_cell("Execute a célula abaixo APÓS preencher os dados acima para ver seu cálculo interativo e mapa PCA."),
    code_cell(
        "vetor_eleitor = np.array([[estatais_widget.value, tabelamento_widget.value, punitivismo_widget.value, educacao_widget.value, corrupcao_widget.value, pesquisa_widget.value, politica_externa_widget.value]])\n\n"
        "dist_partidos = euclidean_distances(vetor_eleitor, df_partidos[cols_opiniao].values)[0]\n"
        "dist_politicos = euclidean_distances(vetor_eleitor, df_politicos[cols_opiniao].values)[0]\n\n"
        "res_partidos = pd.DataFrame({'Partido': df_partidos.index, 'Distancia': dist_partidos}).sort_values(by='Distancia').reset_index(drop=True)\n"
        "res_politicos = pd.DataFrame({'Político': df_politicos.index, 'Distancia': dist_politicos})\n"
        "res_politicos['Veto_Aplicado'] = res_politicos['Político'].apply(lambda x: True if x == rejeicao_widget.value else False)\n"
        "res_politicos.loc[res_politicos['Veto_Aplicado'] == True, 'Distancia'] = np.inf\n"
        "res_politicos = res_politicos.sort_values(by='Distancia').reset_index(drop=True)\n\n"
        "cand_ideal = res_politicos.iloc[0]['Político']\n"
        "part_ideal = res_partidos.iloc[0]['Partido']\n"
        "print(f\"\\n🏆 Político Ideal: {cand_ideal} | Ideologia: {df_politicos.loc[cand_ideal, 'ideologia']}\")\n"
        "print(f\"🏛️ Partido Ideal: {part_ideal} | Ideologia: {df_partidos.loc[part_ideal, 'ideologia']}\")\n\n"
        "# PCA Map\n"
        "pca = PCA(n_components=2)\n"
        "vetores_totais = np.vstack([df_politicos[cols_opiniao].values, df_partidos[cols_opiniao].values, vetor_eleitor])\n"
        "proj = pca.fit_transform(vetores_totais)\n"
        "plt.figure(figsize=(8,6))\n"
        "plt.scatter(proj[:4, 0], proj[:4, 1], c='red', s=200, label='Políticos', marker='o')\n"
        "plt.scatter(proj[4:8, 0], proj[4:8, 1], c='blue', s=200, label='Partidos', marker='s', alpha=0.5)\n"
        "plt.scatter(proj[-1, 0], proj[-1, 1], c='green', s=300, label='VOCÊ', marker='*')\n"
        "for i, txt in enumerate(df_politicos.index): plt.annotate(txt, (proj[i,0], proj[i,1]))\n"
        "for i, txt in enumerate(df_partidos.index): plt.annotate(txt, (proj[i+4,0], proj[i+4,1]), alpha=0.6)\n"
        "plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')\n"
        "plt.title('Mapa Político (PCA)')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
]

# --- NOTEBOOK 04 ---
nb4_cells = [
    md_cell("# Caderno 04: Polarização Afetiva e Dissonância Cognitiva\n\n## Referencial Teórico: Iyengar et al. (2012) e a Polarização Afetiva\nA ciência política contemporânea descobriu que os eleitores hoje não se guiam apenas pelo amor à sua própria \"tribo\", mas pelo ódio ou medo profundo da \"tribo oposta\". Isso se chama **Polarização Afetiva**.\n\nQuando traduzimos isso para a matemática do Voto Espacial, encontramos o que chamamos de **Dissonância Cognitiva no VAA**. Ocorre quando o eleitor, sem saber o gabarito cego, responde ao questionário com respostas que indicam uma proximidade com o candidato X, mas, por força da Polarização Afetiva, ele marcou o candidato X como \"Rejeição Absoluta\".\n\nNeste caderno, analisamos quantos eleitores da nossa base estão sofrendo de Dissonância Cognitiva."),
    code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.metrics.pairwise import euclidean_distances\n\nsns.set_theme(style='whitegrid')"),
    md_cell("## 1. Mapeamento da Identidade Negativa\nA pergunta da rejeição absoluta captura exatamente quem o eleitor considera o \"inimigo\"."),
    code_cell("df = pd.read_csv('../data/raw/mock_voters.csv')\nplt.figure(figsize=(10, 5))\nsns.countplot(data=df, y='rejeicao_absoluta', order=df['rejeicao_absoluta'].value_counts().index, palette='Reds_r', hue='rejeicao_absoluta', legend=False)\nplt.title('Índice de Rejeição Absoluta')\nplt.show()"),
    md_cell("## 2. Aplicando o Veto e Achando a Dissonância"),
    code_cell(
        "politicos_data = {\n"
        "    'Político': ['Lula', 'Flavio Bolsonaro', 'Romeu Zema', 'Ciro Gomes'],\n"
        "    'bloco_d_estatais': [1, 5, 5, 2],\n"
        "    'bloco_d_tabelamento': [4, 1, 1, 3],\n"
        "    'bloco_e_punitivismo': [2, 5, 4, 3],\n"
        "    'bloco_e_educacao': [2, 5, 4, 2],\n"
        "    'bloco_g_corrupcao': [1, 5, 5, 2],\n"
        "    'bloco_g_pesquisa': [5, 1, 2, 5],\n"
        "    'bloco_g_politica_externa': [5, 1, 2, 4]\n"
        "}\n"
        "df_cand = pd.DataFrame(politicos_data).set_index('Político')\n"
        "cols_opiniao = ['bloco_d_estatais', 'bloco_d_tabelamento', 'bloco_e_punitivismo', 'bloco_e_educacao', 'bloco_g_corrupcao', 'bloco_g_pesquisa', 'bloco_g_politica_externa']\n\n"
        "dist_matriz = euclidean_distances(df[cols_opiniao].values, df_cand[cols_opiniao].values)\n"
        "df['candidato_matematico_ideal'] = [df_cand.index[i] for i in np.argmin(dist_matriz, axis=1)]\n\n"
        "def recomendacao_com_veto(row, dist_row):\n"
        "    rejeicao = row['rejeicao_absoluta']\n"
        "    dist_corrigida = dist_row.copy()\n"
        "    if rejeicao in df_cand.index:\n"
        "        idx_rejeitado = df_cand.index.get_loc(rejeicao)\n"
        "        dist_corrigida[idx_rejeitado] = np.inf\n"
        "    return df_cand.index[np.argmin(dist_corrigida)]\n\n"
        "df['candidato_final'] = [recomendacao_com_veto(row, dist_matriz[i]) for i, row in df.iterrows()]\n"
        "df['dissonancia'] = df['candidato_matematico_ideal'] != df['candidato_final']\n\n"
        "print(f\"{df['dissonancia'].mean() * 100:.2f}% dos eleitores possuem dissonância cognitiva (rejeitam quem é tecnicamente mais parecido com eles).\")"
    )
]

if __name__ == "__main__":
    base_dir = r"c:/Users/marce/Documents/Analises/Caderno-de-Analises/Tema_01_Comportamento_Eleitoral_2026/notebooks"
    
    with open(os.path.join(base_dir, "01_exploracao_demografica.ipynb"), "w", encoding="utf-8") as f:
        json.dump(create_notebook(nb1_cells), f, ensure_ascii=False, indent=2)
        
    with open(os.path.join(base_dir, "02_cruzamento_espacial_estatico.ipynb"), "w", encoding="utf-8") as f:
        json.dump(create_notebook(nb2_cells), f, ensure_ascii=False, indent=2)
        
    with open(os.path.join(base_dir, "03_cruzamento_espacial_interativo.ipynb"), "w", encoding="utf-8") as f:
        json.dump(create_notebook(nb3_cells), f, ensure_ascii=False, indent=2)
        
    with open(os.path.join(base_dir, "04_polarizacao_afetiva.ipynb"), "w", encoding="utf-8") as f:
        json.dump(create_notebook(nb4_cells), f, ensure_ascii=False, indent=2)
    
    print("Notebooks 01 a 04 gerados com sucesso!")
