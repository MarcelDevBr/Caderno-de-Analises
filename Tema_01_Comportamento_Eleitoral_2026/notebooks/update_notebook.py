import json
import sys

nb_path = r"c:\Users\marce\Documents\Analises\Caderno-de-Analises\Tema_01_Comportamento_Eleitoral_2026\notebooks\02_cruzamento_espacial_voto.ipynb"

# Create a new notebook from scratch to ensure clean structure
nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
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

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

def create_code_cell(source):
    # Removing the last newline from source if it exists
    lines = [line + "\n" for line in source.split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

nb["cells"] = [
    create_markdown_cell(
        "# Cruzamento Espacial do Voto (VAA) - Perfil Individual\n\n"
        "Neste notebook, aplicamos a Teoria Espacial do Voto de Anthony Downs (1957) para um **único eleitor**.\n"
        "Você insere as respostas do eleitor e o algoritmo mapeia a proximidade dele com os pré-candidatos, "
        "gerando a recomendação (Bússola Eleitoral) baseada na distância matemática e na similaridade de perfil."
    ),
    create_code_cell(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity\n"
        "from sklearn.decomposition import PCA\n"
        "import ipywidgets as widgets\n"
        "from IPython.display import display, HTML\n\n"
        "sns.set_theme(style='whitegrid')"
    ),
    create_markdown_cell(
        "## 1. O Perfil dos Pré-Candidatos (Gabarito)\n"
        "Convertemos as posições públicas de candidatos em coordenadas nos nossos blocos D (Economia) e E (Valores)."
    ),
    create_code_cell(
        "candidatos_data = {\n"
        "    'Candidato': ['Lula', 'Flavio Bolsonaro', 'Romeu Zema', 'Ciro Gomes'],\n"
        "    'bloco_d_estatais': [1, 5, 5, 2],    # 1=Discorda(Estado forte), 5=Concorda(Privatização)\n"
        "    'bloco_d_tabelamento': [4, 1, 1, 3], # 1=Livre Mercado, 5=Intervenção Estatal\n"
        "    'bloco_e_punitivismo': [2, 5, 4, 3], # 1=Garantismo, 5=Punitivismo\n"
        "    'bloco_e_educacao': [2, 5, 4, 2]     # 1=Progressista, 5=Conservador\n"
        "}\n"
        "df_candidatos = pd.DataFrame(candidatos_data).set_index('Candidato')\n"
        "display(df_candidatos)"
    ),
    create_markdown_cell(
        "## 2. Formulário Interativo do Eleitor\n"
        "Preencha o formulário completo (Blocos A a F) para definir o perfil sociodemográfico e ideológico do eleitor."
    ),
    create_code_cell(
        "# Criação dos widgets interativos\n"
        "style = {'description_width': 'initial'}\n\n"
        "nome_widget = widgets.Text(value='Eleitor Exemplo', description='Nome:', style=style)\n\n"
        
        "display(HTML('<h3>Bloco A - Demografia e Clivagens Sociais</h3>'))\n"
        "idade_widget = widgets.IntText(value=30, description='Idade:', style=style)\n"
        "renda_widget = widgets.Dropdown(options=['Até 2 salários mínimos', '2 a 5 salários mínimos', 'Mais de 5 salários mínimos'], description='Faixa de Renda:', style=style)\n"
        "escolaridade_widget = widgets.Dropdown(options=['Ensino Fundamental', 'Ensino Médio', 'Ensino Superior'], description='Escolaridade:', style=style)\n"
        "regiao_widget = widgets.Dropdown(options=['Norte', 'Nordeste', 'Centro-Oeste', 'Sudeste', 'Sul'], description='Região:', style=style)\n"
        "religiao_widget = widgets.Dropdown(options=['Católica', 'Evangélica', 'Outra', 'Sem Religião'], description='Religião:', style=style)\n"
        "beneficio_widget = widgets.Dropdown(options=['Sim', 'Não'], description='Recebe Benefício Social?', style=style)\n\n"
        
        "display(HTML('<h3>Bloco B - Capital Cultural</h3>'))\n"
        "informacao_widget = widgets.Dropdown(options=['TV Aberta', 'Redes Sociais (Instagram/TikTok)', 'Aplicativos de Mensagem (WhatsApp/Telegram)', 'Mídia Independente', 'Rádio'], description='Principal Fonte de Informação:', style=style)\n\n"
        
        "display(HTML('<h3>Bloco C - Matriz de Valores</h3>'))\n"
        "prioridade_widget = widgets.Dropdown(options=['Controle da Inflação', 'Geração de Empregos', 'Combate ao Crime Violento', 'Preservação Ambiental', 'Combate às Desigualdades', 'Defesa da Família Tradicional', 'Liberdade de Expressão'], description='Prioridade Absoluta:', style=style)\n\n"
        
        "display(HTML('<h3>Bloco D - Economia (Eixo X)</h3>'))\n"
        "estatais_widget = widgets.IntSlider(\n"
        "    value=3, min=1, max=5, step=1,\n"
        "    description='Privatização de Estatais (1=Estado Forte, 5=Privatização Total):',\n"
        "    style=style, layout=widgets.Layout(width='80%')\n"
        ")\n"
        "tabelamento_widget = widgets.IntSlider(\n"
        "    value=3, min=1, max=5, step=1,\n"
        "    description='Controle de Preços (1=Livre Mercado, 5=Intervenção Estatal):',\n"
        "    style=style, layout=widgets.Layout(width='80%')\n"
        ")\n\n"
        "display(HTML('<h3>Bloco E - Valores e Moral (Eixo Y)</h3>'))\n"
        "punitivismo_widget = widgets.IntSlider(\n"
        "    value=3, min=1, max=5, step=1,\n"
        "    description='Segurança Pública (1=Ressocialização, 5=Punição Severa):',\n"
        "    style=style, layout=widgets.Layout(width='80%')\n"
        ")\n"
        "educacao_widget = widgets.IntSlider(\n"
        "    value=3, min=1, max=5, step=1,\n"
        "    description='Educação (1=Progressista, 5=Conservadora):',\n"
        "    style=style, layout=widgets.Layout(width='80%')\n"
        ")\n\n"
        "display(HTML('<h3>Bloco F - Decisão e Polarização</h3>'))\n"
        "opcoes_rejeicao = ['Nenhum'] + list(df_candidatos.index)\n"
        "rejeicao_widget = widgets.Dropdown(\n"
        "    options=opcoes_rejeicao,\n"
        "    value='Nenhum',\n"
        "    description='Rejeição Absoluta (Não votaria de jeito nenhum):',\n"
        "    style=style, layout=widgets.Layout(width='50%')\n"
        ")\n\n"
        "# Agrupando e exibindo o formulário completo\n"
        "form = widgets.VBox([\n"
        "    nome_widget, widgets.HTML('<hr>'),\n"
        "    idade_widget, renda_widget, escolaridade_widget, regiao_widget, religiao_widget, beneficio_widget, widgets.HTML('<hr>'),\n"
        "    informacao_widget, widgets.HTML('<hr>'),\n"
        "    prioridade_widget, widgets.HTML('<hr>'),\n"
        "    estatais_widget, tabelamento_widget, widgets.HTML('<hr>'),\n"
        "    punitivismo_widget, educacao_widget, widgets.HTML('<hr>'),\n"
        "    rejeicao_widget\n"
        "])\n"
        "display(form)"
    ),
    create_markdown_cell(
        "## 3. O Motor VAA: Calculando Proximidade e Afastamento\n"
        "O algoritmo captura as respostas do formulário acima, calcula a Distância Euclidiana baseada nas coordenadas ideológicas e aplica a penalidade caso exista Rejeição Absoluta.\n\n"
        "*Execute esta célula APÓS preencher o formulário acima.*"
    ),
    create_code_cell(
        "# Lendo os valores do formulário interativo\n"
        "eleitor_nome = nome_widget.value\n"
        "estatais = estatais_widget.value\n"
        "tabelamento = tabelamento_widget.value\n"
        "punitivismo = punitivismo_widget.value\n"
        "educacao = educacao_widget.value\n"
        "rejeicao_absoluta = rejeicao_widget.value\n\n"
        "vetor_eleitor = np.array([[estatais, tabelamento, punitivismo, educacao]])\n"
        "print(f\"Perfil capturado do Eleitor '{eleitor_nome}': {vetor_eleitor[0]}\\n\")\n\n"
        "print(f\"Demografia: {idade_widget.value} anos, {regiao_widget.value}, {renda_widget.value}, Benefício: {beneficio_widget.value}\")\n"
        "print(f\"Valores: {prioridade_widget.value} | Informação: {informacao_widget.value}\\n\")\n\n"
        "vetores_candidatos = df_candidatos.values\n\n"
        "# --- Distância Euclidiana ---\n"
        "# Quanto MENOR, mais próximo o eleitor está do candidato nas suas opiniões.\n"
        "dist_euclidiana = euclidean_distances(vetor_eleitor, vetores_candidatos)[0]\n\n"
        "resultados = pd.DataFrame({\n"
        "    'Candidato': df_candidatos.index,\n"
        "    'Distancia_Euclidiana': dist_euclidiana\n"
        "})\n\n"
        "# Aplicar filtro de Veto por Polarização Afetiva (Rejeição Absoluta)\n"
        "resultados['Veto_Aplicado'] = resultados['Candidato'].apply(lambda x: True if x == rejeicao_absoluta else False)\n"
        "resultados.loc[resultados['Veto_Aplicado'] == True, 'Distancia_Euclidiana'] = np.inf # Penalidade infinita\n\n"
        "# Ordenar pelo mais próximo (menor distância)\n"
        "resultados = resultados.sort_values(by='Distancia_Euclidiana').reset_index(drop=True)\n\n"
        "print(f\">>> RESULTADO DA BÚSSOLA ELEITORAL PARA: {eleitor_nome} <<<\")\n"
        "display(resultados)\n\n"
        "cand_ideal = resultados.iloc[0]['Candidato']\n"
        "if resultados.iloc[0]['Distancia_Euclidiana'] == np.inf:\n"
        "    print(\"\\nO eleitor rejeitou absolutamente todos os candidatos ou ocorreu um erro no cálculo.\")\n"
        "else:\n"
        "    print(f\"\\nO candidato com maior alinhamento técnico às respostas é: **{cand_ideal}**\")"
    ),
    create_markdown_cell(
        "## 4. Visualização 2D (Bússola Eleitoral)\n"
        "Comprimimos as 4 dimensões de opiniões em 2 usando PCA para que possamos enxergar a posição do eleitor no mapa político."
    ),
    create_code_cell(
        "pca = PCA(n_components=2)\n\n"
        "# Juntar o eleitor com os candidatos para o mapa\n"
        "vetores_totais = np.vstack([df_candidatos.values, vetor_eleitor])\n"
        "proj_2d = pca.fit_transform(vetores_totais)\n\n"
        "proj_cand = proj_2d[:len(df_candidatos)]\n"
        "proj_eleit = proj_2d[-1] # Último é o eleitor\n\n"
        "plt.figure(figsize=(10, 8))\n\n"
        "# Plotar candidatos\n"
        "cores_cand = ['red', 'blue', 'orange', 'purple']\n"
        "for i, (idx, row) in enumerate(df_candidatos.iterrows()):\n"
        "    plt.scatter(proj_cand[i, 0], proj_cand[i, 1], color=cores_cand[i], s=300, edgecolors='black', label=idx, marker='s')\n"
        "    plt.annotate(idx, (proj_cand[i, 0], proj_cand[i, 1]), xytext=(8, -8), textcoords='offset points')\n\n"
        "# Plotar o Eleitor\n"
        "plt.scatter(proj_eleit[0], proj_eleit[1], color='green', s=400, edgecolors='black', label=f\"Eleitor: {eleitor_nome}\", marker='*')\n"
        "plt.annotate(f\"{eleitor_nome}\", (proj_eleit[0], proj_eleit[1]), xytext=(8, 8), textcoords='offset points', fontweight='bold')\n\n"
        "plt.title(f'Bússola Eleitoral 2026 - Mapa Político (PCA)', fontsize=14)\n"
        "plt.xlabel('Eixo Principal 1 (Maior variação de opiniões)')\n"
        "plt.ylabel('Eixo Principal 2')\n"
        "plt.axhline(0, color='gray', linestyle='--', alpha=0.5)\n"
        "plt.axvline(0, color='gray', linestyle='--', alpha=0.5)\n"
        "plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')\n"
        "plt.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully without nbformat.")
