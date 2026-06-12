"""
Aplicação Streamlit da Bússola Eleitoral 2026.
"""
import streamlit as st
import pandas as pd
import numpy as np

def main():
    st.set_page_config(page_title="Bússola Eleitoral 2026", layout="wide")
    st.title("Bússola Eleitoral 2026 - Caderno de Análises")
    
    st.write("Bem-vindo à Bússola Eleitoral. Responda ao questionário abaixo para descobrir sua proximidade com os pré-candidatos à presidência.")
    
    # Exemplo de bloco do questionário
    st.header("Bloco D: O Papel do Estado e a Economia")
    estatais = st.slider("Empresas estatais que dão prejuízo devem ser vendidas para a iniciativa privada.", 1, 5, 3)
    tabelamento = st.slider("O governo deve tabelar preços de alimentos básicos.", 1, 5, 3)

    st.header("Bloco E: Valores Morais e Punitivismo")
    punitivismo = st.slider("Segurança Pública (1=Ressocialização, 5=Punição Severa)", 1, 5, 3)
    educacao = st.slider("Educação (1=Progressista, 5=Conservadora)", 1, 5, 3)

    st.header("Bloco G: Temas Emergentes")
    corrupcao = st.slider("Corrupção (1=Sistêmica, 5=Punitiva)", 1, 5, 3)
    pesquisa = st.slider("Pesquisa (1=Iniciativa Privada, 5=Estado Estratégico)", 1, 5, 3)
    politica_externa = st.slider("Política Externa (1=Alinhamento EUA, 5=Sul Global)", 1, 5, 3)
    politica_externa = st.slider("Política Externa (1=Alinhamento EUA, 5=Sul Global)", 1, 5, 3)
    
    opcoes_rejeicao = ['Nenhum', 'Lula', 'Flavio Bolsonaro', 'Romeu Zema', 'Ciro Gomes']
    rejeicao_absoluta = st.selectbox("Rejeição Absoluta (Não votaria de jeito nenhum):", opcoes_rejeicao)
    
    if st.button("Calcular Bússola"):
        from sklearn.metrics.pairwise import euclidean_distances
        
        partidos_data = {
            'Partido': ['PT', 'PL', 'NOVO', 'PDT'],
            'ideologia': ['Social-Democracia', 'Centrão/Fisiológico', 'Liberalismo Clássico', 'Trabalhismo'],
            'justificativa': ['Partido de massas com foco no estado de bem-estar social', 'Foco na governabilidade', 'Foco em desregulamentação e estado mínimo', 'Defesa da indústria nacional e do trabalhador'],
            'bloco_d_estatais': [2, 4, 5, 2],
            'bloco_d_tabelamento': [3, 2, 1, 3],
            'bloco_e_punitivismo': [2, 4, 3, 3],
            'bloco_e_educacao': [2, 4, 3, 2],
            'bloco_g_corrupcao': [2, 4, 5, 2],
            'bloco_g_pesquisa': [4, 2, 1, 5],
            'bloco_g_politica_externa': [4, 2, 1, 4]
        }
        df_partidos = pd.DataFrame(partidos_data).set_index('Partido')
        
        politicos_data = {
            'Político': ['Lula', 'Flavio Bolsonaro', 'Romeu Zema', 'Ciro Gomes'],
            'ideologia': ['Pragmatismo Progressista', 'Direita Radical', 'Neoliberalismo', 'Nacional-Desenvolvimentismo'],
            'justificativa': ['Conciliação de classes e pautas sociais', 'Pauta de costumes conservadora e punitivismo', 'Livre mercado estrito e alianças de conveniência', 'Estado forte na economia e crítica ao rentismo'],
            'bloco_d_estatais': [1, 5, 5, 2],
            'bloco_d_tabelamento': [4, 1, 1, 3],
            'bloco_e_punitivismo': [2, 5, 4, 3],
            'bloco_e_educacao': [2, 5, 4, 2],
            'bloco_g_corrupcao': [1, 5, 5, 2],
            'bloco_g_pesquisa': [5, 1, 2, 5],
            'bloco_g_politica_externa': [5, 1, 2, 4]
        }
        df_politicos = pd.DataFrame(politicos_data).set_index('Político')
        cols_opiniao = ['bloco_d_estatais', 'bloco_d_tabelamento', 'bloco_e_punitivismo', 'bloco_e_educacao', 'bloco_g_corrupcao', 'bloco_g_pesquisa', 'bloco_g_politica_externa']
        
        vetor_eleitor = np.array([[estatais, tabelamento, punitivismo, educacao, corrupcao, pesquisa, politica_externa]])
        vetores_partidos = df_partidos[cols_opiniao].values
        vetores_politicos = df_politicos[cols_opiniao].values
        
        dist_partidos = euclidean_distances(vetor_eleitor, vetores_partidos)[0]
        dist_politicos = euclidean_distances(vetor_eleitor, vetores_politicos)[0]
        
        res_partidos = pd.DataFrame({'Partido': df_partidos.index, 'Distancia': dist_partidos}).sort_values(by='Distancia').reset_index(drop=True)
        res_politicos = pd.DataFrame({'Político': df_politicos.index, 'Distancia': dist_politicos})
        
        res_politicos['Veto_Aplicado'] = res_politicos['Político'].apply(lambda x: True if x == rejeicao_absoluta else False)
        res_politicos.loc[res_politicos['Veto_Aplicado'] == True, 'Distancia'] = np.inf
        res_politicos = res_politicos.sort_values(by='Distancia').reset_index(drop=True)
        
        cand_ideal = res_politicos.iloc[0]['Político']
        part_ideal = res_partidos.iloc[0]['Partido']
        
        st.markdown("---")
        st.header("🎯 Resultado da Bússola Eleitoral")
        
        if res_politicos.iloc[0]['Distancia'] == np.inf:
            st.error("O eleitor rejeitou absolutamente todos os candidatos possíveis!")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Político Ideal:** {cand_ideal}")
                st.write(f"**Ideologia:** {df_politicos.loc[cand_ideal, 'ideologia']}")
                st.write(f"_{df_politicos.loc[cand_ideal, 'justificativa']}_")
                
                st.markdown("#### Pontos de Atenção (Político):")
                for col in cols_opiniao:
                    diff = abs(df_politicos.loc[cand_ideal, col] - eval(col.replace('bloco_d_', '').replace('bloco_e_', '').replace('bloco_g_', '')))
                    if diff <= 1:
                        st.write(f"✅ Concordância em: {col}")
                    elif diff >= 3:
                        st.write(f"❌ Discordância em: {col}")
                        
            with col2:
                st.info(f"**Partido Ideal:** {part_ideal}")
                st.write(f"**Ideologia:** {df_partidos.loc[part_ideal, 'ideologia']}")
                st.write(f"_{df_partidos.loc[part_ideal, 'justificativa']}_")
                
        st.markdown("---")
        st.header("👤 Classificação do Eleitor")
        media_economia = (estatais + tabelamento) / 2
        media_valores = (punitivismo + educacao) / 2
        if media_economia < 3 and media_valores < 3:
            perfil = 'Esquerda Progressista'
        elif media_economia > 3 and media_valores > 3:
            perfil = 'Direita Conservadora'
        elif media_economia > 3 and media_valores < 3:
            perfil = 'Liberalismo/Libertarianismo'
        else:
            perfil = 'Centro/Moderado'
        st.write(f"Baseado em seus eixos principais, o algoritmo classifica você com um viés de: **{perfil}**")

if __name__ == "__main__":
    main()
