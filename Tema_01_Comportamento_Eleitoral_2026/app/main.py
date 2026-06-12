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
    
    if st.button("Calcular Proximidade"):
        st.success("Cálculo realizado com sucesso! (Em breve: mapa 2D interativo)")
        
if __name__ == "__main__":
    main()
