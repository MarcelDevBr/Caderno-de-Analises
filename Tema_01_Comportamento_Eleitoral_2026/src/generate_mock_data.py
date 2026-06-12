"""
Script para gerar dados sintéticos de eleitores simulando respostas do formulário.
"""
import pandas as pd
import numpy as np
import os

def generate_mock_voters(n_samples=1000):
    """Gera um dataframe com dados sintéticos."""
    np.random.seed(42)
    data = {
        'idade': np.random.choice(['16-24', '25-34', '35-44', '45-59', '60+'], n_samples),
        'faixa_renda': np.random.choice(['Ate 2SM', '2 a 5SM', '5 a 10SM', 'Mais de 10SM'], n_samples),
        'religiao': np.random.choice(['Catolico', 'Evangelico', 'Ateu/Agnostico', 'Outras'], n_samples),
        'bloco_d_estatais': np.random.randint(1, 6, n_samples),
        'bloco_d_tabelamento': np.random.randint(1, 6, n_samples),
        'bloco_e_punitivismo': np.random.randint(1, 6, n_samples),
        'bloco_e_educacao': np.random.randint(1, 6, n_samples),
        'bloco_g_corrupcao': np.random.randint(1, 6, n_samples),
        'bloco_g_pesquisa': np.random.randint(1, 6, n_samples),
        'bloco_g_politica_externa': np.random.randint(1, 6, n_samples),
        'rejeicao_absoluta': np.random.choice(['Lula', 'Flavio Bolsonaro', 'Ciro Gomes', 'Nenhum'], n_samples)
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_mock_voters()
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'mock_voters.csv')
    df.to_csv(output_path, index=False)
    print(f"Dados gerados com sucesso em {output_path}")
