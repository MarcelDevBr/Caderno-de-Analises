import pandas as pd
import urllib.request
import json
import sqlite3
import os
import time

def fetch_bcb(series_id, start_date='01/01/1990', end_date='31/12/2026'):
    url = f'https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados?dataInicial={start_date}&dataFinal={end_date}&formato=json'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    for attempt in range(5):
        try:
            response = urllib.request.urlopen(req, timeout=10)
            data = json.loads(response.read().decode('utf-8'))
            break
        except Exception as e:
            print(f'Attempt {attempt+1} failed for series {series_id}: {e}')
            time.sleep(2)
    else:
        raise Exception(f'Failed to fetch series {series_id} after 5 attempts')
        
    df = pd.DataFrame(data)
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
    df['valor'] = pd.to_numeric(df['valor'])
    return df.set_index('data')

def main():
    print("Baixando dados do Banco Central do Brasil (SGS)...")
    
    print('Coletando Salário Mínimo (1619)...')
    sm = fetch_bcb(1619)
    sm.columns = ['salario_minimo_nominal']
    
    print('Coletando IPCA (433)...')
    ipca = fetch_bcb(433)
    ipca.columns = ['ipca_mensal']

    print('Coletando INPC (188)...')
    inpc = fetch_bcb(188)
    inpc.columns = ['inpc_mensal']

    print('Coletando IGP-M (189)...')
    igpm = fetch_bcb(189)
    igpm.columns = ['igpm_mensal']

    print('Coletando Dólar (3695)...')
    dolar = fetch_bcb(3695)
    dolar.columns = ['dolar_mensal']

    print('Coletando Cesta Básica SP (7483)...')
    cesta = fetch_bcb(7483)
    cesta.columns = ['cesta_basica_sp']

    print("Consolidando os dados em uma única tabela...")
    # Usa o salário mínimo como base para garantir todos os meses
    df = sm.copy()
    df = df.join(ipca, how='left')
    df = df.join(inpc, how='left')
    df = df.join(igpm, how='left')
    df = df.join(dolar, how='left')
    df = df.join(cesta, how='left')
    
    # Opcional: ordenar o índice cronologicamente para garantir
    df = df.sort_index(ascending=True)
    
    # Renomear o índice para 'data'
    df.index.name = 'data'
    
    db_path = 'poder_de_compra.db'
    print(f"Criando banco de dados SQLite em: {db_path}...")
    
    # Remover o banco se já existir para recriar fresco
    if os.path.exists(db_path):
        os.remove(db_path)
        
    conn = sqlite3.connect(db_path)
    
    # Salvar o DataFrame no banco (datetime será salvo como texto no sqlite por padrão do pandas)
    df.to_sql('indicadores_economicos', conn, if_exists='replace', index=True)
    
    conn.close()
    print("Banco de dados SQLite gerado com sucesso! Tabela criada: 'indicadores_economicos'")

if __name__ == '__main__':
    main()
