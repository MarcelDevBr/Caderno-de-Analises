# Tema 01: Comportamento Eleitoral 2026 - Bússola Eleitoral

Este projeto visa a construção de uma Bússola Eleitoral (VAA - Voting Advice Application) fundamentada na Teoria Espacial do Voto.

## Metodologia e Fundamentação
O algoritmo mapeia as dores, demografias e capital cultural do eleitorado, alocando-os em um plano multidimensional. O cálculo da distância (Euclidiana e Cosseno) entre eleitor e pré-candidato determina a recomendação, com um filtro de "rejeição absoluta" embasado em polarização afetiva.

## Estrutura
- `data/`: Contém os dados brutos (`raw`) coletados por formulário e os dados limpos (`processed`).
- `notebooks/`: Análise Exploratória de Dados (EDA) e visualizações.
- `src/`: Módulos Python reusáveis para processamento de dados e cálculo das distâncias matemáticas.
- `app/`: Aplicação interativa desenvolvida em Streamlit.
