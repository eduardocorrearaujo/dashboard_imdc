# Dashboard Streamlit — Previsões de dengue e chikungunya 2026–2027

Dashboard interativo em Streamlit + Plotly para as previsões do ensemble e os casos prováveis observados no Brasil.

## Executar localmente

Requer Python 3.10 ou superior.

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows (PowerShell: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt
streamlit run app.py
```

O Streamlit abrirá o dashboard no navegador, geralmente em `http://localhost:8501`.

## Publicar no Streamlit Community Cloud

1. Crie um repositório no GitHub e envie `app.py`, `requirements.txt` e a pasta `data/` com os seis arquivos `.csv.gz`.
2. Acesse https://share.streamlit.io/ e escolha **Create app**.
3. Selecione repositório, branch e `app.py` como arquivo principal e publique.
4. Para atualizar as previsões, substitua os arquivos `.csv.gz` na pasta `data/` e faça commit/push.

Não publique dados privados; este exemplo pressupõe dados adequados para divulgação pública. Outra opção é apontar a variável de ambiente `DASH_DATA_DIR` para uma pasta acessível ao servidor.

## Estrutura

```text
dashboard_imdc/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── data/
│   ├── dengue_casprov.csv.gz
│   ├── chik_casprov.csv.gz
│   ├── ensemble_weekly_2027_dengue.csv.gz
│   ├── ensemble_weekly_2027_chik.csv.gz
│   ├── ensemble_total_2027_dengue.csv.gz
│   └── ensemble_total_2027_chik.csv.gz
└── tests/
    └── test_dashboard.py
```

## Conteúdo

- Alternância entre dengue e chikungunya.
- Série semanal nacional, com série observada e previsão mediana mais intervalos de 80% e 90%.
- Gráficos semanais individuais organizados em **painéis regionais**, incluindo PR, SC, RS no mesmo painel da Região Sul.
- Comparação de acumulados observados na temporada anterior (2025–2026) com as previsões para 2026–2027.
- Barras com intervalos acumulados de 80% e 90% e download da tabela de acumulados.
- Controles de histórico, regiões exibidas e visualização de dados observados durante a temporada prevista.

## Observações metodológicas

- Os seis arquivos enviados são utilizados sem alterar os valores do ensemble.
- A previsão contém 52 semanas entre 11/10/2026 e 03/10/2027.
- O total observado de BR soma a coluna `casos` de todas as 27 UFs, incluindo ES. As previsões não incluem ES; nos gráficos do Sudeste, ES aparece apenas com observações.
- A previsão nacional usa diretamente a série e o acumulado `BR` dos arquivos do ensemble, que **não equivalem necessariamente à soma das medianas estaduais**.
- Os intervalos dos acumulados vêm diretamente de `ensemble_total_2027_*.csv.gz`, e **não** da soma dos quantis semanais.
- Para comparação de temporadas completas, os casos observados são somados nas 52 semanas imediatamente anteriores ao início da previsão: 12/10/2025 a 04/10/2026.

## Testes

```bash
pip install pytest
pytest -q
```
