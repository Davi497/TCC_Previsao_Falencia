# Comparação de modelos estatísticos e de “machine learning” na previsão de falência corporativa

Scripts em Python utilizados no Trabalho de Conclusão de Curso do MBA em Data Science
e Analytics (USP/Esalq), de autoria de **Davi Coelho Moreira**, sob orientação do
**Prof. Dr. Evandro Marcos Saidel Ribeiro** (2026).

O estudo comparou a Regressão Logística, o Random Forest e o Gradient Boosting
(HistGradientBoostingClassifier) na previsão de falência de empresas polonesas e avaliou
três estratégias de tratamento do desbalanceamento entre as classes: ponderação de
classes, sobreamostragem sintética (SMOTE) e ajuste do limiar de decisão.

## Estrutura do repositório

```
TCC_Previsao_Falencia/
├── scripts/
│   ├── 01_modelagem_original.py   # versão exploratória: modelagem com ponderação de classes
│   ├── 02_modelagem_smote.py      # versão exploratória: aplicação do SMOTE
│   ├── 03_pipeline_final.py       # análise final reportada no TCC
│   └── 04_graficos.py             # Figuras 1 a 4 do TCC
├── dados/
│   ├── README.md                  # como obter os arquivos .arff (não versionados)
│   └── variaveis.md               # definição das 64 variáveis explicativas
├── resultados/                    # criada pelo script 03 (resultados_v6.json)
├── figuras/                       # criada pelo script 04 (Figuras 1 a 4 do TCC)
└── requirements.txt
```

## Relação entre scripts e o texto do TCC

| Script | Papel no trabalho | Elementos do TCC |
|---|---|---|
| `03_pipeline_final.py` | Análise final: divisão treino/teste estratificada (70/30), fluxo de processamento sem vazamento (imputação pela mediana, padronização e SMOTE ajustados apenas no treino), validação cruzada estratificada com cinco dobras, ajuste do limiar de decisão, ROC-AUC, PR-AUC, teste de McNemar e importância por permutação | Tabelas 3 a 7 e Figura 4 (dados) |
| `04_graficos.py` | Geração das figuras a partir de `resultados/resultados_v6.json` | Figuras 1 a 4 |
| `01_modelagem_original.py` | Versão exploratória com ponderação de classes e avaliação por holdout | Etapa preliminar (versões anteriores do texto) |
| `02_modelagem_smote.py` | Versão exploratória com SMOTE aplicado ao treino | Etapa preliminar (versões anteriores do texto) |

Os scripts 01 e 02 documentam a evolução do trabalho. Os resultados apresentados na
versão final do TCC foram gerados exclusivamente pelos scripts 03 e 04.

## Como reproduzir

```bash
pip install -r requirements.txt

# 1) coloque os arquivos 1year.arff ... 5year.arff em dados/ (ver dados/README.md)

# 2) análise final (cerca de 30 minutos; gera resultados/resultados_v6.json)
python scripts/03_pipeline_final.py dados resultados

# 3) figuras (gera os PNG em figuras/)
python scripts/04_graficos.py resultados figuras
```

Os scripts exploratórios (01 e 02) leem os arquivos `.arff` do diretório corrente:

```bash
cd dados
python ../scripts/01_modelagem_original.py
python ../scripts/02_modelagem_smote.py
```

Todas as etapas aleatórias utilizam semente fixa (`random_state = 42`).

## Principais resultados (conjunto de teste)

| Configuração | Precisão | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Gradient Boosting – original (limiar 0,5) | 0,921 | 0,467 | 0,620 | 0,963 | 0,763 |
| Gradient Boosting – SMOTE (limiar 0,5) | 0,431 | 0,751 | 0,547 | 0,956 | 0,707 |
| Gradient Boosting – original (limiar ajustado 0,23) | 0,768 | 0,617 | 0,684 | 0,963 | 0,763 |

## Variáveis

A definição completa dos 64 indicadores financeiros, com a tradução adotada no TCC,
está em [`dados/variaveis.md`](dados/variaveis.md).

## Fonte dos dados

Tomczak, S. 2016. Polish Companies Bankruptcy. UCI Machine Learning Repository.
<https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data>

## Como citar

Moreira, D.C. 2026. Scripts do TCC “Comparação de modelos estatísticos e de machine
learning na previsão de falência corporativa”. Disponível em:
<https://github.com/Davi497/TCC_Previsao_Falencia>.
