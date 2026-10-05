# Definição das variáveis explicativas

Definições dos 64 indicadores financeiros do conjunto **Polish Companies Bankruptcy** (Tomczak, 2016),
conforme a documentação original, com a tradução adotada no TCC.

| Código | Definição (tradução adotada no TCC) |
|---|---|
| Attr1 | Lucro líquido / Ativo total |
| Attr2 | Passivo total / Ativo total |
| Attr3 | Capital de giro / Ativo total |
| Attr4 | Ativo circulante / Passivo circulante |
| Attr5 | [(Caixa + Títulos de curto prazo + Recebíveis − Passivo circulante) / (Despesas operacionais − Depreciação)] × 365 |
| Attr6 | Lucros retidos / Ativo total |
| Attr7 | EBIT / Ativo total |
| Attr8 | Valor contábil do patrimônio líquido / Passivo total |
| Attr9 | Vendas / Ativo total |
| Attr10 | Patrimônio líquido / Ativo total |
| Attr11 | (Lucro bruto + Itens extraordinários + Despesas financeiras) / Ativo total |
| Attr12 | Lucro bruto / Passivo circulante |
| Attr13 | (Lucro bruto + Depreciação) / Vendas |
| Attr14 | (Lucro bruto + Juros) / Ativo total |
| Attr15 | (Passivo total × 365) / (Lucro bruto + Depreciação) |
| Attr16 | (Lucro bruto + Depreciação) / Passivo total |
| Attr17 | Ativo total / Passivo total |
| Attr18 | Lucro bruto / Ativo total |
| Attr19 | Lucro bruto / Vendas |
| Attr20 | (Estoques × 365) / Vendas |
| Attr21 | Vendas (n) / Vendas (n−1) |
| Attr22 | Lucro operacional / Ativo total |
| Attr23 | Lucro líquido / Vendas |
| Attr24 | Lucro bruto (em 3 anos) / Ativo total |
| Attr25 | (Patrimônio líquido − Capital social) / Ativo total |
| Attr26 | (Lucro líquido + Depreciação) / Passivo total |
| Attr27 | Lucro operacional / Despesas financeiras |
| Attr28 | Capital de giro / Ativo não circulante |
| Attr29 | Logaritmo do ativo total |
| Attr30 | (Passivo total − Caixa) / Vendas |
| Attr31 | (Lucro bruto + Juros) / Vendas |
| Attr32 | (Passivo circulante × 365) / Custo dos produtos vendidos |
| Attr33 | Despesas operacionais / Passivo circulante |
| Attr34 | Despesas operacionais / Passivo total |
| Attr35 | Lucro sobre vendas / Ativo total |
| Attr36 | Vendas totais / Ativo total |
| Attr37 | (Ativo circulante − Estoques) / Passivo não circulante |
| Attr38 | Capital permanente / Ativo total |
| Attr39 | Lucro sobre vendas / Vendas |
| Attr40 | (Ativo circulante − Estoques − Recebíveis) / Passivo circulante |
| Attr41 | Passivo total / [(Lucro operacional + Depreciação) × (12/365)] |
| Attr42 | Lucro operacional / Vendas |
| Attr43 | Prazo de recebimento + Prazo de estocagem (dias) |
| Attr44 | (Recebíveis × 365) / Vendas |
| Attr45 | Lucro líquido / Estoques |
| Attr46 | (Ativo circulante − Estoques) / Passivo circulante |
| Attr47 | (Estoques × 365) / Custo dos produtos vendidos |
| Attr48 | (Lucro operacional − Depreciação) / Ativo total |
| Attr49 | (Lucro operacional − Depreciação) / Vendas |
| Attr50 | Ativo circulante / Passivo total |
| Attr51 | Passivo circulante / Ativo total |
| Attr52 | (Passivo circulante × 365) / Custo dos produtos vendidos |
| Attr53 | Patrimônio líquido / Ativo não circulante |
| Attr54 | Capital permanente / Ativo não circulante |
| Attr55 | Capital de giro |
| Attr56 | (Vendas − Custo dos produtos vendidos) / Vendas |
| Attr57 | (Ativo circulante − Estoques − Passivo circulante) / (Vendas − Lucro bruto − Depreciação) |
| Attr58 | Custos totais / Vendas totais |
| Attr59 | Passivo não circulante / Patrimônio líquido |
| Attr60 | Vendas / Estoques |
| Attr61 | Vendas / Recebíveis |
| Attr62 | (Passivo circulante × 365) / Vendas |
| Attr63 | Vendas / Passivo circulante |
| Attr64 | Vendas / Ativo não circulante |
| class | Variável dependente (1 = falida; 0 = não falida) |

Notas: “lucro bruto” traduz o termo “gross profit” da documentação original; “capital permanente”
corresponde à soma do patrimônio líquido e do passivo não circulante; EBIT = lucro antes de juros e
impostos (“Earnings Before Interest and Taxes”).

Fonte: Tomczak, S. 2016. Polish Companies Bankruptcy. UCI Machine Learning Repository.
<https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data>
