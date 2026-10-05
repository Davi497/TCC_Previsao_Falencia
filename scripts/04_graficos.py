# %% [markdown]
# # Gráficos do TCC – Previsão de Falência Corporativa
# Gera as Figuras 1 a 4 do TCC a partir do arquivo "resultados_v6.json",
# produzido pelo script 03_pipeline_final.py (rodar o pipeline antes).
# Uso: python scripts/04_graficos.py resultados figuras
#
# Saídas (PNG, 300 dpi, 16 cm de largura):
#   fig1_matrizes.png    – Figura 1: matrizes de confusão (teste, limiar 0,5)
#   fig2_roc_pr.png      – Figura 2: curvas ROC (A) e Precisão-Recall (B)
#   fig3_estrategias.png – Figura 3: Precisão, Recall e F1 por estratégia
#   fig4_importancia.png – Figura 4: importância por permutação (Gradient Boosting)
#
# Formatação conforme o manual USP/Esalq: sem título no gráfico, sem linhas de
# grade, eixos pretos de 1,5 pt, textos em português e vírgula decimal.

import json, logging, numpy as np, matplotlib; matplotlib.use('Agg')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib.pyplot as plt
# Fonte: Arial (no Linux, Liberation Sans é equivalente métrico)
plt.rcParams.update({'font.family':['Arial','Liberation Sans','DejaVu Sans'],'font.size':9})
import os, sys
BASE = sys.argv[1] if len(sys.argv) > 1 else '.'
OUT = sys.argv[2] if len(sys.argv) > 2 else '.'
os.makedirs(OUT, exist_ok=True)
d=json.load(open(os.path.join(BASE, 'resultados_v6.json'))); R=d['res']; C=d['curves']
M=['Regressão Logística','Random Forest','Gradient Boosting']; T=['Original','SMOTE']
fmt=lambda v:f'{v:.3f}'.replace('.',',')
def axes_style(ax):
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_linewidth(1.5); ax.spines[s].set_color('black')
    ax.tick_params(colors='black')
# ---- Fig 1: confusion matrices
fig,axs=plt.subplots(2,3,figsize=(6.3,4.6))
L='ABCDEF'; k=0
for i,t in enumerate(T):
    for j,m in enumerate(M):
        ax=axs[i,j]; cm=np.array(R[f'{m} | {t}']['test05']['cm'])
        norm=cm/cm.sum(axis=1,keepdims=True)
        ax.imshow(norm,cmap='Blues',vmin=0,vmax=1.15)
        for a in range(2):
            for b in range(2):
                ax.text(b,a,f'{cm[a,b]:,}'.replace(',','.'),ha='center',va='center',fontsize=10,color='white' if norm[a,b]>0.65 else 'black')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Não falida','Falida'],fontsize=8); ax.set_yticklabels(['Não\nfalida','Falida'],fontsize=8)
        ax.set_xlabel('Classe prevista',fontsize=9)
        if j==0: ax.set_ylabel('Classe real',fontsize=9)
        ax.text(-0.30,1.04,L[k],transform=ax.transAxes,fontsize=11,fontweight='bold'); k+=1
        for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig1_matrizes.png'),dpi=300,facecolor='white'); plt.close()
# ---- Fig 2: ROC and PR curves
sty={('Regressão Logística','Original'):('#0072B2','-'),('Regressão Logística','SMOTE'):('#0072B2','--'),
     ('Random Forest','Original'):('#E69F00','-'),('Random Forest','SMOTE'):('#E69F00','--'),
     ('Gradient Boosting','Original'):('#009E73','-'),('Gradient Boosting','SMOTE'):('#009E73','--')}
fig,axs=plt.subplots(1,2,figsize=(6.3,3.9))
for (m,t),(c,ls) in sty.items():
    cv=C[f'{m} | {t}']; r=R[f'{m} | {t}']['test05']
    axs[0].plot(cv['fpr'],cv['tpr'],color=c,ls=ls,lw=1.3,label=f'{m} – {t}')
    axs[1].plot(cv['rc'],cv['pr'],color=c,ls=ls,lw=1.3)
axs[0].plot([0,1],[0,1],color='black',lw=0.8,ls=':')
axs[1].axhline(627/13022,color='black',lw=0.8,ls=':')
axs[0].set_xlabel('Taxa de falsos positivos'); axs[0].set_ylabel('Taxa de verdadeiros positivos')
axs[1].set_xlabel('Sensibilidade (Recall)'); axs[1].set_ylabel('Precisão')
for ax,l in zip(axs,'AB'):
    axes_style(ax); ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,p:f'{v:.1f}'.replace('.',',')))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,p:f'{v:.1f}'.replace('.',',')))
    ax.text(-0.2,1.02,l,transform=ax.transAxes,fontsize=11,fontweight='bold')
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,fontsize=8)
fig.tight_layout(rect=(0,0.17,1,1)); fig.savefig(os.path.join(OUT, 'fig2_roc_pr.png'),dpi=300,facecolor='white'); plt.close()
# ---- Fig 3: strategies bars
S=[('Original (limiar 0,5)','#56B4E9',lambda m:R[f'{m} | Original']['test05']),
   ('SMOTE (limiar 0,5)','#E69F00',lambda m:R[f'{m} | SMOTE']['test05']),
   ('Original (limiar ajustado)','#009E73',lambda m:R[f'{m} | Original']['testthr'])]
fig,axs=plt.subplots(1,3,figsize=(6.3,3.2),sharey=True); x=np.arange(3); w=0.26
for ax,(key,lab),L in zip(axs,[('prec','Precisão'),('rec','Recall'),('f1','F1-Score')],'ABC'):
    for i,(nm,col,f) in enumerate(S):
        vals=[f(m)[key] for m in M]; bars=ax.bar(x+(i-1)*w,vals,w,color=col,label=nm)
        for r_ in bars: ax.text(r_.get_x()+r_.get_width()/2,r_.get_height()+0.012,fmt(r_.get_height()),ha='center',va='bottom',fontsize=6.5,rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(['Regressão\nLogística','Random\nForest','Gradient\nBoosting'],fontsize=7.5)
    ax.set_xlabel(lab); axes_style(ax); ax.set_ylim(0,1.15)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,p:f'{v:.1f}'.replace('.',',')))
    ax.text(-0.14,1.0,L,transform=ax.transAxes,fontsize=11,fontweight='bold')
axs[0].set_ylabel('Valor da métrica (classe falida)')
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,fontsize=8)
fig.tight_layout(rect=(0,0,1,0.92)); fig.savefig(os.path.join(OUT, 'fig3_estrategias.png'),dpi=300,facecolor='white'); plt.close()
# ---- Fig 4: permutation importance
imp=d['imp'][:10][::-1]
short={27:'Lucro operacional / Despesas financeiras (Attr27)',46:'(Ativo circ. − Estoques) / Passivo circ. (Attr46)',
 34:'Despesas operacionais / Passivo total (Attr34)',5:'Intervalo defensivo em dias (Attr5)',58:'Custos totais / Vendas totais (Attr58)',
 56:'(Vendas − Custo dos prod. vendidos) / Vendas (Attr56)',24:'Lucro bruto (3 anos) / Ativo total (Attr24)',6:'Lucros retidos / Ativo total (Attr6)',
 36:'Vendas totais / Ativo total (Attr36)',33:'Despesas operacionais / Passivo circ. (Attr33)'}
fig,ax=plt.subplots(figsize=(6.3,3.6))
labs=[short[int(r['var'][4:])] for r in imp]
ax.barh(labs,[r['mean'] for r in imp],xerr=[r['std'] for r in imp],color='#0072B2',ecolor='black',capsize=2)
ax.set_xlabel('Redução média da PR-AUC após permutação'); axes_style(ax)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,p:f'{v:.1f}'.replace('.',',')))
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig4_importancia.png'),dpi=300,facecolor='white'); plt.close()
print('Figuras geradas: fig1_matrizes.png, fig2_roc_pr.png, fig3_estrategias.png, fig4_importancia.png')
