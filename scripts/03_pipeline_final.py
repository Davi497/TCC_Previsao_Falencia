# %% [markdown]
# # Previsão de Falência Corporativa – Versão 6
# Pipeline sem vazamento (imputação, padronização e SMOTE ajustados apenas no treino de cada dobra),
# validação cruzada estratificada (k = 5) no treino, ajuste de limiar, PR-AUC, teste de McNemar
# e importância por permutação.

# %%
import json, numpy as np, pandas as pd
from scipy.io import arff
from scipy.stats import chi2
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix, precision_recall_curve, roc_curve)
from sklearn.inspection import permutation_importance
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

SEED = 42
import os, sys
DATA = sys.argv[1] if len(sys.argv) > 1 else '.'

# %% 1. Carregamento (arquivos ARFF originais; alternativa: CSV equivalentes)
def load():
    frames = []
    for i in range(1, 6):
        p = os.path.join(DATA, f'{i}year.arff')
        if os.path.exists(p):
            d, _ = arff.loadarff(p); d = pd.DataFrame(d)
            d['class'] = d['class'].str.decode('utf-8').astype(int)
        else:
            d = pd.read_csv(os.path.join(DATA, f'{i}year.csv'), low_memory=False, na_values=['?'])
        frames.append(d[[f'Attr{j}' for j in range(1, 65)] + ['class']])
    return pd.concat(frames, ignore_index=True)

df = load()
X = df[[f'Attr{j}' for j in range(1, 65)]].astype(float).replace([np.inf, -np.inf], np.nan)
y = df['class'].astype(int)
print('Observações:', len(df), '| classes:', y.value_counts().to_dict(), '| ausentes:', int(X.isna().sum().sum()))

# %% 2. Divisão treino/teste (antes de qualquer transformação)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
print('Treino:', y_tr.value_counts().to_dict(), '| Teste:', y_te.value_counts().to_dict())

# %% 3. Configurações (modelagem original x SMOTE)
def models(balanced):
    cw = 'balanced' if balanced else None
    return {
        'Regressão Logística': LogisticRegression(max_iter=1000, random_state=SEED, class_weight=cw),
        'Random Forest': RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, class_weight=cw),
        'Gradient Boosting': HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=6, random_state=SEED),
    }

def pipe(model, smote):
    steps = [('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]
    if smote: steps.append(('smote', SMOTE(random_state=SEED)))
    steps.append(('clf', model))
    return Pipeline(steps)

CONFIGS = {}
for name, m in models(True).items():
    CONFIGS[(name, 'Original')] = pipe(m, False)
for name, m in models(False).items():
    CONFIGS[(name, 'SMOTE')] = pipe(m, True)

# %% 4. Validação cruzada estratificada (k = 5) no treino + limiar ótimo (F1) fora da dobra
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
SC = {'acc': 'accuracy', 'prec': 'precision', 'rec': 'recall', 'f1': 'f1', 'auc': 'roc_auc', 'ap': 'average_precision'}
res = {}
for key, p in CONFIGS.items():
    cvr = cross_validate(p, X_tr, y_tr, cv=cv, scoring=SC, n_jobs=1)
    oof = cross_val_predict(p, X_tr, y_tr, cv=cv, method='predict_proba')[:, 1]
    pr, rc, th = precision_recall_curve(y_tr, oof)
    f1s = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    thr = float(th[np.argmax(f1s)])
    res[key] = {'cv': {k: (float(cvr['test_' + k].mean()), float(cvr['test_' + k].std())) for k in SC}, 'thr': thr}
    print(key, {k: f'{v[0]:.3f}±{v[1]:.3f}' for k, v in res[key]['cv'].items()}, 'limiar', round(thr, 3))

# %% 5. Ajuste final no treino completo e avaliação no teste (limiar 0,5 e limiar ajustado)
def metrics(yt, yp, prob):
    return {'acc': accuracy_score(yt, yp), 'prec': precision_score(yt, yp, zero_division=0), 'rec': recall_score(yt, yp),
            'f1': f1_score(yt, yp), 'auc': roc_auc_score(yt, prob), 'ap': average_precision_score(yt, prob),
            'cm': confusion_matrix(yt, yp).tolist()}
curves, preds = {}, {}
for key, p in CONFIGS.items():
    p.fit(X_tr, y_tr)
    prob = p.predict_proba(X_te)[:, 1]
    y05 = (prob >= 0.5).astype(int); yth = (prob >= res[key]['thr']).astype(int)
    res[key]['test05'] = metrics(y_te, y05, prob)
    res[key]['testthr'] = metrics(y_te, yth, prob)
    preds[key] = y05
    fpr, tpr, _ = roc_curve(y_te, prob); pr, rc, _ = precision_recall_curve(y_te, prob)
    curves[key] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'pr': pr.tolist(), 'rc': rc.tolist()}
    print(key, 'teste 0,5:', {k: round(v, 3) for k, v in res[key]['test05'].items() if k != 'cm'}, res[key]['test05']['cm'])
    print(key, 'teste lim:', {k: round(v, 3) for k, v in res[key]['testthr'].items() if k != 'cm'}, res[key]['testthr']['cm'])

# %% 6. Teste de McNemar (com correção de continuidade): Original x SMOTE, limiar 0,5
yt = y_te.values
for name in ['Regressão Logística', 'Random Forest', 'Gradient Boosting']:
    a = preds[(name, 'Original')] == yt; b = preds[(name, 'SMOTE')] == yt
    n01 = int(np.sum(a & ~b)); n10 = int(np.sum(~a & b))
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if n01 + n10 else 0.0
    pv = float(chi2.sf(stat, 1))
    res[(name, 'McNemar')] = {'n01': n01, 'n10': n10, 'stat': stat, 'p': pv}
    print('McNemar', name, n01, n10, round(stat, 2), pv)

# %% 7. Importância por permutação (Gradient Boosting original, teste, métrica PR-AUC)
gb = CONFIGS[('Gradient Boosting', 'Original')]
pi = permutation_importance(gb, X_te, y_te, scoring='average_precision', n_repeats=10, random_state=SEED, n_jobs=-1)
imp = pd.DataFrame({'var': X.columns, 'mean': pi.importances_mean, 'std': pi.importances_std}).sort_values('mean', ascending=False)
print(imp.head(10))

os.makedirs(sys.argv[2] if len(sys.argv) > 2 else '.', exist_ok=True)
json.dump({'res': {' | '.join(k): v for k, v in res.items()}, 'curves': {' | '.join(k): v for k, v in curves.items()},
           'imp': imp.head(15).to_dict('records'),
           'split': {'tr': y_tr.value_counts().to_dict(), 'te': y_te.value_counts().to_dict()}},
          open(os.path.join(sys.argv[2] if len(sys.argv) > 2 else '.', 'resultados_v6.json'), 'w'), default=int)
