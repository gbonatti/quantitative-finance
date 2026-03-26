"""
=============================================================================
MÓDULO 10E — SVM & KNN para Classificação de Direção
Quant Academy · Finanças Quantitativas
=============================================================================
Support Vector Machines e K-Nearest Neighbors para previsão de direção
do mercado. Hyperparameter tuning com TimeSeriesSplit.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             roc_curve, auc, classification_report)
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')
COLORS = {
    'green': '#00d4a0', 'amber': '#f5a623',
    'blue': '#4a9eff', 'red': '#e05252',
    'bg': '#080c10', 'surface': '#0d1520',
    'text': '#c8d8e8', 'dim': '#6a8aa8'
}
SAVE = dict(dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])


def calcular_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def criar_features(df):
    data = pd.DataFrame(index=df.index)
    close = df['Close']

    for lag in [1, 2, 3, 5, 10]:
        data[f'ret_{lag}'] = close.pct_change(lag)

    for w in [5, 20, 60]:
        data[f'ma{w}_r'] = close / close.rolling(w).mean() - 1

    data['rsi'] = calcular_rsi(close)
    data['vol_20'] = close.pct_change().rolling(20).std()
    data['vol_10'] = close.pct_change().rolling(10).std()
    data['mom_10'] = close / close.shift(10) - 1

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    data['bb_pct'] = (close - (ma20 - 2 * std20)) / (4 * std20)

    data['target'] = (close.shift(-1) > close).astype(int)
    return data.dropna()


def carregar_dados():
    import yfinance as yf
    print("📥 Baixando dados da VALE3.SA (3 anos)...")
    df = yf.download('VALE3.SA', period='3y', auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    print(f"   ✔ {len(df)} observações")

    data = criar_features(df)
    print(f"   ✔ {data.shape[1]-1} features, {len(data)} amostras")
    return data


def treinar_modelos(data):
    feature_cols = [c for c in data.columns if c != 'target']
    X = data[feature_cols].values
    y = data['target'].values

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    tscv = TimeSeriesSplit(n_splits=3)
    resultados = {}

    # ── SVM (RBF) ──
    print("\n🔵 Treinando SVM (RBF kernel)...")
    svm_params = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
    svm_gs = GridSearchCV(SVC(kernel='rbf', probability=True),
                          svm_params, cv=tscv, scoring='f1', n_jobs=-1)
    svm_gs.fit(X_train_sc, y_train)
    svm_pred = svm_gs.predict(X_test_sc)
    svm_proba = svm_gs.predict_proba(X_test_sc)[:, 1]
    print(f"   ✔ Best params: {svm_gs.best_params_}")

    resultados['SVM (RBF)'] = {
        'pred': svm_pred, 'proba': svm_proba,
        'accuracy': accuracy_score(y_test, svm_pred),
        'f1': f1_score(y_test, svm_pred)
    }

    # ── KNN ──
    print("🟢 Treinando KNN...")
    knn_params = {'n_neighbors': [3, 5, 7, 11, 15, 21]}
    knn_gs = GridSearchCV(KNeighborsClassifier(),
                          knn_params, cv=tscv, scoring='f1', n_jobs=-1)
    knn_gs.fit(X_train_sc, y_train)
    knn_pred = knn_gs.predict(X_test_sc)
    knn_proba = knn_gs.predict_proba(X_test_sc)[:, 1]
    print(f"   ✔ Best K: {knn_gs.best_params_['n_neighbors']}")

    resultados['KNN'] = {
        'pred': knn_pred, 'proba': knn_proba,
        'accuracy': accuracy_score(y_test, knn_pred),
        'f1': f1_score(y_test, knn_pred)
    }

    # ── Random Forest (baseline) ──
    print("🌲 Treinando Random Forest (baseline)...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=8,
                                min_samples_leaf=20, random_state=42)
    rf.fit(X_train_sc, y_train)
    rf_pred = rf.predict(X_test_sc)
    rf_proba = rf.predict_proba(X_test_sc)[:, 1]

    resultados['Random Forest'] = {
        'pred': rf_pred, 'proba': rf_proba,
        'accuracy': accuracy_score(y_test, rf_pred),
        'f1': f1_score(y_test, rf_pred)
    }

    for nome, res in resultados.items():
        print(f"\n   📊 {nome}: Acc={res['accuracy']:.4f} | F1={res['f1']:.4f}")

    return resultados, X_train_sc, X_test_sc, y_train, y_test


def grafico_decision_boundary(X_train, y_train, X_test, y_test, resultados):
    """Fronteiras de decisão projetadas em PCA 2D."""
    pca = PCA(n_components=2)
    X_train_2d = pca.fit_transform(X_train)
    X_test_2d = pca.transform(X_test)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor(COLORS['bg'])

    cores_modelo = [COLORS['blue'], COLORS['green'], COLORS['amber']]

    modelos_2d = {
        'SVM (RBF)': SVC(kernel='rbf', C=1, probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=7),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    }

    for idx, (nome, model) in enumerate(modelos_2d.items()):
        ax = axes[idx]
        ax.set_facecolor(COLORS['surface'])

        model.fit(X_train_2d, y_train)

        # Meshgrid para decision boundary
        h = 0.3
        x_min, x_max = X_test_2d[:, 0].min() - 1, X_test_2d[:, 0].max() + 1
        y_min, y_max = X_test_2d[:, 1].min() - 1, X_test_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                              np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.15,
                    colors=[COLORS['red'], COLORS['green']])
        ax.contour(xx, yy, Z, colors=[COLORS['dim']], linewidths=0.5)

        # Pontos de teste
        ax.scatter(X_test_2d[y_test == 0, 0], X_test_2d[y_test == 0, 1],
                   c=COLORS['red'], alpha=0.5, s=15, label='Desce')
        ax.scatter(X_test_2d[y_test == 1, 0], X_test_2d[y_test == 1, 1],
                   c=COLORS['green'], alpha=0.5, s=15, label='Sobe')

        acc = accuracy_score(y_test, model.predict(X_test_2d))
        ax.set_title(f'{nome} · Acc 2D: {acc:.3f}', fontsize=12,
                     color=COLORS['text'], fontweight='bold')
        ax.set_xlabel('PC1', color=COLORS['dim'])
        ax.set_ylabel('PC2', color=COLORS['dim'])
        ax.legend(fontsize=8)
        ax.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m10e_decision_boundaries.png', **SAVE)
    plt.close()
    print("   ✔ graficos/m10e_decision_boundaries.png")


def grafico_roc_comparativo(resultados, y_test):
    """Curvas ROC comparativas."""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['surface'])

    cores = [COLORS['blue'], COLORS['green'], COLORS['amber']]

    for idx, (nome, res) in enumerate(resultados.items()):
        fpr, tpr, _ = roc_curve(y_test, res['proba'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=cores[idx], linewidth=2,
                label=f'{nome} (AUC = {roc_auc:.4f})')

    ax.plot([0, 1], [0, 1], color=COLORS['dim'], linestyle='--', linewidth=1)
    ax.set_xlabel('FPR', color=COLORS['dim'], fontsize=12)
    ax.set_ylabel('TPR', color=COLORS['dim'], fontsize=12)
    ax.set_title('CURVA ROC · SVM vs KNN vs RANDOM FOREST',
                 fontsize=14, color=COLORS['text'], fontweight='bold', pad=15)
    ax.legend(fontsize=10)
    ax.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m10e_roc_comparison.png', **SAVE)
    plt.close()
    print("   ✔ graficos/m10e_roc_comparison.png")


def grafico_confusion_matrices(resultados, y_test):
    """Matrizes de confusão."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(COLORS['bg'])

    for idx, (nome, res) in enumerate(resultados.items()):
        ax = axes[idx]
        cm = confusion_matrix(y_test, res['pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGn',
                    xticklabels=['Desce', 'Sobe'],
                    yticklabels=['Desce', 'Sobe'],
                    ax=ax, cbar=False,
                    annot_kws={'size': 14, 'weight': 'bold'})
        ax.set_title(f'{nome}\nAcc: {res["accuracy"]:.3f} | F1: {res["f1"]:.3f}',
                     fontsize=11, color=COLORS['text'], fontweight='bold')
        ax.set_ylabel('Real', color=COLORS['dim'])
        ax.set_xlabel('Previsto', color=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m10e_confusion_matrices.png', **SAVE)
    plt.close()
    print("   ✔ graficos/m10e_confusion_matrices.png")


def main():
    print("=" * 70)
    print("  MÓDULO 10E · SVM & KNN — CLASSIFICAÇÃO")
    print("=" * 70)

    data = carregar_dados()
    resultados, X_train, X_test, y_train, y_test = treinar_modelos(data)

    print("\n🎨 Gerando gráficos...")
    grafico_decision_boundary(X_train, y_train, X_test, y_test, resultados)
    grafico_roc_comparativo(resultados, y_test)
    grafico_confusion_matrices(resultados, y_test)

    print(f"\n{'=' * 70}")
    print("  ✅ Módulo 10E concluído!")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
