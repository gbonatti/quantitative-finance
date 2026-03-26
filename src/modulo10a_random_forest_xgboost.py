"""
=============================================================================
MÓDULO 10A — Random Forest & XGBoost para Classificação de Direção
Quant Academy · Finanças Quantitativas
=============================================================================
Modelos supervisionados para prever a direção do mercado (sobe/desce).
Feature engineering com indicadores técnicos, avaliação rigorosa.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             classification_report)
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# ── Tema ───────────────────────────────────────────────────────────────────
plt.style.use('dark_background')
COLORS = {
    'green': '#00d4a0', 'amber': '#f5a623',
    'blue': '#4a9eff', 'red': '#e05252',
    'bg': '#080c10', 'surface': '#0d1520',
    'text': '#c8d8e8', 'dim': '#6a8aa8'
}
SAVE = dict(dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])


# ══════════════════════════════════════════════════════════════════════════════
# 1. COLETA E FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def calcular_rsi(series, period=14):
    """Calcula o Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calcular_macd(series, fast=12, slow=26, signal=9):
    """Calcula MACD e Signal Line."""
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line


def criar_features(df):
    """Cria features técnicas para o modelo de ML."""
    data = pd.DataFrame(index=df.index)

    close = df['Close']
    volume = df['Volume'] if 'Volume' in df.columns else pd.Series(0, index=df.index)

    # Retornos passados (lags)
    for lag in [1, 2, 3, 5, 10]:
        data[f'ret_lag_{lag}'] = close.pct_change(lag)

    # Médias móveis
    for janela in [5, 20, 60]:
        ma = close.rolling(janela).mean()
        data[f'ma_{janela}_ratio'] = close / ma - 1

    # RSI
    data['rsi_14'] = calcular_rsi(close, 14)

    # MACD
    macd, signal = calcular_macd(close)
    data['macd'] = macd
    data['macd_signal'] = macd - signal

    # Volatilidade rolling
    data['vol_20'] = close.pct_change().rolling(20).std()
    data['vol_60'] = close.pct_change().rolling(60).std()

    # Volume change
    if volume.sum() > 0:
        data['vol_change'] = volume.pct_change()
        data['vol_ma_ratio'] = volume / volume.rolling(20).mean()

    # Bandas de Bollinger
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    data['bb_upper'] = (close - (ma20 + 2 * std20)) / close
    data['bb_lower'] = (close - (ma20 - 2 * std20)) / close

    # Target: direção do próximo dia (1 = sobe, 0 = desce)
    data['target'] = (close.shift(-1) > close).astype(int)

    return data.dropna()


def carregar_dados():
    """Baixa dados da PETR4 e cria features."""
    import yfinance as yf
    print("📥 Baixando dados da PETR4.SA (3 anos)...")
    df = yf.download('PETR4.SA', period='3y', auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    print(f"   ✔ {len(df)} observações")

    print("🔧 Criando features técnicas...")
    data = criar_features(df)
    print(f"   ✔ {len(data)} amostras com {data.shape[1]-1} features")
    print(f"   ✔ Distribuição target: {data['target'].value_counts().to_dict()}")

    return data


# ══════════════════════════════════════════════════════════════════════════════
# 2. TREINAMENTO E AVALIAÇÃO
# ══════════════════════════════════════════════════════════════════════════════
def treinar_modelos(data):
    """Treina Random Forest e XGBoost com split temporal."""
    feature_cols = [c for c in data.columns if c != 'target']
    X = data[feature_cols]
    y = data['target']

    # Split temporal 80/20
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"\n📊 Split Temporal:")
    print(f"   Treino: {len(X_train)} ({X_train.index[0].date()} → {X_train.index[-1].date()})")
    print(f"   Teste:  {len(X_test)} ({X_test.index[0].date()} → {X_test.index[-1].date()})")

    resultados = {}

    # ── Random Forest ──
    print("\n🌲 Treinando Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_leaf=20,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    resultados['Random Forest'] = {
        'model': rf, 'pred': rf_pred, 'proba': rf_proba,
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred),
        'recall': recall_score(y_test, rf_pred),
        'f1': f1_score(y_test, rf_pred),
    }

    # ── XGBoost ──
    print("🚀 Treinando XGBoost...")
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.01,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, use_label_encoder=False,
            eval_metric='logloss', verbosity=0
        )
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        xgb_proba = xgb.predict_proba(X_test)[:, 1]

        resultados['XGBoost'] = {
            'model': xgb, 'pred': xgb_pred, 'proba': xgb_proba,
            'accuracy': accuracy_score(y_test, xgb_pred),
            'precision': precision_score(y_test, xgb_pred),
            'recall': recall_score(y_test, xgb_pred),
            'f1': f1_score(y_test, xgb_pred),
        }
    except ImportError:
        print("   ⚠ XGBoost não instalado. pip install xgboost")

    # Imprimir métricas
    for nome, res in resultados.items():
        print(f"\n   📊 {nome}:")
        print(f"      Accuracy:  {res['accuracy']:.4f}")
        print(f"      Precision: {res['precision']:.4f}")
        print(f"      Recall:    {res['recall']:.4f}")
        print(f"      F1-Score:  {res['f1']:.4f}")

    return resultados, X_train, X_test, y_train, y_test, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
# 3. GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════
def grafico_feature_importance(resultados, feature_cols):
    """Top features por importância para cada modelo."""
    n_modelos = len(resultados)
    fig, axes = plt.subplots(1, n_modelos, figsize=(8 * n_modelos, 8))
    fig.patch.set_facecolor(COLORS['bg'])

    if n_modelos == 1:
        axes = [axes]

    cores = [COLORS['green'], COLORS['amber']]

    for idx, (nome, res) in enumerate(resultados.items()):
        ax = axes[idx]
        ax.set_facecolor(COLORS['surface'])

        importances = res['model'].feature_importances_
        sorted_idx = np.argsort(importances)[-15:]  # Top 15

        ax.barh(range(len(sorted_idx)),
                importances[sorted_idx],
                color=cores[idx], alpha=0.8)
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_cols[i] for i in sorted_idx],
                           color=COLORS['text'], fontsize=9)
        ax.set_xlabel('Importância', color=COLORS['dim'])
        ax.set_title(f'FEATURE IMPORTANCE · {nome.upper()}',
                     fontsize=13, color=COLORS['text'], fontweight='bold', pad=15)
        ax.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m10a_feature_importance.png', **SAVE)
    plt.close()
    print("   ✔ Gráfico salvo: graficos/m10a_feature_importance.png")


def grafico_confusion_matrix(resultados, y_test):
    """Matrizes de confusão side by side."""
    n_modelos = len(resultados)
    fig, axes = plt.subplots(1, n_modelos, figsize=(7 * n_modelos, 6))
    fig.patch.set_facecolor(COLORS['bg'])

    if n_modelos == 1:
        axes = [axes]

    for idx, (nome, res) in enumerate(resultados.items()):
        ax = axes[idx]
        cm = confusion_matrix(y_test, res['pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGn',
                    xticklabels=['Desce', 'Sobe'],
                    yticklabels=['Desce', 'Sobe'],
                    ax=ax, cbar=False,
                    annot_kws={'size': 14, 'weight': 'bold'})
        ax.set_title(f'CONFUSION MATRIX · {nome.upper()}',
                     fontsize=13, color=COLORS['text'], fontweight='bold', pad=15)
        ax.set_ylabel('Real', color=COLORS['dim'])
        ax.set_xlabel('Previsto', color=COLORS['dim'])
        ax.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m10a_confusion_matrix.png', **SAVE)
    plt.close()
    print("   ✔ Gráfico salvo: graficos/m10a_confusion_matrix.png")


def grafico_roc(resultados, y_test):
    """Curvas ROC comparativas."""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['surface'])

    cores = [COLORS['green'], COLORS['amber'], COLORS['blue']]

    for idx, (nome, res) in enumerate(resultados.items()):
        fpr, tpr, _ = roc_curve(y_test, res['proba'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=cores[idx], linewidth=2,
                label=f'{nome} (AUC = {roc_auc:.4f})')

    ax.plot([0, 1], [0, 1], color=COLORS['dim'], linestyle='--',
            linewidth=1, label='Aleatório (AUC = 0.5)')

    ax.set_xlabel('Taxa de Falso Positivo (FPR)', color=COLORS['dim'])
    ax.set_ylabel('Taxa de Verdadeiro Positivo (TPR)', color=COLORS['dim'])
    ax.set_title('CURVA ROC · COMPARAÇÃO DE MODELOS',
                 fontsize=14, color=COLORS['text'], fontweight='bold', pad=15)
    ax.legend(fontsize=10)
    ax.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m10a_roc_curve.png', **SAVE)
    plt.close()
    print("   ✔ Gráfico salvo: graficos/m10a_roc_curve.png")


def grafico_metricas_comparacao(resultados):
    """Barras comparativas de métricas."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['surface'])

    metricas = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(metricas))
    width = 0.3
    cores = [COLORS['green'], COLORS['amber']]

    for i, (nome, res) in enumerate(resultados.items()):
        vals = [res[m] for m in metricas]
        bars = ax.bar(x + i * width, vals, width, color=cores[i],
                      alpha=0.8, label=nome)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9,
                    color=cores[i])

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                       color=COLORS['text'], fontsize=11)
    ax.set_ylabel('Score', color=COLORS['dim'])
    ax.set_title('COMPARAÇÃO DE MÉTRICAS · RANDOM FOREST vs XGBOOST',
                 fontsize=14, color=COLORS['text'], fontweight='bold', pad=15)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color=COLORS['dim'], linestyle='--', alpha=0.5,
               label='Baseline (50%)')
    ax.legend(fontsize=10)
    ax.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m10a_metricas_comparacao.png', **SAVE)
    plt.close()
    print("   ✔ Gráfico salvo: graficos/m10a_metricas_comparacao.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  MÓDULO 10A · RANDOM FOREST & XGBOOST — CLASSIFICAÇÃO")
    print("=" * 70)

    data = carregar_dados()
    resultados, X_train, X_test, y_train, y_test, feature_cols = treinar_modelos(data)

    print("\n🎨 Gerando gráficos...")
    grafico_feature_importance(resultados, feature_cols)
    grafico_confusion_matrix(resultados, y_test)
    grafico_roc(resultados, y_test)
    grafico_metricas_comparacao(resultados)

    print(f"\n{'=' * 70}")
    print("  ✅ Módulo 10A concluído!")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
