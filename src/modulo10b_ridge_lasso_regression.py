"""
=============================================================================
MÓDULO 10B — Ridge & Lasso Regression
Quant Academy · Finanças Quantitativas
=============================================================================
Regressão regularizada para previsão de retornos financeiros.
Comparação Ridge vs Lasso vs OLS, seleção de hiperparâmetros.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
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
    """Feature engineering para regressão de retornos."""
    data = pd.DataFrame(index=df.index)
    close = df['Close']

    # Target: retorno do próximo dia
    data['target'] = close.pct_change().shift(-1)

    # Retornos passados
    for lag in [1, 2, 3, 5, 10, 20]:
        data[f'ret_{lag}d'] = close.pct_change(lag)

    # Médias móveis ratio
    for w in [5, 10, 20, 60]:
        data[f'ma{w}_ratio'] = close / close.rolling(w).mean() - 1

    # RSI
    data['rsi'] = calcular_rsi(close)

    # Volatilidade
    data['vol_10'] = close.pct_change().rolling(10).std()
    data['vol_20'] = close.pct_change().rolling(20).std()
    data['vol_ratio'] = data['vol_10'] / data['vol_20']

    # Momentum
    data['momentum_10'] = close / close.shift(10) - 1
    data['momentum_20'] = close / close.shift(20) - 1

    # Bollinger %B
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    data['bb_pct'] = (close - (ma20 - 2 * std20)) / (4 * std20)

    return data.dropna()


def carregar_dados():
    import yfinance as yf
    print("📥 Baixando dados do Ibovespa (^BVSP, 3 anos)...")
    df = yf.download('^BVSP', period='3y', auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    print(f"   ✔ {len(df)} observações")

    data = criar_features(df)
    print(f"   ✔ {data.shape[1]-1} features criadas, {len(data)} amostras")
    return data


def treinar_modelos(data):
    """Treina OLS, Ridge e Lasso com TimeSeriesSplit."""
    feature_cols = [c for c in data.columns if c != 'target']
    X = data[feature_cols].values
    y = data['target'].values

    # Split temporal
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Normalizar
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    tscv = TimeSeriesSplit(n_splits=5)

    # ── OLS ──
    print("\n📊 Treinando OLS (sem regularização)...")
    ols = LinearRegression()
    ols.fit(X_train_sc, y_train)
    ols_pred = ols.predict(X_test_sc)

    # ── Ridge ──
    print("📊 Treinando Ridge (L2)...")
    alphas = np.logspace(-4, 4, 100)
    ridge_cv = RidgeCV(alphas=alphas, cv=tscv)
    ridge_cv.fit(X_train_sc, y_train)
    ridge_pred = ridge_cv.predict(X_test_sc)
    print(f"   ✔ Melhor alpha: {ridge_cv.alpha_:.6f}")

    # ── Lasso ──
    print("📊 Treinando Lasso (L1)...")
    lasso_cv = LassoCV(alphas=alphas, cv=tscv, max_iter=10000)
    lasso_cv.fit(X_train_sc, y_train)
    lasso_pred = lasso_cv.predict(X_test_sc)
    print(f"   ✔ Melhor alpha: {lasso_cv.alpha_:.6f}")

    resultados = {}
    for nome, pred, model in [('OLS', ols_pred, ols),
                               ('Ridge', ridge_pred, ridge_cv),
                               ('Lasso', lasso_pred, lasso_cv)]:
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        resultados[nome] = {
            'pred': pred, 'model': model,
            'rmse': rmse, 'mae': mae, 'r2': r2
        }
        print(f"\n   📊 {nome}:")
        print(f"      RMSE: {rmse:.6f}")
        print(f"      MAE:  {mae:.6f}")
        print(f"      R²:   {r2:.6f}")

    return resultados, X_train_sc, X_test_sc, y_train, y_test, feature_cols, alphas


def grafico_coeficientes(resultados, feature_cols):
    """Compara coeficientes dos 3 modelos."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.patch.set_facecolor(COLORS['bg'])

    cores = [COLORS['blue'], COLORS['green'], COLORS['amber']]

    for idx, (nome, res) in enumerate(resultados.items()):
        ax = axes[idx]
        ax.set_facecolor(COLORS['surface'])

        coefs = res['model'].coef_
        sorted_idx = np.argsort(np.abs(coefs))[-15:]

        colors = [COLORS['green'] if c > 0 else COLORS['red'] for c in coefs[sorted_idx]]
        ax.barh(range(len(sorted_idx)), coefs[sorted_idx], color=colors, alpha=0.8)
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_cols[i] for i in sorted_idx],
                           fontsize=8, color=COLORS['text'])
        ax.axvline(0, color=COLORS['dim'], linestyle='--', alpha=0.5)
        ax.set_title(f'COEFICIENTES · {nome}', fontsize=12,
                     color=COLORS['text'], fontweight='bold')
        ax.set_xlabel('Coeficiente', color=COLORS['dim'])
        ax.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m10b_coeficientes.png', **SAVE)
    plt.close()
    print("   ✔ Gráfico salvo: graficos/m10b_coeficientes.png")


def grafico_coefficient_path(X_train, y_train, feature_cols):
    """Coefficient path: como coeficientes variam com alpha."""
    alphas = np.logspace(-4, 4, 200)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(COLORS['bg'])

    for idx, (nome, Model) in enumerate([('Ridge (L2)', Ridge), ('Lasso (L1)', Lasso)]):
        ax = axes[idx]
        ax.set_facecolor(COLORS['surface'])

        coefs = []
        for a in alphas:
            model = Model(alpha=a, max_iter=10000)
            model.fit(X_train, y_train)
            coefs.append(model.coef_)

        coefs = np.array(coefs)

        for i in range(min(coefs.shape[1], 15)):
            ax.plot(alphas, coefs[:, i], linewidth=0.8, alpha=0.7)

        ax.set_xscale('log')
        ax.set_xlabel('Alpha (regularização)', color=COLORS['dim'])
        ax.set_ylabel('Coeficiente', color=COLORS['dim'])
        ax.set_title(f'COEFFICIENT PATH · {nome}', fontsize=13,
                     color=COLORS['text'], fontweight='bold', pad=15)
        ax.axhline(0, color=COLORS['dim'], linestyle='--', alpha=0.3)
        ax.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m10b_coefficient_path.png', **SAVE)
    plt.close()
    print("   ✔ Gráfico salvo: graficos/m10b_coefficient_path.png")


def grafico_previsao_vs_real(resultados, y_test):
    """Scatter plot: previsto vs realizado."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(COLORS['bg'])

    cores = [COLORS['blue'], COLORS['green'], COLORS['amber']]

    for idx, (nome, res) in enumerate(resultados.items()):
        ax = axes[idx]
        ax.set_facecolor(COLORS['surface'])

        ax.scatter(y_test * 100, res['pred'] * 100,
                   alpha=0.3, s=10, color=cores[idx])

        # Linha 45 graus
        lim = max(abs(y_test.max()), abs(y_test.min())) * 100
        ax.plot([-lim, lim], [-lim, lim], color=COLORS['dim'],
                linestyle='--', alpha=0.5)

        ax.set_xlabel('Retorno Real (%)', color=COLORS['dim'])
        ax.set_ylabel('Retorno Previsto (%)', color=COLORS['dim'])
        ax.set_title(f'{nome} · R² = {res["r2"]:.4f}', fontsize=12,
                     color=COLORS['text'], fontweight='bold')
        ax.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m10b_previsao_vs_real.png', **SAVE)
    plt.close()
    print("   ✔ Gráfico salvo: graficos/m10b_previsao_vs_real.png")


def grafico_residuos(resultados, y_test):
    """Distribuição dos resíduos."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(COLORS['bg'])

    cores = [COLORS['blue'], COLORS['green'], COLORS['amber']]

    for idx, (nome, res) in enumerate(resultados.items()):
        ax = axes[idx]
        ax.set_facecolor(COLORS['surface'])

        residuos = (y_test - res['pred']) * 100
        ax.hist(residuos, bins=50, color=cores[idx], alpha=0.7,
                edgecolor='none', density=True)
        ax.axvline(0, color=COLORS['dim'], linestyle='--')
        ax.axvline(residuos.mean(), color=COLORS['red'],
                   label=f'Média: {residuos.mean():.4f}%')

        ax.set_title(f'RESÍDUOS · {nome}', fontsize=12,
                     color=COLORS['text'], fontweight='bold')
        ax.set_xlabel('Resíduo (%)', color=COLORS['dim'])
        ax.legend(fontsize=9)
        ax.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m10b_residuos.png', **SAVE)
    plt.close()
    print("   ✔ Gráfico salvo: graficos/m10b_residuos.png")


def main():
    print("=" * 70)
    print("  MÓDULO 10B · RIDGE & LASSO REGRESSION")
    print("=" * 70)

    data = carregar_dados()
    resultados, X_train, X_test, y_train, y_test, feature_cols, alphas = treinar_modelos(data)

    print("\n🎨 Gerando gráficos...")
    grafico_coeficientes(resultados, feature_cols)
    grafico_coefficient_path(X_train, y_train, feature_cols)
    grafico_previsao_vs_real(resultados, y_test)
    grafico_residuos(resultados, y_test)

    print(f"\n{'=' * 70}")
    print("  ✅ Módulo 10B concluído!")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
