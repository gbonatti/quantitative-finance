"""
=============================================================================
MÓDULO 07 — Previsão & Backtesting
Quant Academy · Finanças Quantitativas
=============================================================================
Como avaliar corretamente a qualidade de previsões financeiras.
Walk-forward validation, métricas de erro e comparação de modelos.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── Tema dos gráficos ──────────────────────────────────────────────────────
plt.style.use('dark_background')
COLORS = {
    'green': '#00d4a0', 'amber': '#f5a623',
    'blue': '#4a9eff', 'red': '#e05252',
    'bg': '#080c10', 'surface': '#0d1520',
    'text': '#c8d8e8', 'dim': '#6a8aa8'
}
SAVE = dict(dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])


# ══════════════════════════════════════════════════════════════════════════════
# 1. COLETA DE DADOS
# ══════════════════════════════════════════════════════════════════════════════
def carregar_dados():
    """Baixa dados do Ibovespa via yfinance."""
    import yfinance as yf
    print("📥 Baixando dados do Ibovespa (^BVSP)...")
    df = yf.download('^BVSP', period='3y', auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Close']].dropna()
    df.columns = ['Close']
    print(f"   ✔ {len(df)} observações de {df.index[0].date()} a {df.index[-1].date()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODELOS DE PREVISÃO
# ══════════════════════════════════════════════════════════════════════════════
def modelo_naive(train, steps=1):
    """Modelo Naïve: previsão = último valor observado."""
    return [train.iloc[-1]] * steps


def modelo_media_movel(train, steps=1, janela=20):
    """Modelo de Média Móvel simples."""
    media = train.iloc[-janela:].mean()
    return [media] * steps


def modelo_exp_smoothing(train, steps=1):
    """Suavização Exponencial Simples (SES)."""
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    # Usar valores numéricos sem índice de datas para evitar warnings de frequência
    model = SimpleExpSmoothing(train.values, initialization_method='estimated')
    fit = model.fit(optimized=True)
    fc = fit.forecast(steps)
    return fc.tolist()


def modelo_arima(train, steps=1):
    """ARIMA com seleção automática de ordem."""
    import sys
    from statsmodels.tsa.arima.model import ARIMA
    try:
        # Usar valores numéricos sem índice de datas para evitar warnings de frequência
        # method='css' é mais robusto numericamente que MLE para séries financeiras
        model = ARIMA(train.values, order=(2, 1, 1))
        fit = model.fit(method_kwargs={'maxiter': 100})
        fc = fit.forecast(steps=steps)
        result = fc.tolist()
        # Checar se previsão é válida (sem NaN/Inf)
        if any(np.isnan(v) or np.isinf(v) for v in result):
            return modelo_naive(train, steps)
        return result
    except BaseException as e:
        # Captura qualquer erro numérico (LinAlgError, FloatingPointError, etc.)
        return modelo_naive(train, steps)


# ══════════════════════════════════════════════════════════════════════════════
# 3. WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def walk_forward(serie, model_func, train_size, n_steps=1, nome='Modelo'):
    """
    Validação walk-forward: re-treina o modelo em janela expansível,
    fazendo previsões 1-step-ahead.
    """
    predictions = []
    actuals = []
    indices = []

    total_iters = len(serie) - train_size - n_steps + 1
    print(f"   ⏳ Walk-forward {nome}: {total_iters} iterações...")

    for i in range(train_size, len(serie) - n_steps + 1):
        train = serie.iloc[:i]
        test_vals = serie.iloc[i:i + n_steps].values

        try:
            pred = model_func(train, steps=n_steps)
        except BaseException:
            pred = [train.iloc[-1]] * n_steps

        predictions.extend(pred)
        actuals.extend(test_vals)
        indices.extend(serie.index[i:i + n_steps])

    return np.array(predictions), np.array(actuals), indices


def calcular_metricas(actual, predicted, nome=''):
    """Calcula MAE, RMSE e MAPE."""
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    # MAPE com proteção contra divisão por zero
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

    print(f"\n   📊 {nome}")
    print(f"      MAE  = {mae:,.2f}")
    print(f"      RMSE = {rmse:,.2f}")
    print(f"      MAPE = {mape:.2f}%")

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


# ══════════════════════════════════════════════════════════════════════════════
# 4. GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════
def grafico_walkforward(serie, resultados, metricas_dict):
    """Gráfico principal: real vs previsões de cada modelo."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                              gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor(COLORS['bg'])
    for ax in axes:
        ax.set_facecolor(COLORS['surface'])

    cores = [COLORS['green'], COLORS['amber'], COLORS['blue'], COLORS['red']]

    # --- Painel Superior: Previsões vs Realizado ---
    ax = axes[0]
    # Plotar série real
    ax.plot(serie.index, serie.values, color=COLORS['text'], alpha=0.4,
            linewidth=0.8, label='Realizado')

    for i, (nome, res) in enumerate(resultados.items()):
        preds, actuals, idx = res
        ax.plot(idx, preds, color=cores[i % len(cores)],
                linewidth=0.8, alpha=0.8, label=f'{nome}')

    ax.set_title('WALK-FORWARD BACKTESTING · PREVISÃO vs REALIZADO',
                 fontsize=14, color=COLORS['text'], fontweight='bold', pad=15)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_ylabel('Preço (pts)', color=COLORS['dim'])
    ax.tick_params(colors=COLORS['dim'])

    # --- Painel Inferior: Comparação de métricas ---
    ax2 = axes[1]
    nomes = list(metricas_dict.keys())
    mape_vals = [metricas_dict[n]['MAPE'] for n in nomes]
    rmse_vals = [metricas_dict[n]['RMSE'] for n in nomes]

    x = np.arange(len(nomes))
    width = 0.35

    bars1 = ax2.bar(x - width/2, mape_vals, width, color=COLORS['green'],
                    alpha=0.8, label='MAPE (%)')
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x + width/2, rmse_vals, width, color=COLORS['amber'],
                          alpha=0.8, label='RMSE')

    ax2.set_xticks(x)
    ax2.set_xticklabels(nomes, color=COLORS['text'])
    ax2.set_ylabel('MAPE (%)', color=COLORS['green'])
    ax2_twin.set_ylabel('RMSE', color=COLORS['amber'])
    ax2.tick_params(colors=COLORS['dim'])
    ax2_twin.tick_params(colors=COLORS['dim'])
    ax2.set_title('COMPARAÇÃO DE MÉTRICAS DE ERRO',
                  fontsize=11, color=COLORS['dim'], pad=10)

    # Valores nas barras
    for bar, val in zip(bars1, mape_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{val:.2f}%', ha='center', va='bottom', fontsize=8,
                 color=COLORS['green'])
    for bar, val in zip(bars2, rmse_vals):
        ax2_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                      f'{val:.0f}', ha='center', va='bottom', fontsize=8,
                      color=COLORS['amber'])

    plt.tight_layout()
    plt.savefig('../graficos/m07_walkforward_backtesting.png', **SAVE)
    plt.close()
    print("\n   ✔ Gráfico salvo: graficos/m07_walkforward_backtesting.png")


def grafico_erros(resultados):
    """Distribuição dos erros de previsão por modelo."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(COLORS['bg'])
    fig.suptitle('DISTRIBUIÇÃO DOS ERROS DE PREVISÃO',
                 fontsize=14, color=COLORS['text'], fontweight='bold', y=1.02)

    cores = [COLORS['green'], COLORS['amber'], COLORS['blue'], COLORS['red']]

    for i, (nome, res) in enumerate(resultados.items()):
        ax = axes[i // 2][i % 2]
        ax.set_facecolor(COLORS['surface'])

        preds, actuals, _ = res
        erros = actuals - preds

        ax.hist(erros, bins=50, color=cores[i], alpha=0.7, edgecolor='none',
                density=True)
        ax.axvline(0, color=COLORS['text'], linestyle='--', alpha=0.5)
        ax.axvline(np.mean(erros), color=COLORS['red'], linestyle='-',
                   alpha=0.8, label=f'Média: {np.mean(erros):,.0f}')

        ax.set_title(nome, fontsize=11, color=cores[i])
        ax.set_xlabel('Erro (Real - Previsto)', color=COLORS['dim'], fontsize=9)
        ax.legend(fontsize=8)
        ax.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m07_distribuicao_erros.png', **SAVE)
    plt.close()
    print("   ✔ Gráfico salvo: graficos/m07_distribuicao_erros.png")


def grafico_zoom(serie, resultados, ultimos_n=60):
    """Zoom nos últimos N dias para visualizar previsões."""
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['surface'])

    cores = [COLORS['green'], COLORS['amber'], COLORS['blue'], COLORS['red']]

    # Pegar os últimos N pontos
    for i, (nome, res) in enumerate(resultados.items()):
        preds, actuals, idx = res
        ax.plot(idx[-ultimos_n:], preds[-ultimos_n:],
                color=cores[i], linewidth=1.5, alpha=0.9, label=nome)

    # Real
    preds0, actuals0, idx0 = list(resultados.values())[0]
    ax.plot(idx0[-ultimos_n:], actuals0[-ultimos_n:],
            color=COLORS['text'], linewidth=2, alpha=0.8, label='Realizado',
            linestyle='--')

    ax.set_title(f'ZOOM · ÚLTIMOS {ultimos_n} DIAS DE PREVISÃO',
                 fontsize=14, color=COLORS['text'], fontweight='bold', pad=15)
    ax.legend(fontsize=10)
    ax.set_ylabel('Preço (pts)', color=COLORS['dim'])
    ax.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m07_zoom_previsoes.png', **SAVE)
    plt.close()
    print("   ✔ Gráfico salvo: graficos/m07_zoom_previsoes.png")


def grafico_tabela_metricas(metricas_dict):
    """Tabela visual com métricas de cada modelo."""
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['surface'])
    ax.axis('off')

    nomes = list(metricas_dict.keys())
    cols = ['Modelo', 'MAE', 'RMSE', 'MAPE (%)']
    cell_text = []
    for nome in nomes:
        m = metricas_dict[nome]
        cell_text.append([nome, f"{m['MAE']:,.2f}", f"{m['RMSE']:,.2f}", f"{m['MAPE']:.2f}%"])

    # Encontrar melhor modelo (menor MAPE)
    melhor = min(metricas_dict, key=lambda x: metricas_dict[x]['MAPE'])

    table = ax.table(cellText=cell_text, colLabels=cols,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Estilizar
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS['dim'])
        if row == 0:
            cell.set_facecolor('#1e3048')
            cell.set_text_props(color=COLORS['green'], fontweight='bold')
        else:
            cell.set_facecolor(COLORS['surface'])
            cell.set_text_props(color=COLORS['text'])

    ax.set_title('COMPARAÇÃO DE MODELOS · BACKTESTING WALK-FORWARD',
                 fontsize=14, color=COLORS['text'], fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('../graficos/m07_tabela_metricas.png', **SAVE)
    plt.close()
    print("   ✔ Gráfico salvo: graficos/m07_tabela_metricas.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  MÓDULO 07 · PREVISÃO & BACKTESTING")
    print("=" * 70)

    df = carregar_dados()
    serie = df['Close']

    # Tamanho de treino: 70% da série
    train_size = int(len(serie) * 0.7)
    print(f"\n📊 Treino: {train_size} obs | Teste: {len(serie) - train_size} obs")

    # ── Rodar Walk-Forward para cada modelo ────────────────────────────────
    modelos = {
        'Naïve (Random Walk)': modelo_naive,
        'Média Móvel (20d)': modelo_media_movel,
        'Suavização Exp.': modelo_exp_smoothing,
        'ARIMA(2,1,1)': modelo_arima,
    }

    resultados = {}
    metricas_dict = {}

    for nome, func in modelos.items():
        preds, actuals, idx = walk_forward(serie, func, train_size, nome=nome)
        resultados[nome] = (preds, actuals, idx)
        metricas_dict[nome] = calcular_metricas(actuals, preds, nome)

    # ── Gráficos ───────────────────────────────────────────────────────────
    print("\n🎨 Gerando gráficos...")
    grafico_walkforward(serie, resultados, metricas_dict)
    grafico_erros(resultados)
    grafico_zoom(serie, resultados, ultimos_n=60)
    grafico_tabela_metricas(metricas_dict)

    # ── Resumo Final ───────────────────────────────────────────────────────
    melhor = min(metricas_dict, key=lambda x: metricas_dict[x]['RMSE'])
    print(f"\n{'=' * 70}")
    print(f"  🏆 MELHOR MODELO (menor RMSE): {melhor}")
    print(f"     RMSE = {metricas_dict[melhor]['RMSE']:,.2f}")
    print(f"     MAPE = {metricas_dict[melhor]['MAPE']:.2f}%")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
