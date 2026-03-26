"""
=============================================================================
MÓDULO 05 — Modelos ARIMA & SARIMA
Quant Academy · Finanças Quantitativas
=============================================================================
Seleção de modelo, previsão e diagnósticos de resíduos.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
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


def carregar_dados():
    import yfinance as yf
    print("📥 Baixando dados do Ibovespa (^BVSP, 3 anos)...")
    df = yf.download('^BVSP', period='3y', auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    serie = df['Close'].dropna()
    print(f"   ✔ {len(serie)} observações")
    return serie


def selecao_modelo(serie):
    """Compara diferentes ordens ARIMA via AIC/BIC."""
    print("\n📊 Seleção de modelo — comparando ordens ARIMA...")

    retornos = np.log(serie / serie.shift(1)).dropna()

    resultados = []
    ordens = [(1,0,0), (1,0,1), (2,0,1), (2,1,1), (3,1,1),
              (1,1,0), (2,1,0), (0,1,1), (0,1,2), (1,1,1)]

    for ordem in ordens:
        try:
            model = ARIMA(serie, order=ordem)
            fit = model.fit()
            resultados.append({
                'Ordem': f'ARIMA{ordem}',
                'AIC': fit.aic,
                'BIC': fit.bic,
                'Log-Lik': fit.llf
            })
        except Exception:
            pass

    df_res = pd.DataFrame(resultados).sort_values('AIC')
    print(f"\n   {'Ordem':<18} {'AIC':>12} {'BIC':>12} {'Log-Lik':>12}")
    print(f"   {'─' * 54}")
    for _, row in df_res.head(6).iterrows():
        print(f"   {row['Ordem']:<18} {row['AIC']:>12,.1f} {row['BIC']:>12,.1f} {row['Log-Lik']:>12,.1f}")

    melhor = df_res.iloc[0]['Ordem']
    print(f"\n   🏆 Melhor modelo (AIC): {melhor}")

    return df_res


def auto_arima_fit(serie):
    """Seleção automática com pmdarima."""
    try:
        from pmdarima import auto_arima
        print("\n📊 Auto-ARIMA (pmdarima)...")
        modelo = auto_arima(
            serie, start_p=0, max_p=5, start_q=0, max_q=5,
            d=None, seasonal=False, ic='aic',
            stepwise=True, suppress_warnings=True, trace=False
        )
        print(f"   ✔ Melhor ordem: ARIMA{modelo.order}")
        print(f"   ✔ AIC: {modelo.aic():.1f}")
        return modelo
    except ImportError:
        print("   ⚠ pmdarima não instalado. Usando ARIMA(2,1,1).")
        return None


def ajustar_e_prever(serie, ordem=(2, 1, 1), n_forecast=30):
    """Ajusta ARIMA e faz previsão."""
    print(f"\n📈 Ajustando ARIMA{ordem} e prevendo {n_forecast} passos...")

    model = ARIMA(serie, order=ordem)
    result = model.fit()

    # Previsão
    forecast_result = result.get_forecast(steps=n_forecast)
    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()

    # Resíduos
    residuos = result.resid

    # Ljung-Box
    lb = acorr_ljungbox(residuos, lags=[10, 20], return_df=True)
    print(f"\n   Diagnóstico de Resíduos:")
    print(f"      Média: {residuos.mean():.4f}")
    print(f"      Desvio: {residuos.std():.4f}")
    print(f"      Ljung-Box (lag=20) p-value: {lb['lb_pvalue'].iloc[-1]:.4f}")

    return result, forecast, conf_int, residuos


def grafico_previsao(serie, forecast, conf_int, ordem):
    """Série original + previsão com intervalo de confiança."""
    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['surface'])

    # Últimos 200 dias + previsão
    n_show = 200
    ax.plot(serie.index[-n_show:], serie.values[-n_show:],
            color=COLORS['text'], linewidth=1.5, label='Observado')

    ax.plot(forecast.index, forecast.values,
            color=COLORS['green'], linewidth=2, label=f'Previsão ARIMA{ordem}')

    ax.fill_between(conf_int.index,
                    conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                    alpha=0.2, color=COLORS['green'], label='IC 95%')

    ax.axvline(serie.index[-1], color=COLORS['amber'], linestyle='--',
               alpha=0.7, label='Início Previsão')

    ax.set_title(f'PREVISÃO ARIMA{ordem} · IBOVESPA',
                 fontsize=14, color=COLORS['text'], fontweight='bold', pad=15)
    ax.set_xlabel('Data', color=COLORS['dim'])
    ax.set_ylabel('Pontos', color=COLORS['dim'])
    ax.legend(fontsize=10)
    ax.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m05_arima_previsao.png', **SAVE)
    plt.close()
    print("   ✔ graficos/m05_arima_previsao.png")


def grafico_diagnostico_residuos(residuos, ordem):
    """Diagnóstico completo dos resíduos."""
    from scipy import stats

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(COLORS['bg'])
    fig.suptitle(f'DIAGNÓSTICO DE RESÍDUOS · ARIMA{ordem}',
                 fontsize=14, color=COLORS['text'], fontweight='bold', y=1.02)

    # 1. Resíduos no tempo
    ax = axes[0][0]
    ax.set_facecolor(COLORS['surface'])
    ax.plot(residuos, color=COLORS['green'], linewidth=0.5, alpha=0.8)
    ax.axhline(0, color=COLORS['dim'], linestyle='--')
    ax.set_title('Resíduos ao Longo do Tempo', fontsize=11, color=COLORS['green'])
    ax.tick_params(colors=COLORS['dim'])

    # 2. Histograma dos resíduos
    ax2 = axes[0][1]
    ax2.set_facecolor(COLORS['surface'])
    ax2.hist(residuos, bins=50, color=COLORS['blue'], alpha=0.7,
             density=True, edgecolor='none')
    # Curva normal ajustada
    x_norm = np.linspace(residuos.min(), residuos.max(), 100)
    ax2.plot(x_norm, stats.norm.pdf(x_norm, residuos.mean(), residuos.std()),
             color=COLORS['amber'], linewidth=2, label='Normal')
    ax2.set_title('Distribuição dos Resíduos', fontsize=11, color=COLORS['blue'])
    ax2.legend(fontsize=9)
    ax2.tick_params(colors=COLORS['dim'])

    # 3. ACF dos resíduos
    ax3 = axes[1][0]
    ax3.set_facecolor(COLORS['surface'])
    plot_acf(residuos, lags=30, ax=ax3, color=COLORS['amber'])
    ax3.set_title('ACF dos Resíduos', fontsize=11, color=COLORS['amber'])
    ax3.tick_params(colors=COLORS['dim'])

    # 4. QQ-Plot
    ax4 = axes[1][1]
    ax4.set_facecolor(COLORS['surface'])
    stats.probplot(residuos, dist='norm', plot=ax4)
    ax4.get_lines()[0].set_markerfacecolor(COLORS['blue'])
    ax4.get_lines()[0].set_markeredgecolor(COLORS['blue'])
    ax4.get_lines()[0].set_markersize(3)
    ax4.get_lines()[1].set_color(COLORS['green'])
    ax4.set_title('QQ-Plot vs Normal', fontsize=11, color=COLORS['text'])
    ax4.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m05_diagnostico_residuos.png', **SAVE)
    plt.close()
    print("   ✔ graficos/m05_diagnostico_residuos.png")


def grafico_comparacao_aic(df_res):
    """Comparação visual de AIC/BIC entre modelos."""
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['surface'])

    df_plot = df_res.head(8).sort_values('AIC')
    x = np.arange(len(df_plot))
    width = 0.35

    ax.bar(x - width/2, df_plot['AIC'], width, color=COLORS['green'],
           alpha=0.8, label='AIC')
    ax.bar(x + width/2, df_plot['BIC'], width, color=COLORS['blue'],
           alpha=0.8, label='BIC')

    ax.set_xticks(x)
    ax.set_xticklabels(df_plot['Ordem'], color=COLORS['text'], rotation=45)
    ax.set_ylabel('Critério de Informação', color=COLORS['dim'])
    ax.set_title('COMPARAÇÃO AIC / BIC ENTRE MODELOS',
                 fontsize=14, color=COLORS['text'], fontweight='bold', pad=15)
    ax.legend(fontsize=10)
    ax.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m05_comparacao_aic_bic.png', **SAVE)
    plt.close()
    print("   ✔ graficos/m05_comparacao_aic_bic.png")


def main():
    print("=" * 70)
    print("  MÓDULO 05 · MODELOS ARIMA & SARIMA")
    print("=" * 70)

    serie = carregar_dados()

    # Seleção de modelo
    df_res = selecao_modelo(serie)

    # Auto-ARIMA
    auto_model = auto_arima_fit(serie)

    # Usar a melhor ordem
    if auto_model:
        ordem = auto_model.order
    else:
        ordem = (2, 1, 1)

    # Ajustar e prever
    result, forecast, conf_int, residuos = ajustar_e_prever(serie, ordem, n_forecast=30)

    # Gráficos
    print("\n🎨 Gerando gráficos...")
    grafico_previsao(serie, forecast, conf_int, ordem)
    grafico_diagnostico_residuos(residuos, ordem)
    grafico_comparacao_aic(df_res)

    print(f"\n{'=' * 70}")
    print("  ✅ Módulo 05 concluído!")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
