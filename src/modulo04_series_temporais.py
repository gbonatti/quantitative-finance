"""
=============================================================================
MÓDULO 04 — Conceitos Fundamentais de Séries Temporais
Quant Academy · Finanças Quantitativas
=============================================================================
Estacionariedade, decomposição, ACF, PACF e testes estatísticos.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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
    print("📥 Baixando dados da PETR4.SA (2 anos)...")
    df = yf.download('PETR4.SA', period='2y', auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Close', 'Volume']].dropna()
    df.columns = ['Close', 'Volume']
    retornos = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    print(f"   ✔ {len(df)} observações")
    return df, retornos


def teste_adf(serie, nome=''):
    """Teste Augmented Dickey-Fuller."""
    result = adfuller(serie, autolag='AIC')
    print(f"\n   📊 ADF — {nome}")
    print(f"      Estatística: {result[0]:.4f}")
    print(f"      p-value:     {result[1]:.6f}")
    print(f"      Lags:        {result[2]}")
    estacionaria = result[1] < 0.05
    print(f"      Estacionária: {'✔ SIM' if estacionaria else '✘ NÃO'} (5%)")
    return estacionaria


def teste_kpss(serie, nome=''):
    """Teste KPSS (H0: estacionária)."""
    stat, p, lags, crit = kpss(serie, regression='c', nlags='auto')
    print(f"\n   📊 KPSS — {nome}")
    print(f"      Estatística: {stat:.4f}")
    print(f"      p-value:     {p:.6f}")
    estacionaria = p >= 0.05
    print(f"      Estacionária: {'✔ SIM' if estacionaria else '✘ NÃO'} (5%)")
    return estacionaria


def grafico_preco_vs_retorno(df, retornos):
    """Demonstra que preços são não-estacionários mas retornos são."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.patch.set_facecolor(COLORS['bg'])
    fig.suptitle('ESTACIONARIEDADE · PREÇOS vs RETORNOS',
                 fontsize=16, color=COLORS['text'], fontweight='bold', y=1.02)

    # Preço
    ax = axes[0][0]
    ax.set_facecolor(COLORS['surface'])
    ax.plot(df.index, df['Close'], color=COLORS['amber'], linewidth=1)
    ax.set_title('PREÇO (NÃO-ESTACIONÁRIO)', fontsize=12,
                 color=COLORS['amber'], fontweight='bold')
    ax.set_ylabel('R$', color=COLORS['dim'])
    ax.tick_params(colors=COLORS['dim'])

    # Retornos
    ax2 = axes[0][1]
    ax2.set_facecolor(COLORS['surface'])
    ax2.plot(retornos.index, retornos * 100, color=COLORS['green'],
             linewidth=0.5, alpha=0.8)
    ax2.axhline(0, color=COLORS['dim'], linestyle='--', alpha=0.3)
    ax2.set_title('LOG-RETORNOS (ESTACIONÁRIO)', fontsize=12,
                  color=COLORS['green'], fontweight='bold')
    ax2.set_ylabel('%', color=COLORS['dim'])
    ax2.tick_params(colors=COLORS['dim'])

    # Distribuição do preço (muda ao longo do tempo)
    ax3 = axes[1][0]
    ax3.set_facecolor(COLORS['surface'])
    n = len(df)
    primeira_metade = df['Close'].iloc[:n//2]
    segunda_metade = df['Close'].iloc[n//2:]
    ax3.hist(primeira_metade, bins=40, alpha=0.6, color=COLORS['blue'],
             label='1ª Metade', density=True)
    ax3.hist(segunda_metade, bins=40, alpha=0.6, color=COLORS['amber'],
             label='2ª Metade', density=True)
    ax3.set_title('DISTRIBUIÇÃO DO PREÇO (MUDA)', fontsize=11,
                  color=COLORS['dim'])
    ax3.legend(fontsize=9)
    ax3.tick_params(colors=COLORS['dim'])

    # Distribuição dos retornos (estável)
    ax4 = axes[1][1]
    ax4.set_facecolor(COLORS['surface'])
    ret1 = retornos.iloc[:len(retornos)//2]
    ret2 = retornos.iloc[len(retornos)//2:]
    ax4.hist(ret1 * 100, bins=40, alpha=0.6, color=COLORS['blue'],
             label='1ª Metade', density=True)
    ax4.hist(ret2 * 100, bins=40, alpha=0.6, color=COLORS['green'],
             label='2ª Metade', density=True)
    ax4.set_title('DISTRIBUIÇÃO DOS RETORNOS (ESTÁVEL)', fontsize=11,
                  color=COLORS['dim'])
    ax4.legend(fontsize=9)
    ax4.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m04_estacionariedade.png', **SAVE)
    plt.close()
    print("   ✔ graficos/m04_estacionariedade.png")


def grafico_decomposicao(df):
    """Decomposição da série temporal."""
    # Resample para semanal para melhor visualização da decomposição
    weekly = df['Close'].resample('W').last().dropna()

    decomp = seasonal_decompose(weekly, model='additive', period=13)

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.patch.set_facecolor(COLORS['bg'])
    fig.suptitle('DECOMPOSIÇÃO DA SÉRIE TEMPORAL · PETR4 (SEMANAL)',
                 fontsize=14, color=COLORS['text'], fontweight='bold', y=1.01)

    componentes = [
        ('OBSERVADO', decomp.observed, COLORS['text']),
        ('TENDÊNCIA', decomp.trend, COLORS['green']),
        ('SAZONALIDADE', decomp.seasonal, COLORS['blue']),
        ('RESÍDUO', decomp.resid, COLORS['amber']),
    ]

    for ax, (titulo, dados, cor) in zip(axes, componentes):
        ax.set_facecolor(COLORS['surface'])
        ax.plot(dados, color=cor, linewidth=1)
        ax.set_title(titulo, fontsize=11, color=cor, loc='left')
        ax.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m04_decomposicao.png', **SAVE)
    plt.close()
    print("   ✔ graficos/m04_decomposicao.png")


def _plot_acf_manual(ax, valores, nlags, cor, titulo):
    """Plota ACF/PACF manualmente com barras estilizadas."""
    ax.set_facecolor(COLORS['surface'])
    n = len(valores)
    conf = 1.96 / np.sqrt(n)

    lags_range = range(len(valores[:nlags+1]))
    ax.bar(lags_range, valores[:nlags+1], width=0.3, color=cor, alpha=0.8, edgecolor='none')
    ax.axhline(0, color=COLORS['dim'], linewidth=0.5)
    ax.axhline(conf, color=COLORS['red'], linestyle='--', alpha=0.6, linewidth=1)
    ax.axhline(-conf, color=COLORS['red'], linestyle='--', alpha=0.6, linewidth=1)
    ax.fill_between(lags_range, -conf, conf, alpha=0.05, color=COLORS['red'])
    ax.set_title(titulo, fontsize=11, color=cor)
    ax.set_ylim(-0.15, 0.15) if abs(valores[1:nlags+1]).max() < 0.15 else None
    ax.tick_params(colors=COLORS['dim'])


def grafico_acf_pacf(retornos):
    """ACF e PACF dos retornos e retornos ao quadrado."""
    nlags = 40

    acf_ret = acf(retornos, nlags=nlags)
    pacf_ret = pacf(retornos, nlags=nlags)
    acf_ret2 = acf(retornos ** 2, nlags=nlags)
    pacf_ret2 = pacf(retornos ** 2, nlags=nlags)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(COLORS['bg'])
    fig.suptitle('ACF & PACF · RETORNOS E RETORNOS²',
                 fontsize=14, color=COLORS['text'], fontweight='bold', y=1.02)

    _plot_acf_manual(axes[0][0], acf_ret, nlags, COLORS['green'], 'ACF — Retornos')
    _plot_acf_manual(axes[0][1], pacf_ret, nlags, COLORS['green'], 'PACF — Retornos')
    _plot_acf_manual(axes[1][0], acf_ret2, nlags, COLORS['amber'], 'ACF — Retornos²')
    _plot_acf_manual(axes[1][1], pacf_ret2, nlags, COLORS['amber'], 'PACF — Retornos²')

    # Ajustar ylim dos retornos² para ver a persistência
    for ax in axes[1]:
        ax.set_ylim(-0.1, max(0.3, acf_ret2[1:].max() * 1.2))

    plt.tight_layout()
    plt.savefig('../graficos/m04_acf_pacf.png', **SAVE)
    plt.close()
    print("   ✔ graficos/m04_acf_pacf.png")

    print("\n   💡 ACF de retornos² significativa → indica volatility clustering (efeito ARCH)")


def grafico_rolling_stats(df, retornos):
    """Média e desvio padrão rolling para verificar estacionariedade visual."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    fig.patch.set_facecolor(COLORS['bg'])

    # Preço - rolling stats (não-estacionário)
    ax = axes[0]
    ax.set_facecolor(COLORS['surface'])
    preco = df['Close']
    ax.plot(preco, color=COLORS['amber'], alpha=0.5, linewidth=0.8, label='Preço')
    ax.plot(preco.rolling(60).mean(), color=COLORS['red'], linewidth=2,
            label='Média Rolling (60d)')
    ax.fill_between(preco.index,
                    preco.rolling(60).mean() - preco.rolling(60).std(),
                    preco.rolling(60).mean() + preco.rolling(60).std(),
                    alpha=0.15, color=COLORS['red'])
    ax.set_title('PREÇO · MÉDIA E DESVIO ROLLING (NÃO-ESTACIONÁRIO)',
                 fontsize=12, color=COLORS['amber'], fontweight='bold')
    ax.legend(fontsize=9)
    ax.tick_params(colors=COLORS['dim'])

    # Retornos - rolling stats (estacionário)
    ax2 = axes[1]
    ax2.set_facecolor(COLORS['surface'])
    ax2.plot(retornos * 100, color=COLORS['green'], alpha=0.3, linewidth=0.5)
    ax2.plot(retornos.rolling(60).mean() * 100, color=COLORS['blue'],
             linewidth=2, label='Média Rolling (60d)')
    ax2.fill_between(retornos.index,
                     (retornos.rolling(60).mean() - retornos.rolling(60).std()) * 100,
                     (retornos.rolling(60).mean() + retornos.rolling(60).std()) * 100,
                     alpha=0.15, color=COLORS['blue'])
    ax2.axhline(0, color=COLORS['dim'], linestyle='--', alpha=0.3)
    ax2.set_title('RETORNOS · MÉDIA E DESVIO ROLLING (ESTACIONÁRIO)',
                  fontsize=12, color=COLORS['green'], fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m04_rolling_stats.png', **SAVE)
    plt.close()
    print("   ✔ graficos/m04_rolling_stats.png")


def main():
    print("=" * 70)
    print("  MÓDULO 04 · SÉRIES TEMPORAIS — CONCEITOS FUNDAMENTAIS")
    print("=" * 70)

    df, retornos = carregar_dados()

    # ── Testes de Estacionariedade ────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  TESTES DE ESTACIONARIEDADE")
    print(f"{'─' * 70}")

    teste_adf(df['Close'], 'Preço PETR4')
    teste_kpss(df['Close'], 'Preço PETR4')

    teste_adf(retornos, 'Log-Retornos PETR4')
    teste_kpss(retornos, 'Log-Retornos PETR4')

    # ── Gráficos ──────────────────────────────────────────────────────────
    print("\n🎨 Gerando gráficos...")
    grafico_preco_vs_retorno(df, retornos)
    grafico_decomposicao(df)
    grafico_acf_pacf(retornos)
    grafico_rolling_stats(df, retornos)

    print(f"\n{'=' * 70}")
    print("  ✅ Módulo 04 concluído!")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
