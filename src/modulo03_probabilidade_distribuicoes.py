"""
=============================================================================
MÓDULO 03 — Probabilidade & Distribuições
=============================================================================
Quant Academy · Finanças Quantitativas com Python

Conteúdo:
  - Distribuições: Normal, t-Student, Log-Normal
  - Fitting de distribuições em dados reais
  - Teorema do Limite Central (demonstração)
  - VaR Paramétrico (Normal e t-Student)
  - QQ-Plot e testes de normalidade

Dados reais: IBOVESPA (^BVSP) via yfinance
=============================================================================
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
import os

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor': '#0d1520', 'axes.facecolor': '#0d1520',
    'axes.edgecolor': '#1e3048', 'axes.labelcolor': '#c8d8e8',
    'text.color': '#c8d8e8', 'xtick.color': '#6a8aa8',
    'ytick.color': '#6a8aa8', 'grid.color': '#1e3048',
    'grid.alpha': 0.5, 'font.family': 'monospace', 'font.size': 10,
    'figure.dpi': 120,
})

COLORS = {
    'green': '#00d4a0', 'amber': '#f5a623', 'red': '#e05252',
    'blue': '#4a9eff', 'cyan': '#7ec8e3', 'purple': '#b07ee8',
}

GRAFICOS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'graficos')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(GRAFICOS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def baixar_dados():
    """Baixa dados do IBOVESPA e algumas ações."""
    print("=" * 60)
    print("  MÓDULO 03 — PROBABILIDADE & DISTRIBUIÇÕES")
    print("=" * 60)

    tickers = ['^BVSP', 'PETR4.SA', 'VALE3.SA', 'ITUB4.SA']
    nomes = ['IBOVESPA', 'PETR4', 'VALE3', 'ITUB4']

    dados = yf.download(tickers, start='2015-01-01', end='2025-12-31',
                        auto_adjust=True, progress=False)
    precos = dados['Close']
    precos.columns = nomes
    precos = precos.dropna()

    log_retornos = np.log(precos / precos.shift(1)).dropna()

    print(f"\n✓ {len(log_retornos)} dias | Período: {log_retornos.index[0].strftime('%Y')} - {log_retornos.index[-1].strftime('%Y')}")
    return precos, log_retornos


def grafico_comparacao_distribuicoes(log_retornos):
    """Gráfico 1: Comparação Normal vs t-Student vs Log-Normal."""
    print("\n── Fitting de Distribuições ─────────────────────────────────")

    r = log_retornos['IBOVESPA'].dropna().values

    # Fit distribuições
    mu_norm, sigma_norm = stats.norm.fit(r)
    df_t, loc_t, scale_t = stats.t.fit(r)
    shape_ln, loc_ln, scale_ln = stats.lognorm.fit(r[r > 0])  # Log-normal precisa > 0

    print(f"\n  Normal:     μ = {mu_norm:.6f}, σ = {sigma_norm:.6f}")
    print(f"  t-Student:  ν = {df_t:.2f}, loc = {loc_t:.6f}, scale = {scale_t:.6f}")

    # Testes de aderência (KS test)
    ks_norm = stats.kstest(r, 'norm', args=(mu_norm, sigma_norm))
    ks_t = stats.kstest(r, 't', args=(df_t, loc_t, scale_t))

    print(f"\n  KS Test Normal:    stat={ks_norm.statistic:.4f}, p={ks_norm.pvalue:.4f}")
    print(f"  KS Test t-Student: stat={ks_t.statistic:.4f}, p={ks_t.pvalue:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Gráfico 1: Histograma + fits ──
    ax = axes[0, 0]
    x = np.linspace(r.min(), r.max(), 500)

    ax.hist(r, bins=100, density=True, alpha=0.5, color=COLORS['blue'],
            edgecolor='none', label='Dados reais')
    ax.plot(x, stats.norm.pdf(x, mu_norm, sigma_norm),
            color=COLORS['amber'], linewidth=2, label=f'Normal(σ={sigma_norm:.4f})')
    ax.plot(x, stats.t.pdf(x, df_t, loc_t, scale_t),
            color=COLORS['red'], linewidth=2, linestyle='--',
            label=f't-Student(ν={df_t:.1f})')

    ax.set_title('FITTING — IBOVESPA LOG-RETORNOS',
                 fontsize=11, fontweight='bold', color='#e8f4ff')
    ax.legend(fontsize=8, framealpha=0.3, edgecolor='#1e3048')
    ax.grid(True, alpha=0.3)

    # ── Gráfico 2: Zoom nas caudas ──
    ax = axes[0, 1]
    # Cauda esquerda
    mask_left = r < np.percentile(r, 5)
    ax.hist(r[mask_left], bins=50, density=True, alpha=0.5,
            color=COLORS['red'], edgecolor='none', label='Cauda esquerda')

    x_left = np.linspace(r.min(), np.percentile(r, 5), 200)
    ax.plot(x_left, stats.norm.pdf(x_left, mu_norm, sigma_norm),
            color=COLORS['amber'], linewidth=2, label='Normal')
    ax.plot(x_left, stats.t.pdf(x_left, df_t, loc_t, scale_t),
            color=COLORS['green'], linewidth=2, linestyle='--', label='t-Student')

    ax.set_title('ZOOM NAS CAUDAS (Fat Tails)',
                 fontsize=11, fontweight='bold', color='#e8f4ff')
    ax.legend(fontsize=8, framealpha=0.3, edgecolor='#1e3048')
    ax.grid(True, alpha=0.3)

    # ── Gráfico 3: QQ-Plot Normal ──
    ax = axes[1, 0]
    stats.probplot(r, dist='norm', plot=ax)
    ax.get_lines()[0].set(color=COLORS['cyan'], markersize=2, alpha=0.5)
    ax.get_lines()[1].set(color=COLORS['red'], linewidth=1.5)
    ax.set_title('QQ-PLOT vs NORMAL', fontsize=11, fontweight='bold', color='#e8f4ff')
    ax.grid(True, alpha=0.3)

    # ── Gráfico 4: QQ-Plot t-Student ──
    ax = axes[1, 1]
    stats.probplot(r, dist='t', sparams=(df_t,), plot=ax)
    ax.get_lines()[0].set(color=COLORS['cyan'], markersize=2, alpha=0.5)
    ax.get_lines()[1].set(color=COLORS['green'], linewidth=1.5)
    ax.set_title(f'QQ-PLOT vs t-STUDENT(ν={df_t:.1f})',
                 fontsize=11, fontweight='bold', color='#e8f4ff')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, 'm03_distribuicoes_fitting.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Gráfico salvo: {path}")


def grafico_teorema_limite_central():
    """Gráfico 2: Demonstração visual do TLC."""
    print("\n── Teorema do Limite Central ────────────────────────────────")

    np.random.seed(42)
    n_amostras = 10000

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    distribuicoes = [
        ('Exponencial(λ=1)', lambda n: np.random.exponential(1, n)),
        ('Uniforme(0,1)', lambda n: np.random.uniform(0, 1, n)),
        ('Bernoulli(p=0.3)', lambda n: np.random.binomial(1, 0.3, n)),
    ]

    for col, (nome, gen) in enumerate(distribuicoes):
        # Distribuição original
        ax = axes[0, col]
        amostra = gen(n_amostras)
        ax.hist(amostra, bins=50, density=True, alpha=0.7,
                color=COLORS['blue'], edgecolor='none')
        ax.set_title(f'ORIGINAL: {nome}', fontsize=10, fontweight='bold', color='#e8f4ff')
        ax.grid(True, alpha=0.3)

        # Média de n=30 amostras (TLC)
        ax2 = axes[1, col]
        medias = np.array([gen(30).mean() for _ in range(n_amostras)])
        ax2.hist(medias, bins=50, density=True, alpha=0.7,
                 color=COLORS['green'], edgecolor='none', label='Médias (n=30)')

        # Fit normal
        mu, sigma = medias.mean(), medias.std()
        x = np.linspace(medias.min(), medias.max(), 200)
        ax2.plot(x, stats.norm.pdf(x, mu, sigma), color=COLORS['amber'],
                 linewidth=2, label=f'Normal fit')

        ax2.set_title(f'TLC: MÉDIA DE 30 AMOSTRAS', fontsize=10,
                      fontweight='bold', color='#e8f4ff')
        ax2.legend(fontsize=8, framealpha=0.3)
        ax2.grid(True, alpha=0.3)

    plt.suptitle('TEOREMA DO LIMITE CENTRAL — Qualquer distribuição → Normal',
                 fontsize=14, fontweight='bold', color='#e8f4ff', y=1.02)
    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, 'm03_teorema_limite_central.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Gráfico salvo: {path}")


def grafico_var_parametrico(log_retornos):
    """Gráfico 3: VaR paramétrico — Normal vs t-Student."""
    print("\n── Value at Risk (VaR) Paramétrico ──────────────────────────")

    r = log_retornos['IBOVESPA'].dropna()
    mu, sigma = r.mean(), r.std()
    df_t, loc_t, scale_t = stats.t.fit(r)

    # Calcular VaR para diferentes níveis
    niveis = [0.90, 0.95, 0.99, 0.999]
    investimento = 1_000_000  # R$ 1 milhão

    print(f"\n  Investimento: R$ {investimento:,.0f}")
    print(f"\n  {'Nível':<10} {'VaR Normal':>15} {'VaR t-Student':>15} {'VaR Histórico':>15}")
    print("  " + "-" * 58)

    for nivel in niveis:
        z = stats.norm.ppf(1 - nivel)
        var_norm = -(mu + z * sigma) * investimento
        var_t = -stats.t.ppf(1 - nivel, df_t, loc_t, scale_t) * investimento
        var_hist = -np.percentile(r, (1 - nivel) * 100) * investimento
        print(f"  {nivel:.1%}     R$ {var_norm:>12,.0f}  R$ {var_t:>12,.0f}  R$ {var_hist:>12,.0f}")

    # Gráfico
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # VaR visual
    ax = axes[0]
    x = np.linspace(r.min(), r.max(), 500)

    # Distribuição
    pdf_norm = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, pdf_norm, color=COLORS['blue'], linewidth=2, label='Normal')
    ax.fill_between(x, pdf_norm, where=(x < stats.norm.ppf(0.05, mu, sigma)),
                    alpha=0.4, color=COLORS['red'], label='VaR 95% (perda)')

    var_95_norm = mu - 1.645 * sigma
    var_99_norm = mu - 2.326 * sigma
    var_95_t = stats.t.ppf(0.05, df_t, loc_t, scale_t)

    ax.axvline(x=var_95_norm, color=COLORS['amber'], linestyle='--',
               linewidth=1.5, label=f'VaR 95% Normal = {var_95_norm:.4f}')
    ax.axvline(x=var_99_norm, color=COLORS['red'], linestyle='--',
               linewidth=1.5, label=f'VaR 99% Normal = {var_99_norm:.4f}')
    ax.axvline(x=var_95_t, color=COLORS['green'], linestyle=':',
               linewidth=1.5, label=f'VaR 95% t-Student = {var_95_t:.4f}')

    ax.set_title('VaR PARAMÉTRICO — IBOVESPA',
                 fontsize=11, fontweight='bold', color='#e8f4ff')
    ax.set_xlabel('Log-Return Diário')
    ax.legend(fontsize=7, framealpha=0.3, edgecolor='#1e3048', loc='upper left')
    ax.grid(True, alpha=0.3)

    # Rolling VaR
    ax2 = axes[1]
    window = 252
    rolling_mu = r.rolling(window).mean()
    rolling_sigma = r.rolling(window).std()
    var_95_rolling = rolling_mu - 1.645 * rolling_sigma

    ax2.plot(r.index, r, alpha=0.3, linewidth=0.5, color=COLORS['blue'], label='Retornos')
    ax2.plot(var_95_rolling.index, var_95_rolling, color=COLORS['red'],
             linewidth=1.5, label='VaR 95% Rolling (252d)')
    ax2.fill_between(var_95_rolling.index, var_95_rolling, r.min(),
                     alpha=0.1, color=COLORS['red'])

    # Violações
    violacoes = r < var_95_rolling
    ax2.scatter(r.index[violacoes], r[violacoes], color=COLORS['red'],
                s=10, zorder=5, alpha=0.7, label=f'Violações ({violacoes.sum()})')

    ax2.set_title('VaR 95% ROLLING (252 dias)',
                  fontsize=11, fontweight='bold', color='#e8f4ff')
    ax2.legend(fontsize=8, framealpha=0.3, edgecolor='#1e3048')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, 'm03_var_parametrico.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Gráfico salvo: {path}")


def grafico_comparacao_ativos(log_retornos):
    """Gráfico 4: Comparação de distribuições entre ativos."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ativos = log_retornos.columns
    cores = [COLORS['green'], COLORS['blue'], COLORS['amber'], COLORS['red']]

    for i, (ativo, cor) in enumerate(zip(ativos, cores)):
        ax = axes[i // 2, i % 2]
        r = log_retornos[ativo].dropna()

        ax.hist(r, bins=80, density=True, alpha=0.6, color=cor, edgecolor='none')

        # Normal fit
        mu, sigma = r.mean(), r.std()
        x = np.linspace(r.min(), r.max(), 300)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), color='white',
                linewidth=1.5, linestyle='--', alpha=0.7)

        # Estatísticas no gráfico
        texto = (f'μ={mu:.5f}  σ={sigma:.4f}\n'
                 f'Skew={r.skew():.2f}  Kurt={r.kurtosis():.2f}')
        ax.text(0.98, 0.95, texto, transform=ax.transAxes, fontsize=8,
                va='top', ha='right', color='#c8d8e8',
                bbox=dict(boxstyle='round', facecolor='#0d1520', edgecolor='#1e3048', alpha=0.9))

        ax.set_title(f'DISTRIBUIÇÃO — {ativo}', fontsize=11,
                     fontweight='bold', color='#e8f4ff')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, 'm03_comparacao_distribuicoes.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Gráfico salvo: {path}")


def main():
    precos, log_retornos = baixar_dados()

    print("\n── Gerando Gráficos ────────────────────────────────────────")
    grafico_comparacao_distribuicoes(log_retornos)
    grafico_teorema_limite_central()
    grafico_var_parametrico(log_retornos)
    grafico_comparacao_ativos(log_retornos)

    print("\n" + "=" * 60)
    print("  MÓDULO 03 CONCLUÍDO ✓")
    print("  4 gráficos gerados em /graficos/")
    print("=" * 60)


if __name__ == '__main__':
    main()
