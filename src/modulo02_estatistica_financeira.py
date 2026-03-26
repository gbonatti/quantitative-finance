"""
=============================================================================
MÓDULO 02 — Matemática & Estatística Financeira
=============================================================================
Quant Academy · Finanças Quantitativas com Python

Conteúdo:
  - 4 momentos estatísticos (média, variância, assimetria, curtose)
  - Matriz de covariância e correlação
  - Heatmap de correlação
  - Movimento Browniano Geométrico (GBM)
  - Variância do portfólio (w^T Σ w)

Dados reais: ações brasileiras + IBOVESPA via yfinance
=============================================================================
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# ── Configuração de estilo ─────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1520',
    'axes.facecolor': '#0d1520',
    'axes.edgecolor': '#1e3048',
    'axes.labelcolor': '#c8d8e8',
    'text.color': '#c8d8e8',
    'xtick.color': '#6a8aa8',
    'ytick.color': '#6a8aa8',
    'grid.color': '#1e3048',
    'grid.alpha': 0.5,
    'font.family': 'monospace',
    'font.size': 10,
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
    """Baixa dados de ações brasileiras."""
    print("=" * 60)
    print("  MÓDULO 02 — ESTATÍSTICA FINANCEIRA")
    print("  Baixando dados reais...")
    print("=" * 60)

    tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA',
               'WEGE3.SA', 'RENT3.SA', 'BBAS3.SA']
    nomes = ['PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'ABEV3', 'WEGE3', 'RENT3', 'BBAS3']

    dados = yf.download(tickers, start='2020-01-01', end='2025-12-31',
                        auto_adjust=True, progress=False)
    precos = dados['Close']
    precos.columns = nomes
    precos = precos.dropna()

    log_retornos = np.log(precos / precos.shift(1)).dropna()

    print(f"\n✓ {len(precos)} dias | {len(nomes)} ativos")
    return precos, log_retornos, nomes


def momentos_estatisticos(log_retornos):
    """Calcula os 4 momentos estatísticos para cada ativo."""
    print("\n── Os 4 Momentos Estatísticos ───────────────────────────────")

    momentos = {}
    for col in log_retornos.columns:
        r = log_retornos[col]
        momentos[col] = {
            '1º Média (μ)': r.mean(),
            '2º Variância (σ²)': r.var(),
            '3º Assimetria (Skew)': r.skew(),
            '4º Curtose (Excess)': r.kurtosis(),
            'Volatilidade Diária': r.std(),
            'Volatilidade Anual (%)': r.std() * np.sqrt(252) * 100,
        }

    df = pd.DataFrame(momentos).T
    print("\n" + df.to_string(float_format='{:,.4f}'.format))

    # Teste de normalidade (Jarque-Bera)
    print("\n── Teste de Normalidade (Jarque-Bera) ───────────────────────")
    print(f"  {'Ativo':<10} {'JB Stat':>12} {'p-valor':>12} {'Normal?':>10}")
    print("  " + "-" * 46)
    for col in log_retornos.columns:
        jb_stat, p_val = stats.jarque_bera(log_retornos[col])
        normal = "Sim" if p_val > 0.05 else "NÃO"
        print(f"  {col:<10} {jb_stat:>12.2f} {p_val:>12.6f} {normal:>10}")

    return df


def grafico_momentos(log_retornos):
    """Gráfico 1: 4 momentos estatísticos em barras."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ativos = log_retornos.columns
    cores = [COLORS['green'], COLORS['blue'], COLORS['amber'], COLORS['red'],
             COLORS['cyan'], COLORS['purple'], '#ff6b9d', '#98c379']

    # Média anualizada
    ax = axes[0, 0]
    medias = log_retornos.mean() * 252 * 100
    bars = ax.bar(ativos, medias, color=cores, alpha=0.8, edgecolor='none')
    ax.set_title('1º MOMENTO — RETORNO MÉDIO ANUAL (%)', fontsize=11,
                 fontweight='bold', color='#e8f4ff')
    ax.axhline(y=0, color='#6a8aa8', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    for bar, val in zip(bars, medias):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8, color='#c8d8e8')

    # Volatilidade anualizada
    ax = axes[0, 1]
    vols = log_retornos.std() * np.sqrt(252) * 100
    bars = ax.bar(ativos, vols, color=cores, alpha=0.8, edgecolor='none')
    ax.set_title('2º MOMENTO — VOLATILIDADE ANUAL (%)', fontsize=11,
                 fontweight='bold', color='#e8f4ff')
    ax.grid(True, alpha=0.3)
    for bar, val in zip(bars, vols):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8, color='#c8d8e8')

    # Assimetria
    ax = axes[1, 0]
    skews = log_retornos.skew()
    bar_colors = [COLORS['green'] if s > 0 else COLORS['red'] for s in skews]
    bars = ax.bar(ativos, skews, color=bar_colors, alpha=0.8, edgecolor='none')
    ax.set_title('3º MOMENTO — ASSIMETRIA (SKEWNESS)', fontsize=11,
                 fontweight='bold', color='#e8f4ff')
    ax.axhline(y=0, color='#6a8aa8', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # Curtose
    ax = axes[1, 1]
    kurts = log_retornos.kurtosis()
    bars = ax.bar(ativos, kurts, color=cores, alpha=0.8, edgecolor='none')
    ax.axhline(y=0, color=COLORS['amber'], linewidth=1, linestyle='--',
               label='Normal = 0 (excess)')
    ax.set_title('4º MOMENTO — CURTOSE EXCESSIVA', fontsize=11,
                 fontweight='bold', color='#e8f4ff')
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, 'm02_momentos_estatisticos.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Gráfico salvo: {path}")


def grafico_heatmap_correlacao(log_retornos):
    """Gráfico 2: Heatmap de correlação."""
    corr = log_retornos.corr()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Colormap personalizado
    cmap = LinearSegmentedColormap.from_list(
        'quant', ['#e05252', '#0d1520', '#00d4a0'], N=256)

    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    # Labels
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.columns)

    # Valores nas células
    for i in range(len(corr)):
        for j in range(len(corr)):
            color = '#0d1520' if abs(corr.iloc[i, j]) > 0.6 else '#c8d8e8'
            ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.yaxis.set_tick_params(color='#6a8aa8')

    ax.set_title('HEATMAP DE CORRELAÇÃO — RETORNOS DIÁRIOS',
                 fontsize=13, fontweight='bold', color='#e8f4ff', pad=20)

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, 'm02_heatmap_correlacao.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Gráfico salvo: {path}")

    # Salvar matriz
    corr.to_csv(os.path.join(DATA_DIR, 'matriz_correlacao.csv'))
    return corr


def grafico_gbm_simulacao():
    """Gráfico 3: Simulação de Movimento Browniano Geométrico."""
    print("\n── Simulação GBM (Geometric Brownian Motion) ────────────────")

    np.random.seed(42)

    S0 = 100     # Preço inicial
    mu = 0.15    # Drift anual 15%
    sigma = 0.30 # Volatilidade anual 30%
    T = 1        # 1 ano
    dt = 1/252   # Passos diários
    n_paths = 500
    n_steps = int(T / dt)

    # Simular caminhos
    Z = np.random.standard_normal((n_paths, n_steps))
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    paths = S0 * np.exp(np.cumsum(log_returns, axis=1))
    paths = np.column_stack([np.full(n_paths, S0), paths])

    t = np.linspace(0, T, n_steps + 1)

    # Estatísticas
    media_final = paths[:, -1].mean()
    mediana_final = np.median(paths[:, -1])
    esperado = S0 * np.exp(mu * T)

    print(f"  S0 = {S0} | μ = {mu:.0%} | σ = {sigma:.0%} | T = {T} ano")
    print(f"  E[S_T] teórico = {esperado:.2f}")
    print(f"  Média simulada  = {media_final:.2f}")
    print(f"  Mediana simulada = {mediana_final:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Caminhos
    ax = axes[0]
    for i in range(min(100, n_paths)):
        ax.plot(t, paths[i], alpha=0.15, linewidth=0.5, color=COLORS['cyan'])

    # Percentis
    p5 = np.percentile(paths, 5, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    ax.fill_between(t, p5, p95, alpha=0.2, color=COLORS['green'], label='Intervalo 90%')
    ax.plot(t, p50, color=COLORS['amber'], linewidth=2, label='Mediana')
    ax.plot(t, S0 * np.exp(mu * t), color=COLORS['red'], linewidth=2,
            linestyle='--', label=f'E[S_t] = S₀·e^(μt)')

    ax.set_title('SIMULAÇÃO GBM — 500 CAMINHOS', fontsize=11,
                 fontweight='bold', color='#e8f4ff')
    ax.set_xlabel('Tempo (anos)')
    ax.set_ylabel('Preço')
    ax.legend(fontsize=8, framealpha=0.3, edgecolor='#1e3048')
    ax.grid(True, alpha=0.3)

    # Distribuição final
    ax2 = axes[1]
    ax2.hist(paths[:, -1], bins=50, density=True, alpha=0.7,
             color=COLORS['blue'], edgecolor='none')

    # Fit log-normal teórica
    x_range = np.linspace(paths[:, -1].min(), paths[:, -1].max(), 200)
    mu_ln = np.log(S0) + (mu - 0.5 * sigma**2) * T
    sigma_ln = sigma * np.sqrt(T)
    ax2.plot(x_range, stats.lognorm.pdf(x_range, s=sigma_ln, scale=np.exp(mu_ln)),
             color=COLORS['amber'], linewidth=2, label='Log-Normal Teórica')

    ax2.axvline(x=esperado, color=COLORS['green'], linestyle='--',
                label=f'E[S_T] = {esperado:.0f}')
    ax2.axvline(x=mediana_final, color=COLORS['red'], linestyle=':',
                label=f'Mediana = {mediana_final:.0f}')

    ax2.set_title('DISTRIBUIÇÃO DO PREÇO FINAL', fontsize=11,
                  fontweight='bold', color='#e8f4ff')
    ax2.set_xlabel('Preço Final S_T')
    ax2.legend(fontsize=8, framealpha=0.3, edgecolor='#1e3048')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, 'm02_gbm_simulacao.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Gráfico salvo: {path}")


def grafico_risco_portfolio(log_retornos):
    """Gráfico 4: Risco de portfólio — diversificação."""
    print("\n── Análise de Risco de Portfólio ────────────────────────────")

    # Selecionar 5 ativos para o exemplo
    ativos_sel = log_retornos.columns[:5]
    ret = log_retornos[ativos_sel]

    # Matriz de covariância anualizada
    cov_matrix = ret.cov() * 252
    ret_anuais = ret.mean() * 252

    # Simulação Monte Carlo de portfólios
    np.random.seed(42)
    n_portfolios = 10000
    resultados = np.zeros((n_portfolios, 3))
    pesos_list = []

    rf = 0.1075  # SELIC

    for i in range(n_portfolios):
        w = np.random.dirichlet(np.ones(len(ativos_sel)))
        pesos_list.append(w)

        ret_port = w @ ret_anuais.values
        vol_port = np.sqrt(w @ cov_matrix.values @ w)
        sharpe = (ret_port - rf) / vol_port

        resultados[i] = [vol_port, ret_port, sharpe]

    # Melhor Sharpe
    idx_sharpe = resultados[:, 2].argmax()
    # Mínima variância
    idx_minvar = resultados[:, 0].argmin()

    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(resultados[:, 0] * 100, resultados[:, 1] * 100,
                         c=resultados[:, 2], cmap='viridis', s=3, alpha=0.5)

    # Max Sharpe
    ax.scatter(resultados[idx_sharpe, 0] * 100, resultados[idx_sharpe, 1] * 100,
               c=COLORS['red'], marker='*', s=300, zorder=5, edgecolors='white',
               linewidths=1, label=f'Max Sharpe (SR={resultados[idx_sharpe,2]:.2f})')

    # Min Var
    ax.scatter(resultados[idx_minvar, 0] * 100, resultados[idx_minvar, 1] * 100,
               c=COLORS['amber'], marker='D', s=150, zorder=5, edgecolors='white',
               linewidths=1, label='Mín. Variância')

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, label='Sharpe Ratio')
    cbar.ax.yaxis.set_tick_params(color='#6a8aa8')

    ax.set_title('SIMULAÇÃO MONTE CARLO — 10.000 PORTFÓLIOS',
                 fontsize=13, fontweight='bold', color='#e8f4ff', pad=15)
    ax.set_xlabel('Volatilidade Anual (%)')
    ax.set_ylabel('Retorno Anual (%)')
    ax.legend(fontsize=9, framealpha=0.3, edgecolor='#1e3048')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, 'm02_portfolios_monte_carlo.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Gráfico salvo: {path}")

    # Imprimir pesos ótimos
    print(f"\n  Portfólio Max Sharpe (SR={resultados[idx_sharpe,2]:.2f}):")
    for ativo, peso in zip(ativos_sel, pesos_list[idx_sharpe]):
        print(f"    {ativo}: {peso:.1%}")
    print(f"    Retorno: {resultados[idx_sharpe,1]:.1%} | Vol: {resultados[idx_sharpe,0]:.1%}")


def main():
    precos, log_retornos, nomes = baixar_dados()
    df_momentos = momentos_estatisticos(log_retornos)

    print("\n── Gerando Gráficos ────────────────────────────────────────")
    grafico_momentos(log_retornos)
    grafico_heatmap_correlacao(log_retornos)
    grafico_gbm_simulacao()
    grafico_risco_portfolio(log_retornos)

    # Salvar dados
    df_momentos.to_csv(os.path.join(DATA_DIR, 'momentos_estatisticos.csv'))
    log_retornos.to_csv(os.path.join(DATA_DIR, 'log_retornos_8ativos.csv'))

    print("\n" + "=" * 60)
    print("  MÓDULO 02 CONCLUÍDO ✓")
    print("  4 gráficos gerados em /graficos/")
    print("=" * 60)


if __name__ == '__main__':
    main()
