"""
=============================================================================
MÓDULO 09 — Simulação de Monte Carlo
Quant Academy · Finanças Quantitativas
=============================================================================
Simular milhares de futuros possíveis: GBM, precificação de opções,
distribuição de preços futuros e convergência do estimador.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
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
# 1. COLETA DE DADOS
# ══════════════════════════════════════════════════════════════════════════════
def carregar_dados():
    """Baixa dados da PETR4 via yfinance."""
    import yfinance as yf
    print("📥 Baixando dados da PETR4.SA...")
    df = yf.download('PETR4.SA', period='2y', auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Close']].dropna()
    df.columns = ['Close']

    retornos = np.log(df['Close'] / df['Close'].shift(1)).dropna()

    S0 = df['Close'].iloc[-1]
    mu = retornos.mean() * 252  # Drift anualizado
    sigma = retornos.std() * np.sqrt(252)  # Vol anualizada

    print(f"   ✔ Último preço: R$ {S0:.2f}")
    print(f"   ✔ Drift anual (μ): {mu:.2%}")
    print(f"   ✔ Volatilidade anual (σ): {sigma:.2%}")

    return df, retornos, S0, mu, sigma


# ══════════════════════════════════════════════════════════════════════════════
# 2. SIMULAÇÃO GBM (GEOMETRIC BROWNIAN MOTION)
# ══════════════════════════════════════════════════════════════════════════════
def simular_gbm(S0, mu, sigma, T=1.0, dt=1/252, n_paths=1000):
    """
    Simula caminhos de preço usando Movimento Browniano Geométrico.
    S_t = S_0 * exp[(μ - σ²/2)t + σW_t]
    """
    n_steps = int(T / dt)
    Z = np.random.standard_normal((n_paths, n_steps))

    # Incrementos do log-preço
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z

    # Preços: produto cumulativo
    log_paths = np.cumsum(log_returns, axis=1)
    paths = S0 * np.exp(log_paths)

    # Inserir S0 no início
    paths = np.column_stack([np.full(n_paths, S0), paths])

    return paths


# ══════════════════════════════════════════════════════════════════════════════
# 3. PRECIFICAÇÃO DE OPÇÕES — MONTE CARLO vs BLACK-SCHOLES
# ══════════════════════════════════════════════════════════════════════════════
def black_scholes_call(S, K, r, sigma, T):
    """Preço de Call Européia via Black-Scholes (fórmula analítica)."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)


def mc_call_price(S0, K, r, sigma, T, n_sim=100_000):
    """Precificação de Call Européia via Monte Carlo."""
    Z = np.random.standard_normal(n_sim)
    S_T = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(S_T - K, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sim)
    return price, std_error


def convergencia_mc(S0, K, r, sigma, T, bs_price, n_max=200_000, step=1000):
    """Mostra convergência do preço MC conforme N aumenta."""
    ns = range(step, n_max + 1, step)
    prices = []
    errors = []

    np.random.seed(42)
    Z_all = np.random.standard_normal(n_max)

    for n in ns:
        Z = Z_all[:n]
        S_T = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
        payoffs = np.maximum(S_T - K, 0)
        price = np.exp(-r * T) * np.mean(payoffs)
        se = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n)
        prices.append(price)
        errors.append(se)

    return list(ns), prices, errors


# ══════════════════════════════════════════════════════════════════════════════
# 4. GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════
def grafico_simulacao_gbm(paths, S0, T=1.0):
    """Gráfico principal: caminhos GBM coloridos pelo preço final."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8),
                              gridspec_kw={'width_ratios': [3, 1]})
    fig.patch.set_facecolor(COLORS['bg'])

    n_paths, n_steps = paths.shape
    t = np.linspace(0, T, n_steps)
    final_prices = paths[:, -1]

    # ── Painel Esquerdo: Caminhos ──
    ax = axes[0]
    ax.set_facecolor(COLORS['surface'])

    # Colorir por preço final
    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(final_prices.min(), final_prices.max())

    # Plotar amostra de caminhos (para não sobrecarregar)
    n_plot = min(300, n_paths)
    indices = np.random.choice(n_paths, n_plot, replace=False)

    for idx in indices:
        color = cmap(norm(final_prices[idx]))
        ax.plot(t * 252, paths[idx], color=color, alpha=0.15, linewidth=0.5)

    # Percentis
    p5 = np.percentile(paths, 5, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    ax.plot(t * 252, p50, color=COLORS['green'], linewidth=2, label='Mediana')
    ax.fill_between(t * 252, p5, p95, alpha=0.15, color=COLORS['green'],
                    label='Intervalo 90%')
    ax.axhline(S0, color=COLORS['dim'], linestyle='--', alpha=0.5,
               label=f'S₀ = R$ {S0:.2f}')

    ax.set_title(f'MONTE CARLO · {n_paths} CAMINHOS GBM',
                 fontsize=14, color=COLORS['text'], fontweight='bold', pad=15)
    ax.set_xlabel('Dias Úteis', color=COLORS['dim'])
    ax.set_ylabel('Preço (R$)', color=COLORS['dim'])
    ax.legend(fontsize=9, loc='upper left')
    ax.tick_params(colors=COLORS['dim'])

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Preço Final (R$)', color=COLORS['dim'])
    cbar.ax.tick_params(colors=COLORS['dim'])

    # ── Painel Direito: Distribuição dos preços finais ──
    ax2 = axes[1]
    ax2.set_facecolor(COLORS['surface'])
    ax2.hist(final_prices, bins=60, orientation='horizontal',
             color=COLORS['blue'], alpha=0.7, edgecolor='none', density=True)
    ax2.axhline(S0, color=COLORS['dim'], linestyle='--', alpha=0.5)
    ax2.axhline(np.median(final_prices), color=COLORS['green'], linewidth=2)

    # Estatísticas
    prob_lucro = (final_prices > S0).mean()
    info = (f'Média: R$ {final_prices.mean():.2f}\n'
            f'Mediana: R$ {np.median(final_prices):.2f}\n'
            f'P(lucro): {prob_lucro:.1%}\n'
            f'Min: R$ {final_prices.min():.2f}\n'
            f'Max: R$ {final_prices.max():.2f}')
    ax2.text(0.95, 0.95, info, transform=ax2.transAxes,
             fontsize=9, color=COLORS['text'], va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor=COLORS['bg'], alpha=0.8))

    ax2.set_title('DISTRIBUIÇÃO\nPREÇO FINAL', fontsize=11,
                  color=COLORS['dim'], pad=10)
    ax2.set_xlabel('Densidade', color=COLORS['dim'])
    ax2.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m09_simulacao_gbm.png', **SAVE)
    plt.close()
    print("   ✔ Gráfico salvo: graficos/m09_simulacao_gbm.png")


def grafico_precificacao_opcao(S0, K, r, sigma, T, bs_price, mc_price, mc_se):
    """Gráfico de convergência MC e comparação com Black-Scholes."""
    ns, prices, errors = convergencia_mc(S0, K, r, sigma, T, bs_price)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(COLORS['bg'])

    # ── Convergência ──
    ax = axes[0]
    ax.set_facecolor(COLORS['surface'])

    prices = np.array(prices)
    errors = np.array(errors)

    ax.plot(ns, prices, color=COLORS['blue'], linewidth=1, alpha=0.8)
    ax.fill_between(ns, prices - 1.96 * errors, prices + 1.96 * errors,
                    alpha=0.2, color=COLORS['blue'], label='IC 95%')
    ax.axhline(bs_price, color=COLORS['green'], linewidth=2, linestyle='--',
               label=f'Black-Scholes: R$ {bs_price:.4f}')

    ax.set_title('CONVERGÊNCIA DO MONTE CARLO',
                 fontsize=13, color=COLORS['text'], fontweight='bold', pad=15)
    ax.set_xlabel('Número de Simulações', color=COLORS['dim'])
    ax.set_ylabel('Preço da Call (R$)', color=COLORS['dim'])
    ax.legend(fontsize=9)
    ax.tick_params(colors=COLORS['dim'])

    # ── Histograma de payoffs ──
    ax2 = axes[1]
    ax2.set_facecolor(COLORS['surface'])

    np.random.seed(123)
    Z = np.random.standard_normal(100_000)
    S_T = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(S_T - K, 0)

    # Separar ITM e OTM
    itm = S_T[S_T > K]
    otm = S_T[S_T <= K]

    ax2.hist(S_T, bins=100, color=COLORS['blue'], alpha=0.4, edgecolor='none',
             density=True, label='S_T total')
    ax2.axvline(K, color=COLORS['amber'], linewidth=2, linestyle='--',
                label=f'Strike K = R$ {K:.2f}')
    ax2.axvline(S0, color=COLORS['green'], linewidth=2, linestyle='-',
                label=f'Spot S₀ = R$ {S0:.2f}')

    prob_itm = (S_T > K).mean()
    info = (f'P(ITM): {prob_itm:.1%}\n'
            f'E[payoff]: R$ {payoffs.mean():.4f}\n'
            f'Preço MC: R$ {mc_price:.4f}\n'
            f'BS Price: R$ {bs_price:.4f}')
    ax2.text(0.95, 0.95, info, transform=ax2.transAxes,
             fontsize=9, color=COLORS['text'], va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor=COLORS['bg'], alpha=0.8))

    ax2.set_title('DISTRIBUIÇÃO DE S_T · PAYOFF DA CALL',
                  fontsize=13, color=COLORS['text'], fontweight='bold', pad=15)
    ax2.set_xlabel('Preço Final S_T (R$)', color=COLORS['dim'])
    ax2.set_ylabel('Densidade', color=COLORS['dim'])
    ax2.legend(fontsize=9)
    ax2.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m09_precificacao_opcao_mc.png', **SAVE)
    plt.close()
    print("   ✔ Gráfico salvo: graficos/m09_precificacao_opcao_mc.png")


def grafico_distribuicao_final(paths, S0):
    """Distribuição detalhada dos preços finais simulados."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(COLORS['bg'])

    final_prices = paths[:, -1]
    retornos_sim = np.log(final_prices / S0)

    # ── Distribuição de preços finais ──
    ax = axes[0]
    ax.set_facecolor(COLORS['surface'])
    ax.hist(final_prices, bins=80, color=COLORS['green'], alpha=0.7,
            edgecolor='none', density=True)
    ax.axvline(S0, color=COLORS['amber'], linewidth=2, linestyle='--',
               label=f'S₀ = R$ {S0:.2f}')
    ax.axvline(np.mean(final_prices), color=COLORS['red'], linewidth=2,
               label=f'E[S_T] = R$ {np.mean(final_prices):.2f}')
    ax.set_title('DISTRIBUIÇÃO DO PREÇO FINAL', fontsize=12,
                 color=COLORS['text'], fontweight='bold')
    ax.set_xlabel('Preço (R$)', color=COLORS['dim'])
    ax.legend(fontsize=8)
    ax.tick_params(colors=COLORS['dim'])

    # ── Distribuição de retornos simulados ──
    ax2 = axes[1]
    ax2.set_facecolor(COLORS['surface'])
    ax2.hist(retornos_sim * 100, bins=80, color=COLORS['blue'], alpha=0.7,
             edgecolor='none', density=True)
    ax2.axvline(0, color=COLORS['dim'], linestyle='--')
    ax2.set_title('DISTRIBUIÇÃO DOS RETORNOS', fontsize=12,
                  color=COLORS['text'], fontweight='bold')
    ax2.set_xlabel('Retorno (%)', color=COLORS['dim'])
    ax2.tick_params(colors=COLORS['dim'])

    # ── QQ-Plot ──
    ax3 = axes[2]
    ax3.set_facecolor(COLORS['surface'])
    stats.probplot(retornos_sim, dist='norm', plot=ax3)
    ax3.get_lines()[0].set_markerfacecolor(COLORS['blue'])
    ax3.get_lines()[0].set_markeredgecolor(COLORS['blue'])
    ax3.get_lines()[0].set_markersize(3)
    ax3.get_lines()[1].set_color(COLORS['green'])
    ax3.set_title('QQ-PLOT · RETORNOS SIMULADOS vs NORMAL',
                  fontsize=12, color=COLORS['text'], fontweight='bold')
    ax3.set_xlabel('Quantis Teóricos', color=COLORS['dim'])
    ax3.set_ylabel('Quantis Observados', color=COLORS['dim'])
    ax3.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m09_distribuicao_final.png', **SAVE)
    plt.close()
    print("   ✔ Gráfico salvo: graficos/m09_distribuicao_final.png")


def grafico_cenarios(paths, S0, T=1.0):
    """Fan chart com percentis ao longo do tempo."""
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['surface'])

    n_steps = paths.shape[1]
    t = np.arange(n_steps)

    percentis = [5, 10, 25, 50, 75, 90, 95]
    cores_fan = ['#e05252', '#f5a623', '#4a9eff', '#00d4a0',
                 '#4a9eff', '#f5a623', '#e05252']
    alphas = [0.1, 0.15, 0.2, 1.0, 0.2, 0.15, 0.1]

    # Fan chart
    p_vals = {p: np.percentile(paths, p, axis=0) for p in percentis}

    ax.fill_between(t, p_vals[5], p_vals[95], alpha=0.1, color=COLORS['blue'],
                    label='P5-P95')
    ax.fill_between(t, p_vals[10], p_vals[90], alpha=0.15, color=COLORS['blue'],
                    label='P10-P90')
    ax.fill_between(t, p_vals[25], p_vals[75], alpha=0.25, color=COLORS['blue'],
                    label='P25-P75')
    ax.plot(t, p_vals[50], color=COLORS['green'], linewidth=2, label='Mediana (P50)')

    ax.axhline(S0, color=COLORS['dim'], linestyle='--', alpha=0.5)

    ax.set_title('FAN CHART · PROJEÇÃO DE CENÁRIOS',
                 fontsize=14, color=COLORS['text'], fontweight='bold', pad=15)
    ax.set_xlabel('Dias Úteis', color=COLORS['dim'])
    ax.set_ylabel('Preço (R$)', color=COLORS['dim'])
    ax.legend(fontsize=9, loc='upper left')
    ax.tick_params(colors=COLORS['dim'])

    # Anotações
    for p in [5, 25, 50, 75, 95]:
        ax.annotate(f'P{p}: R$ {p_vals[p][-1]:.2f}',
                    xy=(n_steps - 1, p_vals[p][-1]),
                    xytext=(10, 0), textcoords='offset points',
                    fontsize=8, color=COLORS['text'],
                    va='center')

    plt.tight_layout()
    plt.savefig('../graficos/m09_fan_chart_cenarios.png', **SAVE)
    plt.close()
    print("   ✔ Gráfico salvo: graficos/m09_fan_chart_cenarios.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  MÓDULO 09 · SIMULAÇÃO DE MONTE CARLO")
    print("=" * 70)

    df, retornos, S0, mu, sigma = carregar_dados()

    # ── 1. Simulação GBM ──────────────────────────────────────────────────
    print("\n📊 Simulando 1.000 caminhos GBM (252 dias)...")
    np.random.seed(42)
    paths = simular_gbm(S0, mu, sigma, T=1.0, dt=1/252, n_paths=1000)

    final_prices = paths[:, -1]
    print(f"\n  Preço Final:")
    print(f"    Média:   R$ {final_prices.mean():.2f}")
    print(f"    Mediana: R$ {np.median(final_prices):.2f}")
    print(f"    Min:     R$ {final_prices.min():.2f}")
    print(f"    Max:     R$ {final_prices.max():.2f}")
    print(f"    P(S_T > S_0): {(final_prices > S0).mean():.1%}")

    # VaR via Monte Carlo
    retornos_sim = np.log(final_prices / S0)
    var_95 = -np.percentile(retornos_sim, 5)
    print(f"\n    VaR 95% (1 ano): {var_95:.2%}")
    print(f"    VaR 95% (R$): R$ {S0 * var_95:,.2f}")

    # ── 2. Precificação de Opção ──────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  PRECIFICAÇÃO DE CALL EUROPÉIA")
    print(f"{'─' * 70}")

    r = 0.1075  # Taxa Selic aproximada
    K = S0  # ATM
    T = 0.25  # 3 meses

    bs_price = black_scholes_call(S0, K, r, sigma, T)
    mc_price, mc_se = mc_call_price(S0, K, r, sigma, T, n_sim=500_000)

    print(f"\n  Parâmetros:")
    print(f"    S₀ (Spot):     R$ {S0:.2f}")
    print(f"    K (Strike):    R$ {K:.2f}")
    print(f"    r (Selic):     {r:.2%}")
    print(f"    σ (Vol):       {sigma:.2%}")
    print(f"    T (Prazo):     {T:.2f} anos ({T*252:.0f} dias)")

    print(f"\n  Resultados:")
    print(f"    Black-Scholes: R$ {bs_price:.4f}")
    print(f"    Monte Carlo:   R$ {mc_price:.4f} ± {mc_se:.4f}")
    print(f"    Diferença:     R$ {abs(bs_price - mc_price):.4f} ({abs(bs_price - mc_price)/bs_price:.2%})")

    # ── 3. Gráficos ──────────────────────────────────────────────────────
    print("\n🎨 Gerando gráficos...")
    grafico_simulacao_gbm(paths, S0)
    grafico_precificacao_opcao(S0, K, r, sigma, T, bs_price, mc_price, mc_se)
    grafico_distribuicao_final(paths, S0)
    grafico_cenarios(paths, S0)

    print(f"\n{'=' * 70}")
    print("  ✅ Módulo 09 concluído!")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
