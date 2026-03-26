"""
=============================================================================
MÓDULO 12 — Derivativos & Precificação de Opções
=============================================================================
Quant Academy · Finanças Quantitativas com Python

Conteúdo:
  - Black-Scholes Model (Call & Put Européias)
  - Greeks: Delta, Gamma, Theta, Vega, Rho
  - Superfície de volatilidade implícita
  - Estratégias com opções: Straddle, Butterfly, Iron Condor
  - Paridade Put-Call
  - Sensibilidade dos Greeks

Gráficos interativos e análise completa
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm
from scipy.optimize import brentq
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


# ══════════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES ENGINE
# ══════════════════════════════════════════════════════════════════════════

def black_scholes(S, K, r, sigma, T, option='call'):
    """Calcula preço e Greeks de opção européia via Black-Scholes."""
    if T <= 0:
        if option == 'call':
            return {'price': max(S - K, 0), 'delta': 1.0 if S > K else 0.0,
                    'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        else:
            return {'price': max(K - S, 0), 'delta': -1.0 if S < K else 0.0,
                    'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    return {
        'price': price, 'delta': delta, 'gamma': gamma,
        'theta': theta, 'vega': vega, 'rho': rho,
        'd1': d1, 'd2': d2
    }


def implied_volatility(market_price, S, K, r, T, option='call'):
    """Calcula volatilidade implícita via Brent's method."""
    def objective(sigma):
        return black_scholes(S, K, r, sigma, T, option)['price'] - market_price

    try:
        return brentq(objective, 0.001, 5.0)
    except ValueError:
        return np.nan


def demonstrar_black_scholes():
    """Demonstração do modelo Black-Scholes."""
    print("=" * 60)
    print("  MÓDULO 12 — DERIVATIVOS & PRECIFICAÇÃO DE OPÇÕES")
    print("=" * 60)

    S = 100     # Preço do ativo
    K = 100     # Strike (ATM)
    r = 0.1075  # Taxa SELIC
    sigma = 0.30  # Volatilidade 30%
    T = 0.25    # 3 meses

    call = black_scholes(S, K, r, sigma, T, 'call')
    put = black_scholes(S, K, r, sigma, T, 'put')

    print(f"\n  Parâmetros: S={S}, K={K}, r={r:.2%}, σ={sigma:.0%}, T={T} anos")
    print(f"\n  {'Métrica':<15} {'Call':>10} {'Put':>10}")
    print("  " + "-" * 37)
    for key in ['price', 'delta', 'gamma', 'theta', 'vega', 'rho']:
        print(f"  {key.capitalize():<15} {call[key]:>10.4f} {put[key]:>10.4f}")

    # Verificar Paridade Put-Call
    parity_left = call['price'] - put['price']
    parity_right = S - K * np.exp(-r * T)
    print(f"\n  Paridade Put-Call:")
    print(f"    C - P = {parity_left:.4f}")
    print(f"    S - K·e^(-rT) = {parity_right:.4f}")
    print(f"    Diferença: {abs(parity_left - parity_right):.10f}")


def grafico_greeks(S=100, K=100, r=0.1075, sigma=0.30, T=0.25):
    """Gráfico 1: Greeks em função do preço do ativo."""
    print("\n── Gerando Gráficos ────────────────────────────────────────")

    S_range = np.linspace(60, 140, 200)

    greeks = {'delta': [], 'gamma': [], 'theta': [], 'vega': []}
    greeks_put = {'delta': [], 'gamma': [], 'theta': [], 'vega': []}

    for s in S_range:
        call = black_scholes(s, K, r, sigma, T, 'call')
        put = black_scholes(s, K, r, sigma, T, 'put')
        for g in greeks:
            greeks[g].append(call[g])
            greeks_put[g].append(put[g])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Delta
    ax = axes[0, 0]
    ax.plot(S_range, greeks['delta'], color=COLORS['green'], linewidth=2, label='Call Delta')
    ax.plot(S_range, greeks_put['delta'], color=COLORS['red'], linewidth=2, label='Put Delta')
    ax.axhline(y=0, color='#6a8aa8', linewidth=0.5)
    ax.axvline(x=K, color=COLORS['amber'], linewidth=1, linestyle='--', alpha=0.5, label=f'Strike={K}')
    ax.set_title('DELTA (Δ) — Sensibilidade ao Preço',
                 fontsize=11, fontweight='bold', color='#e8f4ff')
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(True, alpha=0.3)

    # Gamma
    ax = axes[0, 1]
    ax.plot(S_range, greeks['gamma'], color=COLORS['cyan'], linewidth=2, label='Gamma')
    ax.axvline(x=K, color=COLORS['amber'], linewidth=1, linestyle='--', alpha=0.5)
    ax.set_title('GAMMA (Γ) — Curvatura do Delta',
                 fontsize=11, fontweight='bold', color='#e8f4ff')
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(True, alpha=0.3)

    # Theta
    ax = axes[1, 0]
    ax.plot(S_range, greeks['theta'], color=COLORS['amber'], linewidth=2, label='Call Theta')
    ax.plot(S_range, greeks_put['theta'], color=COLORS['purple'], linewidth=2, label='Put Theta')
    ax.axvline(x=K, color=COLORS['amber'], linewidth=1, linestyle='--', alpha=0.5)
    ax.set_title('THETA (Θ) — Decaimento Temporal (/dia)',
                 fontsize=11, fontweight='bold', color='#e8f4ff')
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(True, alpha=0.3)

    # Vega
    ax = axes[1, 1]
    ax.plot(S_range, greeks['vega'], color=COLORS['purple'], linewidth=2, label='Vega')
    ax.axvline(x=K, color=COLORS['amber'], linewidth=1, linestyle='--', alpha=0.5)
    ax.set_title('VEGA (ν) — Sensibilidade à Volatilidade',
                 fontsize=11, fontweight='bold', color='#e8f4ff')
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel('Preço do Ativo (S)')

    plt.suptitle(f'GREEKS — Call & Put (K={K}, σ={sigma:.0%}, T={T*12:.0f}m)',
                 fontsize=14, fontweight='bold', color='#e8f4ff', y=1.01)
    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, 'm12_greeks.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Gráfico salvo: {path}")


def grafico_estrategias_opcoes():
    """Gráfico 2: Payoff de estratégias com opções."""
    S_range = np.linspace(60, 140, 500)
    K = 100
    r = 0.1075
    sigma = 0.30
    T = 0.25

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Long Straddle ──
    ax = axes[0, 0]
    call_price = black_scholes(K, K, r, sigma, T, 'call')['price']
    put_price = black_scholes(K, K, r, sigma, T, 'put')['price']
    custo = call_price + put_price

    payoff_call = np.maximum(S_range - K, 0) - call_price
    payoff_put = np.maximum(K - S_range, 0) - put_price
    payoff_total = payoff_call + payoff_put

    ax.plot(S_range, payoff_call, color=COLORS['green'], linewidth=1, alpha=0.5, label='Long Call')
    ax.plot(S_range, payoff_put, color=COLORS['red'], linewidth=1, alpha=0.5, label='Long Put')
    ax.plot(S_range, payoff_total, color=COLORS['cyan'], linewidth=2.5, label='Straddle')
    ax.fill_between(S_range, payoff_total, 0, where=(payoff_total > 0),
                    alpha=0.15, color=COLORS['green'])
    ax.fill_between(S_range, payoff_total, 0, where=(payoff_total < 0),
                    alpha=0.15, color=COLORS['red'])
    ax.axhline(y=0, color='#6a8aa8', linewidth=0.5)
    ax.set_title(f'LONG STRADDLE (custo: ${custo:.2f})',
                 fontsize=11, fontweight='bold', color='#e8f4ff')
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(True, alpha=0.3)

    # ── Bull Call Spread ──
    ax = axes[0, 1]
    K1, K2 = 95, 105
    c1 = black_scholes(K, K1, r, sigma, T, 'call')['price']
    c2 = black_scholes(K, K2, r, sigma, T, 'call')['price']
    custo = c1 - c2

    payoff = np.maximum(S_range - K1, 0) - np.maximum(S_range - K2, 0) - custo
    ax.plot(S_range, payoff, color=COLORS['green'], linewidth=2.5)
    ax.fill_between(S_range, payoff, 0, where=(payoff > 0), alpha=0.15, color=COLORS['green'])
    ax.fill_between(S_range, payoff, 0, where=(payoff < 0), alpha=0.15, color=COLORS['red'])
    ax.axhline(y=0, color='#6a8aa8', linewidth=0.5)
    ax.set_title(f'BULL CALL SPREAD ({K1}/{K2}, custo: ${custo:.2f})',
                 fontsize=11, fontweight='bold', color='#e8f4ff')
    ax.grid(True, alpha=0.3)

    # ── Butterfly Spread ──
    ax = axes[1, 0]
    K1, K2, K3 = 90, 100, 110
    c1 = black_scholes(K, K1, r, sigma, T, 'call')['price']
    c2 = black_scholes(K, K2, r, sigma, T, 'call')['price']
    c3 = black_scholes(K, K3, r, sigma, T, 'call')['price']
    custo = c1 - 2*c2 + c3

    payoff = (np.maximum(S_range - K1, 0) - 2*np.maximum(S_range - K2, 0)
              + np.maximum(S_range - K3, 0) - custo)
    ax.plot(S_range, payoff, color=COLORS['purple'], linewidth=2.5)
    ax.fill_between(S_range, payoff, 0, where=(payoff > 0), alpha=0.15, color=COLORS['green'])
    ax.fill_between(S_range, payoff, 0, where=(payoff < 0), alpha=0.15, color=COLORS['red'])
    ax.axhline(y=0, color='#6a8aa8', linewidth=0.5)
    ax.set_title(f'BUTTERFLY SPREAD ({K1}/{K2}/{K3})',
                 fontsize=11, fontweight='bold', color='#e8f4ff')
    ax.grid(True, alpha=0.3)

    # ── Iron Condor ──
    ax = axes[1, 1]
    K1, K2, K3, K4 = 85, 95, 105, 115
    p1 = black_scholes(K, K1, r, sigma, T, 'put')['price']
    p2 = black_scholes(K, K2, r, sigma, T, 'put')['price']
    c3 = black_scholes(K, K3, r, sigma, T, 'call')['price']
    c4 = black_scholes(K, K4, r, sigma, T, 'call')['price']
    credito = p2 - p1 + c3 - c4

    payoff = (-np.maximum(K1 - S_range, 0) + np.maximum(K2 - S_range, 0)
              + np.maximum(S_range - K3, 0) * (-1) + np.maximum(S_range - K4, 0) * (1)
              + credito)

    # Iron Condor recalculado corretamente
    payoff = (
        -np.maximum(K2 - S_range, 0) + np.maximum(K1 - S_range, 0)  # Bull Put Spread
        - np.maximum(S_range - K3, 0) + np.maximum(S_range - K4, 0)  # Bear Call Spread
        + credito
    )

    ax.plot(S_range, payoff, color=COLORS['amber'], linewidth=2.5)
    ax.fill_between(S_range, payoff, 0, where=(payoff > 0), alpha=0.15, color=COLORS['green'])
    ax.fill_between(S_range, payoff, 0, where=(payoff < 0), alpha=0.15, color=COLORS['red'])
    ax.axhline(y=0, color='#6a8aa8', linewidth=0.5)
    ax.set_title(f'IRON CONDOR ({K1}/{K2}/{K3}/{K4})',
                 fontsize=11, fontweight='bold', color='#e8f4ff')
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel('Preço do Ativo na Expiração (S_T)')
        ax.set_ylabel('Lucro/Prejuízo ($)')

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, 'm12_estrategias_opcoes.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Gráfico salvo: {path}")


def grafico_superficie_volatilidade():
    """Gráfico 3: Superfície de volatilidade implícita (smile)."""
    from mpl_toolkits.mplot3d import Axes3D

    S = 100
    r = 0.1075

    # Criar grid de strikes e maturidades
    strikes = np.linspace(70, 130, 30)
    maturidades = np.linspace(0.05, 2.0, 25)

    K_grid, T_grid = np.meshgrid(strikes, maturidades)

    # Simular volatility smile/skew
    # σ_impl = σ_ATM + skew * moneyness + smile * moneyness²
    sigma_atm = 0.25
    skew = -0.15  # Typical equity skew
    smile = 0.10  # Smile curvature

    moneyness = np.log(K_grid / S)
    vol_surface = sigma_atm + skew * moneyness + smile * moneyness**2

    # Adicionar efeito do tempo (term structure)
    vol_surface += 0.03 * np.exp(-T_grid)  # Vol maior para curto prazo

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')

    cmap = LinearSegmentedColormap.from_list('vol', ['#4a9eff', '#00d4a0', '#f5a623', '#e05252'])

    surf = ax.plot_surface(K_grid, T_grid, vol_surface * 100,
                           cmap=cmap, alpha=0.85, edgecolor='none',
                           antialiased=True)

    ax.set_xlabel('Strike (K)', fontsize=10, color='#c8d8e8', labelpad=10)
    ax.set_ylabel('Maturidade (T anos)', fontsize=10, color='#c8d8e8', labelpad=10)
    ax.set_zlabel('Vol. Implícita (%)', fontsize=10, color='#c8d8e8', labelpad=10)
    ax.set_title('SUPERFÍCIE DE VOLATILIDADE IMPLÍCITA',
                 fontsize=14, fontweight='bold', color='#e8f4ff', pad=20)

    ax.set_facecolor('#0d1520')
    fig.colorbar(surf, ax=ax, shrink=0.6, label='Vol (%)')

    ax.view_init(elev=25, azim=135)

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, 'm12_superficie_volatilidade.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Gráfico salvo: {path}")


def grafico_theta_decay():
    """Gráfico 4: Decaimento temporal (Theta) ao longo do tempo."""
    S = 100
    K = 100
    r = 0.1075
    sigma = 0.30

    T_range = np.linspace(1.0, 0.001, 200)  # 1 ano até expiração

    # Diferentes moneyness
    strikes = {'ITM (K=90)': 90, 'ATM (K=100)': 100, 'OTM (K=110)': 110}
    cores_strikes = [COLORS['green'], COLORS['amber'], COLORS['red']]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Preço ao longo do tempo
    ax = axes[0, 0]
    for (label, k), cor in zip(strikes.items(), cores_strikes):
        prices = [black_scholes(S, k, r, sigma, t, 'call')['price'] for t in T_range]
        ax.plot((1 - T_range) * 365, prices, color=cor, linewidth=2, label=label)

    ax.set_title('PREÇO DA CALL AO LONGO DO TEMPO',
                 fontsize=11, fontweight='bold', color='#e8f4ff')
    ax.set_xlabel('Dias até Expiração')
    ax.set_ylabel('Preço ($)')
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # Theta ao longo do tempo
    ax = axes[0, 1]
    for (label, k), cor in zip(strikes.items(), cores_strikes):
        thetas = [black_scholes(S, k, r, sigma, t, 'call')['theta'] for t in T_range]
        ax.plot((1 - T_range) * 365, thetas, color=cor, linewidth=2, label=label)

    ax.set_title('THETA AO LONGO DO TEMPO',
                 fontsize=11, fontweight='bold', color='#e8f4ff')
    ax.set_xlabel('Dias até Expiração')
    ax.set_ylabel('Theta ($/dia)')
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # Delta ao longo do tempo
    ax = axes[1, 0]
    for (label, k), cor in zip(strikes.items(), cores_strikes):
        deltas = [black_scholes(S, k, r, sigma, t, 'call')['delta'] for t in T_range]
        ax.plot((1 - T_range) * 365, deltas, color=cor, linewidth=2, label=label)

    ax.set_title('DELTA AO LONGO DO TEMPO',
                 fontsize=11, fontweight='bold', color='#e8f4ff')
    ax.set_xlabel('Dias até Expiração')
    ax.set_ylabel('Delta')
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # Volatility Smile para diferentes maturidades
    ax = axes[1, 1]
    K_range = np.linspace(70, 130, 100)
    maturidades = {'T=1m': 1/12, 'T=3m': 3/12, 'T=6m': 6/12, 'T=1a': 1.0}
    cores_t = [COLORS['red'], COLORS['amber'], COLORS['green'], COLORS['blue']]

    for (label, t), cor in zip(maturidades.items(), cores_t):
        # Simulated smile
        moneyness = np.log(K_range / S)
        sigma_impl = 0.25 - 0.15 * moneyness + 0.10 * moneyness**2 + 0.03 * np.exp(-t)
        ax.plot(K_range, sigma_impl * 100, color=cor, linewidth=2, label=label)

    ax.axvline(x=S, color='#6a8aa8', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_title('VOLATILITY SMILE POR MATURIDADE',
                 fontsize=11, fontweight='bold', color='#e8f4ff')
    ax.set_xlabel('Strike (K)')
    ax.set_ylabel('Vol. Implícita (%)')
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, 'm12_theta_decay_analise.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Gráfico salvo: {path}")


def main():
    demonstrar_black_scholes()
    grafico_greeks()
    grafico_estrategias_opcoes()
    grafico_superficie_volatilidade()
    grafico_theta_decay()

    print("\n" + "=" * 60)
    print("  MÓDULO 12 CONCLUÍDO ✓")
    print("  4 gráficos gerados em /graficos/")
    print("  🎓 CURSO QUANT ACADEMY COMPLETO!")
    print("=" * 60)


if __name__ == '__main__':
    main()
