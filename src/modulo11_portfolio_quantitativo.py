"""
=============================================================================
MÓDULO 11 — Gestão Quantitativa de Portfólio
=============================================================================
Quant Academy · Finanças Quantitativas com Python

Conteúdo:
  - Markowitz Mean-Variance Optimization
  - Fronteira Eficiente
  - Sharpe, Sortino, Calmar, Information Ratio
  - Monte Carlo Portfolio Simulation
  - Black-Litterman Model
  - Otimização com restrições reais

Dados reais: 10 ações brasileiras via yfinance
=============================================================================
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
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
    """Baixa dados de 10 ações brasileiras diversificadas."""
    print("=" * 60)
    print("  MÓDULO 11 — GESTÃO QUANTITATIVA DE PORTFÓLIO")
    print("=" * 60)

    tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA',
               'WEGE3.SA', 'RENT3.SA', 'BBAS3.SA', 'SUZB3.SA', 'GGBR4.SA']
    nomes = ['PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'ABEV3',
             'WEGE3', 'RENT3', 'BBAS3', 'SUZB3', 'GGBR4']

    dados = yf.download(tickers, start='2020-01-01', end='2025-12-31',
                        auto_adjust=True, progress=False)
    precos = dados['Close']
    precos.columns = nomes
    precos = precos.dropna()

    log_retornos = np.log(precos / precos.shift(1)).dropna()

    print(f"\n✓ {len(precos)} dias | {len(nomes)} ativos")
    return precos, log_retornos, nomes


def calcular_metricas_portfolio(w, ret_anuais, cov_anual, rf=0.1075):
    """Calcula métricas do portfólio dado pesos w."""
    ret_port = w @ ret_anuais
    vol_port = np.sqrt(w @ cov_anual @ w)
    sharpe = (ret_port - rf) / vol_port
    return ret_port, vol_port, sharpe


def otimizar_portfolio(log_retornos, nomes):
    """Encontra portfólios ótimos via otimização."""
    print("\n── Otimização de Portfólio (Markowitz) ──────────────────────")

    ret_anuais = log_retornos.mean() * 252
    cov_anual = log_retornos.cov() * 252
    n = len(nomes)
    rf = 0.1075

    # Restrições
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 0.30) for _ in range(n)]  # Max 30% por ativo

    # Max Sharpe
    def neg_sharpe(w):
        r, v, s = calcular_metricas_portfolio(w, ret_anuais.values, cov_anual.values, rf)
        return -s

    w0 = np.ones(n) / n
    res_sharpe = minimize(neg_sharpe, w0, method='SLSQP',
                          bounds=bounds, constraints=constraints)
    w_sharpe = res_sharpe.x

    # Mínima Variância
    def portfolio_vol(w):
        return np.sqrt(w @ cov_anual.values @ w)

    res_minvar = minimize(portfolio_vol, w0, method='SLSQP',
                          bounds=bounds, constraints=constraints)
    w_minvar = res_minvar.x

    # Resultados
    r_s, v_s, s_s = calcular_metricas_portfolio(w_sharpe, ret_anuais.values, cov_anual.values, rf)
    r_m, v_m, s_m = calcular_metricas_portfolio(w_minvar, ret_anuais.values, cov_anual.values, rf)

    print(f"\n  Max Sharpe:  Ret={r_s:.1%} | Vol={v_s:.1%} | Sharpe={s_s:.2f}")
    print(f"  Mín. Var:    Ret={r_m:.1%} | Vol={v_m:.1%} | Sharpe={s_m:.2f}")

    print(f"\n  Pesos Max Sharpe:")
    for nome, peso in sorted(zip(nomes, w_sharpe), key=lambda x: -x[1]):
        if peso > 0.01:
            print(f"    {nome}: {peso:.1%}")

    return ret_anuais, cov_anual, w_sharpe, w_minvar


def grafico_fronteira_eficiente(log_retornos, nomes, ret_anuais, cov_anual,
                                 w_sharpe, w_minvar):
    """Gráfico 1: Fronteira eficiente com Monte Carlo."""
    n = len(nomes)
    rf = 0.1075

    # Monte Carlo
    np.random.seed(42)
    n_port = 15000
    resultados = np.zeros((n_port, 3))

    for i in range(n_port):
        w = np.random.dirichlet(np.ones(n))
        r, v, s = calcular_metricas_portfolio(w, ret_anuais.values, cov_anual.values, rf)
        resultados[i] = [v, r, s]

    # Fronteira eficiente analítica
    target_returns = np.linspace(resultados[:, 1].min(), resultados[:, 1].max(), 100)
    ef_vols = []
    ef_rets = []

    for target in target_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w, t=target: w @ ret_anuais.values - t}
        ]
        bounds = [(0, 0.30) for _ in range(n)]

        def vol_obj(w):
            return np.sqrt(w @ cov_anual.values @ w)

        res = minimize(vol_obj, np.ones(n)/n, method='SLSQP',
                       bounds=bounds, constraints=constraints)
        if res.success:
            ef_vols.append(res.fun)
            ef_rets.append(target)

    fig, ax = plt.subplots(figsize=(14, 9))

    # Nuvem Monte Carlo
    scatter = ax.scatter(resultados[:, 0] * 100, resultados[:, 1] * 100,
                         c=resultados[:, 2], cmap='viridis', s=2, alpha=0.4)

    # Fronteira eficiente
    ax.plot(np.array(ef_vols) * 100, np.array(ef_rets) * 100,
            color=COLORS['red'], linewidth=3, label='Fronteira Eficiente', zorder=4)

    # Max Sharpe
    r_s, v_s, s_s = calcular_metricas_portfolio(w_sharpe, ret_anuais.values, cov_anual.values, rf)
    ax.scatter(v_s * 100, r_s * 100, c=COLORS['red'], marker='*', s=400,
               zorder=5, edgecolors='white', linewidths=1.5,
               label=f'Max Sharpe (SR={s_s:.2f})')

    # Min Var
    r_m, v_m, s_m = calcular_metricas_portfolio(w_minvar, ret_anuais.values, cov_anual.values, rf)
    ax.scatter(v_m * 100, r_m * 100, c=COLORS['amber'], marker='D', s=200,
               zorder=5, edgecolors='white', linewidths=1.5,
               label=f'Mín. Variância (SR={s_m:.2f})')

    # Ativos individuais
    for i, nome in enumerate(nomes):
        vol_i = np.sqrt(cov_anual.values[i, i]) * 100
        ret_i = ret_anuais.values[i] * 100
        ax.scatter(vol_i, ret_i, s=60, zorder=5, edgecolors='white', linewidths=0.5)
        ax.annotate(nome, (vol_i, ret_i), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, color='#c8d8e8')

    # Capital Market Line
    x_cml = np.linspace(0, resultados[:, 0].max() * 100, 100)
    y_cml = rf * 100 + s_s * x_cml
    ax.plot(x_cml, y_cml, color=COLORS['green'], linewidth=1.5,
            linestyle='--', alpha=0.7, label='Capital Market Line')

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, label='Sharpe Ratio')

    ax.set_title('FRONTEIRA EFICIENTE DE MARKOWITZ — 10 AÇÕES BR',
                 fontsize=14, fontweight='bold', color='#e8f4ff', pad=15)
    ax.set_xlabel('Volatilidade Anual (%)', fontsize=12)
    ax.set_ylabel('Retorno Anual (%)', fontsize=12)
    ax.legend(fontsize=9, framealpha=0.3, edgecolor='#1e3048', loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, 'm11_fronteira_eficiente.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Gráfico salvo: {path}")


def grafico_pesos_portfolios(nomes, w_sharpe, w_minvar):
    """Gráfico 2: Comparação de pesos dos portfólios ótimos."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cores = [COLORS['green'], COLORS['blue'], COLORS['amber'], COLORS['red'],
             COLORS['cyan'], COLORS['purple'], '#ff6b9d', '#98c379', '#d4a0ff', '#ffb86c']

    # Max Sharpe
    ax = axes[0]
    idx = np.argsort(w_sharpe)[::-1]
    nomes_sort = [nomes[i] for i in idx]
    pesos_sort = w_sharpe[idx]
    cores_sort = [cores[i] for i in idx]

    bars = ax.barh(nomes_sort, pesos_sort * 100, color=cores_sort, alpha=0.8)
    for bar, val in zip(bars, pesos_sort):
        if val > 0.01:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{val:.1%}', va='center', fontsize=9, color='#c8d8e8')
    ax.set_title('PESOS — MAX SHARPE', fontsize=12, fontweight='bold', color='#e8f4ff')
    ax.set_xlabel('Peso (%)')
    ax.grid(True, alpha=0.3)

    # Min Var
    ax = axes[1]
    idx = np.argsort(w_minvar)[::-1]
    nomes_sort = [nomes[i] for i in idx]
    pesos_sort = w_minvar[idx]
    cores_sort = [cores[i] for i in idx]

    bars = ax.barh(nomes_sort, pesos_sort * 100, color=cores_sort, alpha=0.8)
    for bar, val in zip(bars, pesos_sort):
        if val > 0.01:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{val:.1%}', va='center', fontsize=9, color='#c8d8e8')
    ax.set_title('PESOS — MÍNIMA VARIÂNCIA', fontsize=12, fontweight='bold', color='#e8f4ff')
    ax.set_xlabel('Peso (%)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, 'm11_pesos_portfolios.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Gráfico salvo: {path}")


def grafico_metricas_performance(log_retornos, nomes, w_sharpe):
    """Gráfico 3: Performance do portfólio otimizado vs benchmark."""
    print("\n── Performance do Portfólio ─────────────────────────────────")

    # Portfólio otimizado
    ret_port = (log_retornos[nomes] * w_sharpe).sum(axis=1)
    cum_port = np.exp(ret_port.cumsum()) - 1

    # Equal weight
    w_eq = np.ones(len(nomes)) / len(nomes)
    ret_eq = (log_retornos[nomes] * w_eq).sum(axis=1)
    cum_eq = np.exp(ret_eq.cumsum()) - 1

    # Calcular drawdown
    def calc_drawdown(cumret):
        wealth = (1 + cumret)
        peak = wealth.cummax()
        dd = (wealth - peak) / peak
        return dd

    dd_port = calc_drawdown(cum_port)
    dd_eq = calc_drawdown(cum_eq)

    rf = 0.1075

    # Métricas
    def calc_metricas(ret, nome):
        ret_anual = ret.mean() * 252
        vol_anual = ret.std() * np.sqrt(252)
        sharpe = (ret_anual - rf) / vol_anual
        sortino = (ret_anual - rf) / (ret[ret < 0].std() * np.sqrt(252))
        max_dd = calc_drawdown(np.exp(ret.cumsum()) - 1).min()
        calmar = ret_anual / abs(max_dd) if max_dd != 0 else 0
        return {
            'Retorno Anual': f'{ret_anual:.1%}',
            'Volatilidade': f'{vol_anual:.1%}',
            'Sharpe': f'{sharpe:.2f}',
            'Sortino': f'{sortino:.2f}',
            'Max Drawdown': f'{max_dd:.1%}',
            'Calmar': f'{calmar:.2f}',
        }

    m_port = calc_metricas(ret_port, 'Max Sharpe')
    m_eq = calc_metricas(ret_eq, 'Equal Weight')

    print(f"\n  {'Métrica':<18} {'Max Sharpe':>15} {'Equal Weight':>15}")
    print("  " + "-" * 50)
    for k in m_port:
        print(f"  {k:<18} {m_port[k]:>15} {m_eq[k]:>15}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[2, 1])

    # Retorno acumulado
    ax = axes[0]
    ax.plot(cum_port.index, cum_port * 100, color=COLORS['green'],
            linewidth=2, label='Max Sharpe Portfolio')
    ax.plot(cum_eq.index, cum_eq * 100, color=COLORS['amber'],
            linewidth=1.5, linestyle='--', label='Equal Weight')
    ax.fill_between(cum_port.index, 0, cum_port * 100, alpha=0.1, color=COLORS['green'])
    ax.axhline(y=0, color='#6a8aa8', linewidth=0.5)
    ax.set_title('RETORNO ACUMULADO — PORTFÓLIO OTIMIZADO vs EQUAL WEIGHT',
                 fontsize=12, fontweight='bold', color='#e8f4ff', pad=10)
    ax.set_ylabel('Retorno (%)')
    ax.legend(fontsize=10, framealpha=0.3, edgecolor='#1e3048')
    ax.grid(True, alpha=0.3)

    # Drawdown
    ax2 = axes[1]
    ax2.fill_between(dd_port.index, 0, dd_port * 100, alpha=0.5, color=COLORS['red'])
    ax2.plot(dd_eq.index, dd_eq * 100, color=COLORS['amber'],
             linewidth=1, linestyle='--', alpha=0.7)
    ax2.set_title('DRAWDOWN', fontsize=11, fontweight='bold', color='#e8f4ff')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, 'm11_performance_portfolio.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Gráfico salvo: {path}")


def black_litterman(log_retornos, nomes, cov_anual):
    """Gráfico 4: Modelo Black-Litterman."""
    print("\n── Black-Litterman Model ────────────────────────────────────")

    # Remover ativos com NaN na covariância (ex: delisted)
    valid_mask = ~cov_anual.isnull().any(axis=0)
    cov_clean = cov_anual.loc[valid_mask, valid_mask]
    nomes_bl = [n for n, v in zip(nomes, valid_mask) if v]
    n = len(nomes_bl)

    if n < 3:
        print("  ⚠ Ativos insuficientes para Black-Litterman.")
        return

    rf = 0.1075
    delta = 2.5  # Coeficiente de aversão ao risco

    # Capitalização de mercado (proxy: pesos iguais)
    w_mkt = np.ones(n) / n

    # Retornos de equilíbrio implícitos (CAPM reverso)
    Sigma = cov_clean.values
    Pi = delta * Sigma @ w_mkt

    # Views do gestor (exemplo)
    # View 1: PETR4 vai superar VALE3 em 5%
    # View 2: WEGE3 terá retorno absoluto de 20%
    P = np.zeros((2, n))
    if 'PETR4' in nomes_bl and 'VALE3' in nomes_bl:
        P[0, nomes_bl.index('PETR4')] = 1
        P[0, nomes_bl.index('VALE3')] = -1
    if 'WEGE3' in nomes_bl:
        P[1, nomes_bl.index('WEGE3')] = 1

    Q = np.array([0.05, 0.20])  # Retornos esperados das views

    tau = 0.05  # Escalar de incerteza
    Omega = np.diag(np.diag(tau * P @ Sigma @ P.T))

    # Proteger contra Omega singular (diagonal zero)
    Omega[Omega == 0] = 1e-6

    # Black-Litterman: retornos combinados
    inv_tau_Sigma = np.linalg.inv(tau * Sigma)
    inv_Omega = np.linalg.inv(Omega)

    mu_BL = np.linalg.inv(inv_tau_Sigma + P.T @ inv_Omega @ P) @ \
            (inv_tau_Sigma @ Pi + P.T @ inv_Omega @ Q)
    nomes = nomes_bl  # usar nomes limpos daqui em diante

    print(f"\n  Retornos de Equilíbrio (Π) vs Black-Litterman (μ_BL):")
    print(f"  {'Ativo':<10} {'Equilíbrio':>12} {'B-L':>12} {'Δ':>10}")
    print("  " + "-" * 46)
    for i, nome in enumerate(nomes):
        print(f"  {nome:<10} {Pi[i]:>12.2%} {mu_BL[i]:>12.2%} {(mu_BL[i]-Pi[i]):>10.2%}")

    # Gráfico
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Comparação retornos
    ax = axes[0]
    x_pos = np.arange(n)
    width = 0.35

    ax.barh(x_pos - width/2, Pi * 100, width, color=COLORS['blue'],
            alpha=0.8, label='Equilíbrio (CAPM)')
    ax.barh(x_pos + width/2, mu_BL * 100, width, color=COLORS['green'],
            alpha=0.8, label='Black-Litterman')

    ax.set_yticks(x_pos)
    ax.set_yticklabels(nomes)
    ax.set_xlabel('Retorno Esperado (%)')
    ax.set_title('RETORNOS: EQUILÍBRIO vs BLACK-LITTERMAN',
                 fontsize=11, fontweight='bold', color='#e8f4ff')
    ax.legend(fontsize=9, framealpha=0.3)
    ax.grid(True, alpha=0.3)

    # Otimizar com retornos BL
    def neg_sharpe_bl(w):
        ret = w @ mu_BL
        vol = np.sqrt(w @ Sigma @ w)
        return -(ret - rf) / vol

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 0.30) for _ in range(n)]
    res = minimize(neg_sharpe_bl, np.ones(n)/n, method='SLSQP',
                   bounds=bounds, constraints=constraints)
    w_bl = res.x

    # Pesos BL vs Equal Weight
    ax2 = axes[1]
    ax2.barh(x_pos - width/2, w_mkt * 100, width, color=COLORS['amber'],
             alpha=0.8, label='Equal Weight')
    ax2.barh(x_pos + width/2, w_bl * 100, width, color=COLORS['green'],
             alpha=0.8, label='Black-Litterman')

    ax2.set_yticks(x_pos)
    ax2.set_yticklabels(nomes)
    ax2.set_xlabel('Peso (%)')
    ax2.set_title('ALOCAÇÃO: EQUAL WEIGHT vs BLACK-LITTERMAN',
                  fontsize=11, fontweight='bold', color='#e8f4ff')
    ax2.legend(fontsize=9, framealpha=0.3)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, 'm11_black_litterman.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Gráfico salvo: {path}")


def main():
    precos, log_retornos, nomes = baixar_dados()
    ret_anuais, cov_anual, w_sharpe, w_minvar = otimizar_portfolio(log_retornos, nomes)

    print("\n── Gerando Gráficos ────────────────────────────────────────")
    grafico_fronteira_eficiente(log_retornos, nomes, ret_anuais, cov_anual, w_sharpe, w_minvar)
    grafico_pesos_portfolios(nomes, w_sharpe, w_minvar)
    grafico_metricas_performance(log_retornos, nomes, w_sharpe)
    black_litterman(log_retornos, nomes, cov_anual)

    print("\n" + "=" * 60)
    print("  MÓDULO 11 CONCLUÍDO ✓")
    print("  4 gráficos gerados em /graficos/")
    print("=" * 60)


if __name__ == '__main__':
    main()
