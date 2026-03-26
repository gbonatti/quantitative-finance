"""
=============================================================================
MÓDULO 08 — Value at Risk (VaR) & CVaR (Expected Shortfall)
Quant Academy · Finanças Quantitativas
=============================================================================
As principais métricas de risco de mercado usadas por bancos, fundos e
reguladores. Implementação dos 3 métodos: Paramétrico, Histórico e Monte Carlo.
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
# 1. COLETA DE DADOS — PORTFÓLIO DE 5 AÇÕES BRASILEIRAS
# ══════════════════════════════════════════════════════════════════════════════
def carregar_portfolio():
    """Baixa dados de 5 ações e calcula retornos do portfólio."""
    import yfinance as yf

    tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBAS3.SA', 'WEGE3.SA']
    print("📥 Baixando dados do portfólio...")
    print(f"   Ativos: {', '.join(tickers)}")

    dados = yf.download(tickers, period='2y', auto_adjust=True, progress=False)
    precos = dados['Close'].dropna()

    # Retornos log
    retornos = np.log(precos / precos.shift(1)).dropna()

    # Portfólio equal-weight
    n_ativos = len(tickers)
    pesos = np.array([1 / n_ativos] * n_ativos)
    retornos_portfolio = retornos @ pesos

    print(f"   ✔ {len(retornos)} dias de retornos")
    print(f"   ✔ Pesos: {pesos}")
    print(f"   ✔ Retorno médio diário: {retornos_portfolio.mean():.4%}")
    print(f"   ✔ Volatilidade diária: {retornos_portfolio.std():.4%}")

    return retornos, retornos_portfolio, pesos, tickers


# ══════════════════════════════════════════════════════════════════════════════
# 2. CÁLCULO DO VaR — 3 MÉTODOS
# ══════════════════════════════════════════════════════════════════════════════
def var_parametrico(retornos, confidence=0.95):
    """VaR Paramétrico (assume distribuição Normal)."""
    mu = retornos.mean()
    sigma = retornos.std()
    z = stats.norm.ppf(1 - confidence)
    var = -(mu + z * sigma)
    return var


def var_historico(retornos, confidence=0.95):
    """VaR Histórico (percentil empírico)."""
    alpha = 1 - confidence
    var = -np.percentile(retornos, alpha * 100)
    return var


def var_monte_carlo(retornos, confidence=0.95, n_sim=100_000):
    """VaR Monte Carlo (simula retornos futuros com GBM)."""
    mu = retornos.mean()
    sigma = retornos.std()
    # Simular retornos
    sim = np.random.normal(mu, sigma, n_sim)
    alpha = 1 - confidence
    var = -np.percentile(sim, alpha * 100)
    return var


def cvar(retornos, confidence=0.95, metodo='historico'):
    """CVaR (Expected Shortfall) — perda esperada na cauda."""
    alpha = 1 - confidence
    if metodo == 'historico':
        var_pct = np.percentile(retornos, alpha * 100)
        tail = retornos[retornos <= var_pct]
        return -tail.mean()
    elif metodo == 'parametrico':
        mu = retornos.mean()
        sigma = retornos.std()
        z = stats.norm.ppf(alpha)
        cvar_val = -(mu - sigma * stats.norm.pdf(z) / alpha)
        return cvar_val
    elif metodo == 'monte_carlo':
        mu = retornos.mean()
        sigma = retornos.std()
        sim = np.random.normal(mu, sigma, 100_000)
        var_pct = np.percentile(sim, alpha * 100)
        tail = sim[sim <= var_pct]
        return -tail.mean()


# ══════════════════════════════════════════════════════════════════════════════
# 3. GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════
def grafico_distribuicao_var(retornos_port, valor_portfolio=1_000_000):
    """Distribuição de retornos com linhas de VaR e CVaR."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor(COLORS['bg'])

    for conf, ax in zip([0.95, 0.99], axes):
        ax.set_facecolor(COLORS['surface'])

        # Histograma
        n, bins, patches = ax.hist(retornos_port * 100, bins=80,
                                    density=True, alpha=0.6,
                                    color=COLORS['blue'], edgecolor='none')

        # Colorir cauda
        var_val = var_historico(retornos_port, conf)
        for b, p in zip(bins, patches):
            if b < -var_val * 100:
                p.set_facecolor(COLORS['red'])
                p.set_alpha(0.8)

        # Linhas VaR e CVaR
        var_h = var_historico(retornos_port, conf)
        cvar_h = cvar(retornos_port, conf, 'historico')

        ax.axvline(-var_h * 100, color=COLORS['amber'], linewidth=2,
                   linestyle='--', label=f'VaR {conf*100:.0f}%: {var_h:.2%}')
        ax.axvline(-cvar_h * 100, color=COLORS['red'], linewidth=2,
                   linestyle='--', label=f'CVaR {conf*100:.0f}%: {cvar_h:.2%}')

        # Monetário
        var_brl = var_h * valor_portfolio
        cvar_brl = cvar_h * valor_portfolio

        ax.set_title(f'DISTRIBUIÇÃO DE RETORNOS · VaR {conf*100:.0f}%',
                     fontsize=13, color=COLORS['text'], fontweight='bold', pad=15)
        ax.set_xlabel('Retorno Diário (%)', color=COLORS['dim'])
        ax.set_ylabel('Densidade', color=COLORS['dim'])
        ax.tick_params(colors=COLORS['dim'])

        # Info box
        info_text = (f'Portfolio: R$ {valor_portfolio:,.0f}\n'
                     f'VaR: R$ {var_brl:,.0f}\n'
                     f'CVaR: R$ {cvar_brl:,.0f}')
        ax.text(0.02, 0.95, info_text, transform=ax.transAxes,
                fontsize=9, color=COLORS['text'],
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=COLORS['bg'], alpha=0.8))

        ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    plt.savefig('../graficos/m08_distribuicao_var_cvar.png', **SAVE)
    plt.close()
    print("   ✔ Gráfico salvo: graficos/m08_distribuicao_var_cvar.png")


def grafico_comparacao_metodos(retornos_port, valor_portfolio=1_000_000):
    """Compara os 3 métodos de VaR em diferentes níveis de confiança."""
    confiancas = [0.90, 0.95, 0.99, 0.999]

    metodos_var = {
        'Paramétrico': var_parametrico,
        'Histórico': var_historico,
        'Monte Carlo': var_monte_carlo,
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(COLORS['bg'])

    cores_metodo = [COLORS['green'], COLORS['blue'], COLORS['amber']]

    # --- VaR ---
    ax = axes[0]
    ax.set_facecolor(COLORS['surface'])

    x = np.arange(len(confiancas))
    width = 0.25

    for i, (nome, func) in enumerate(metodos_var.items()):
        vars_val = [func(retornos_port, c) * valor_portfolio for c in confiancas]
        bars = ax.bar(x + i * width, vars_val, width, color=cores_metodo[i],
                      alpha=0.8, label=nome)
        for bar, val in zip(bars, vars_val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                    f'R${val:,.0f}', ha='center', va='bottom', fontsize=7,
                    color=cores_metodo[i])

    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{c*100:.0f}%' for c in confiancas], color=COLORS['text'])
    ax.set_ylabel('VaR (R$)', color=COLORS['dim'])
    ax.set_title('VALUE AT RISK · COMPARAÇÃO DE MÉTODOS',
                 fontsize=13, color=COLORS['text'], fontweight='bold', pad=15)
    ax.legend(fontsize=9)
    ax.tick_params(colors=COLORS['dim'])

    # --- CVaR ---
    ax2 = axes[1]
    ax2.set_facecolor(COLORS['surface'])

    metodos_cvar = ['parametrico', 'historico', 'monte_carlo']
    nomes_cvar = ['Paramétrico', 'Histórico', 'Monte Carlo']

    for i, (metodo, nome) in enumerate(zip(metodos_cvar, nomes_cvar)):
        cvars_val = [cvar(retornos_port, c, metodo) * valor_portfolio
                     for c in confiancas]
        bars = ax2.bar(x + i * width, cvars_val, width, color=cores_metodo[i],
                       alpha=0.8, label=nome)
        for bar, val in zip(bars, cvars_val):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                     f'R${val:,.0f}', ha='center', va='bottom', fontsize=7,
                     color=cores_metodo[i])

    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f'{c*100:.0f}%' for c in confiancas], color=COLORS['text'])
    ax2.set_ylabel('CVaR (R$)', color=COLORS['dim'])
    ax2.set_title('EXPECTED SHORTFALL (CVaR) · COMPARAÇÃO',
                  fontsize=13, color=COLORS['text'], fontweight='bold', pad=15)
    ax2.legend(fontsize=9)
    ax2.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m08_comparacao_metodos.png', **SAVE)
    plt.close()
    print("   ✔ Gráfico salvo: graficos/m08_comparacao_metodos.png")


def grafico_var_rolling(retornos_port, janela=60, confidence=0.95):
    """VaR rolling (janela deslizante) ao longo do tempo."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                              gridspec_kw={'height_ratios': [2, 1]})
    fig.patch.set_facecolor(COLORS['bg'])

    # VaR rolling
    var_rolling = retornos_port.rolling(janela).apply(
        lambda x: -np.percentile(x, (1 - confidence) * 100), raw=True
    )
    cvar_rolling = retornos_port.rolling(janela).apply(
        lambda x: -x[x <= np.percentile(x, (1 - confidence) * 100)].mean(), raw=True
    )

    # Painel superior: retornos + VaR
    ax = axes[0]
    ax.set_facecolor(COLORS['surface'])
    ax.fill_between(retornos_port.index, retornos_port * 100, 0,
                    where=retornos_port > 0, alpha=0.3, color=COLORS['green'])
    ax.fill_between(retornos_port.index, retornos_port * 100, 0,
                    where=retornos_port < 0, alpha=0.3, color=COLORS['red'])
    ax.plot(-var_rolling * 100, color=COLORS['amber'], linewidth=1.5,
            label=f'VaR {confidence*100:.0f}% ({janela}d)')
    ax.plot(-cvar_rolling * 100, color=COLORS['red'], linewidth=1.5,
            label=f'CVaR {confidence*100:.0f}% ({janela}d)')

    # Marcar violações
    violacoes = retornos_port[retornos_port < -var_rolling]
    ax.scatter(violacoes.index, violacoes * 100, color=COLORS['red'],
               s=30, zorder=5, marker='x', label=f'Violações: {len(violacoes)}')

    ax.set_title(f'RETORNOS DIÁRIOS · VaR/CVaR ROLLING {janela} DIAS',
                 fontsize=14, color=COLORS['text'], fontweight='bold', pad=15)
    ax.set_ylabel('Retorno (%)', color=COLORS['dim'])
    ax.legend(fontsize=9)
    ax.tick_params(colors=COLORS['dim'])

    # Painel inferior: Volatilidade rolling
    ax2 = axes[1]
    ax2.set_facecolor(COLORS['surface'])
    vol_rolling = retornos_port.rolling(janela).std() * np.sqrt(252) * 100
    ax2.fill_between(vol_rolling.index, vol_rolling, alpha=0.3, color=COLORS['blue'])
    ax2.plot(vol_rolling, color=COLORS['blue'], linewidth=1)
    ax2.set_title(f'VOLATILIDADE ANUALIZADA ROLLING ({janela}d)',
                  fontsize=11, color=COLORS['dim'], pad=10)
    ax2.set_ylabel('Volatilidade (%)', color=COLORS['dim'])
    ax2.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m08_var_rolling.png', **SAVE)
    plt.close()
    print("   ✔ Gráfico salvo: graficos/m08_var_rolling.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  MÓDULO 08 · VALUE AT RISK (VaR) & CVaR")
    print("=" * 70)

    retornos, retornos_port, pesos, tickers = carregar_portfolio()
    valor_portfolio = 1_000_000

    # ── Cálculos ───────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  PORTFÓLIO: R$ {valor_portfolio:,.0f}")
    print(f"  ATIVOS: {', '.join(tickers)} (equal-weight)")
    print(f"{'─' * 70}")

    for conf in [0.95, 0.99]:
        print(f"\n  ═══ Nível de Confiança: {conf*100:.0f}% ═══")

        vp = var_parametrico(retornos_port, conf)
        vh = var_historico(retornos_port, conf)
        vmc = var_monte_carlo(retornos_port, conf)

        cp = cvar(retornos_port, conf, 'parametrico')
        ch = cvar(retornos_port, conf, 'historico')
        cmc = cvar(retornos_port, conf, 'monte_carlo')

        print(f"\n  {'Método':<18} {'VaR (%)':>10} {'VaR (R$)':>15} {'CVaR (%)':>10} {'CVaR (R$)':>15}")
        print(f"  {'─'*68}")
        print(f"  {'Paramétrico':<18} {vp:>10.2%} {vp*valor_portfolio:>15,.0f} {cp:>10.2%} {cp*valor_portfolio:>15,.0f}")
        print(f"  {'Histórico':<18} {vh:>10.2%} {vh*valor_portfolio:>15,.0f} {ch:>10.2%} {ch*valor_portfolio:>15,.0f}")
        print(f"  {'Monte Carlo':<18} {vmc:>10.2%} {vmc*valor_portfolio:>15,.0f} {cmc:>10.2%} {cmc*valor_portfolio:>15,.0f}")

    # ── Gráficos ───────────────────────────────────────────────────────────
    print("\n🎨 Gerando gráficos...")
    grafico_distribuicao_var(retornos_port, valor_portfolio)
    grafico_comparacao_metodos(retornos_port, valor_portfolio)
    grafico_var_rolling(retornos_port)

    print(f"\n{'=' * 70}")
    print("  ✅ Módulo 08 concluído!")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
