"""
=============================================================================
MÓDULO 01 — Introdução às Finanças Quantitativas: Retornos Financeiros
=============================================================================
Quant Academy · Finanças Quantitativas com Python

Conteúdo:
  - Retorno simples vs log-return
  - Retorno acumulado
  - Volatilidade anualizada
  - Sharpe Ratio
  - Gráficos: preços, retornos, distribuição, retorno acumulado

Dados reais: ações brasileiras (PETR4, VALE3, ITUB4, BBDC4, ABEV3) via yfinance
=============================================================================
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
    'green': '#00d4a0',
    'amber': '#f5a623',
    'red': '#e05252',
    'blue': '#4a9eff',
    'cyan': '#7ec8e3',
    'purple': '#b07ee8',
}

# ── Diretório de saída ─────────────────────────────────────────────────────
GRAFICOS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'graficos')
os.makedirs(GRAFICOS_DIR, exist_ok=True)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)


def baixar_dados():
    """Baixa dados de ações brasileiras reais."""
    print("=" * 60)
    print("  MÓDULO 01 — RETORNOS FINANCEIROS")
    print("  Baixando dados reais de ações brasileiras...")
    print("=" * 60)

    tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA']
    nomes = ['PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'ABEV3']

    dados = yf.download(tickers, start='2020-01-01', end='2025-12-31',
                        auto_adjust=True, progress=False)

    precos = dados['Close']
    precos.columns = nomes
    precos = precos.dropna()

    # Salvar dados
    precos.to_csv(os.path.join(DATA_DIR, 'precos_acoes_br.csv'))
    print(f"\n✓ Dados salvos: {len(precos)} dias de negociação")
    print(f"  Período: {precos.index[0].strftime('%Y-%m-%d')} a {precos.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Ativos: {', '.join(nomes)}")

    return precos, nomes


def calcular_retornos(precos):
    """Calcula retornos simples e logarítmicos."""
    print("\n── Calculando Retornos ──────────────────────────────────────")

    # Retorno simples
    retornos_simples = precos.pct_change().dropna()

    # Retorno logarítmico
    log_retornos = np.log(precos / precos.shift(1)).dropna()

    # Retorno acumulado
    retorno_acumulado = (1 + retornos_simples).cumprod() - 1

    # Comparação entre retornos simples e log
    print("\n  Comparação: Retorno Simples vs Log-Return (primeiros 5 dias):")
    print(f"  {'Ativo':<10} {'Simples':>12} {'Log':>12} {'Diferença':>12}")
    print("  " + "-" * 48)
    for col in precos.columns[:3]:
        rs = retornos_simples[col].iloc[0]
        rl = log_retornos[col].iloc[0]
        print(f"  {col:<10} {rs:>12.6f} {rl:>12.6f} {abs(rs-rl):>12.8f}")

    return retornos_simples, log_retornos, retorno_acumulado


def estatisticas_retornos(log_retornos):
    """Calcula estatísticas descritivas dos retornos."""
    print("\n── Estatísticas de Retornos Anualizados ─────────────────────")

    stats_dict = {}
    for col in log_retornos.columns:
        r = log_retornos[col]
        media_anual = r.mean() * 252
        vol_anual = r.std() * np.sqrt(252)
        rf = 0.1075  # SELIC aproximada
        sharpe = (media_anual - rf) / vol_anual
        max_dd = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()

        stats_dict[col] = {
            'Retorno Anual (%)': media_anual * 100,
            'Volatilidade Anual (%)': vol_anual * 100,
            'Sharpe Ratio': sharpe,
            'Max Drawdown (%)': max_dd * 100,
            'Assimetria': r.skew(),
            'Curtose': r.kurtosis(),
        }

    df_stats = pd.DataFrame(stats_dict).T
    print("\n" + df_stats.to_string(float_format='{:,.2f}'.format))
    print(f"\n  * Taxa livre de risco (SELIC): {rf*100:.2f}% a.a.")

    return df_stats


def grafico_precos(precos, nomes):
    """Gráfico 1: Evolução de preços normalizados."""
    fig, ax = plt.subplots(figsize=(14, 6))

    precos_norm = precos / precos.iloc[0] * 100
    cores = list(COLORS.values())

    for i, col in enumerate(precos_norm.columns):
        ax.plot(precos_norm.index, precos_norm[col],
                color=cores[i % len(cores)], linewidth=1.5, label=col, alpha=0.9)

    ax.set_title('EVOLUÇÃO DE PREÇOS NORMALIZADOS (BASE 100)',
                 fontsize=13, fontweight='bold', color='#e8f4ff', pad=15)
    ax.set_ylabel('Preço Normalizado')
    ax.legend(loc='upper left', framealpha=0.3, edgecolor='#1e3048')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color='#6a8aa8', linestyle='--', alpha=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, 'm01_precos_normalizados.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Gráfico salvo: {path}")


def grafico_retornos(log_retornos):
    """Gráfico 2: Retornos diários de PETR4."""
    ativo = log_retornos.columns[0]
    r = log_retornos[ativo]

    fig, ax = plt.subplots(figsize=(14, 5))

    colors_bar = np.where(r >= 0, COLORS['green'], COLORS['red'])
    ax.bar(r.index, r.values, color=colors_bar, alpha=0.7, width=1)

    # Bandas de ±2σ
    sigma = r.std()
    ax.axhline(y=2*sigma, color=COLORS['amber'], linestyle='--', alpha=0.7, label=f'+2σ = {2*sigma:.4f}')
    ax.axhline(y=-2*sigma, color=COLORS['amber'], linestyle='--', alpha=0.7, label=f'-2σ = {-2*sigma:.4f}')
    ax.axhline(y=0, color='#6a8aa8', linewidth=0.5)

    ax.set_title(f'LOG-RETORNOS DIÁRIOS — {ativo}',
                 fontsize=13, fontweight='bold', color='#e8f4ff', pad=15)
    ax.set_ylabel('Log-Return')
    ax.legend(loc='upper right', framealpha=0.3, edgecolor='#1e3048')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, 'm01_retornos_diarios.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Gráfico salvo: {path}")


def grafico_distribuicao(log_retornos):
    """Gráfico 3: Distribuição de retornos vs Normal."""
    ativo = log_retornos.columns[0]
    r = log_retornos[ativo].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histograma
    ax = axes[0]
    n, bins, patches = ax.hist(r, bins=80, density=True, alpha=0.7,
                                color=COLORS['blue'], edgecolor='none')

    # Fit normal
    mu, sigma = r.mean(), r.std()
    x = np.linspace(r.min(), r.max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color=COLORS['amber'],
            linewidth=2, label=f'Normal(μ={mu:.4f}, σ={sigma:.4f})')

    # Fit t-Student
    params = stats.t.fit(r)
    ax.plot(x, stats.t.pdf(x, *params), color=COLORS['red'],
            linewidth=2, linestyle='--', label=f't-Student(ν={params[0]:.1f})')

    # VaR 95%
    var_95 = mu - 1.645 * sigma
    ax.axvline(x=var_95, color=COLORS['red'], linestyle=':', alpha=0.8,
               label=f'VaR 95% = {var_95:.4f}')

    ax.set_title(f'DISTRIBUIÇÃO DE LOG-RETORNOS — {ativo}',
                 fontsize=11, fontweight='bold', color='#e8f4ff')
    ax.legend(fontsize=8, framealpha=0.3, edgecolor='#1e3048')
    ax.grid(True, alpha=0.3)

    # QQ-Plot
    ax2 = axes[1]
    stats.probplot(r, dist='norm', plot=ax2)
    ax2.get_lines()[0].set(color=COLORS['cyan'], markersize=2, alpha=0.5)
    ax2.get_lines()[1].set(color=COLORS['red'], linewidth=1.5)
    ax2.set_title(f'QQ-PLOT — {ativo} vs Normal',
                  fontsize=11, fontweight='bold', color='#e8f4ff')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, 'm01_distribuicao_retornos.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Gráfico salvo: {path}")


def grafico_retorno_acumulado(retorno_acumulado):
    """Gráfico 4: Retorno acumulado."""
    fig, ax = plt.subplots(figsize=(14, 6))

    cores = list(COLORS.values())
    for i, col in enumerate(retorno_acumulado.columns):
        ax.fill_between(retorno_acumulado.index, 0, retorno_acumulado[col],
                        alpha=0.1, color=cores[i % len(cores)])
        ax.plot(retorno_acumulado.index, retorno_acumulado[col],
                color=cores[i % len(cores)], linewidth=1.5, label=col)

    ax.set_title('RETORNO ACUMULADO',
                 fontsize=13, fontweight='bold', color='#e8f4ff', pad=15)
    ax.set_ylabel('Retorno Acumulado')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.axhline(y=0, color='#6a8aa8', linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', framealpha=0.3, edgecolor='#1e3048')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, 'm01_retorno_acumulado.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Gráfico salvo: {path}")


def main():
    """Execução principal do Módulo 01."""
    precos, nomes = baixar_dados()
    retornos_simples, log_retornos, retorno_acumulado = calcular_retornos(precos)
    df_stats = estatisticas_retornos(log_retornos)

    # Salvar estatísticas
    df_stats.to_csv(os.path.join(DATA_DIR, 'estatisticas_retornos.csv'))

    # Gerar gráficos
    print("\n── Gerando Gráficos ────────────────────────────────────────")
    grafico_precos(precos, nomes)
    grafico_retornos(log_retornos)
    grafico_distribuicao(log_retornos)
    grafico_retorno_acumulado(retorno_acumulado)

    print("\n" + "=" * 60)
    print("  MÓDULO 01 CONCLUÍDO ✓")
    print("  4 gráficos gerados em /graficos/")
    print("  Dados salvos em /data/")
    print("=" * 60)


if __name__ == '__main__':
    main()
