"""
=============================================================================
MÓDULO 06 — Modelos GARCH & Volatilidade
=============================================================================
Quant Academy · Finanças Quantitativas com Python

Conteúdo:
  - Volatility clustering
  - ARCH(q) e GARCH(p,q) — Engle (1982), Bollerslev (1986)
  - EGARCH (efeito alavancagem)
  - GJR-GARCH
  - Previsão de volatilidade
  - Comparação de modelos via AIC/BIC

Dados reais: IBOVESPA e PETR4 via yfinance
=============================================================================
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from arch import arch_model
from scipy import stats
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
    """Baixa dados do IBOVESPA e PETR4."""
    print("=" * 60)
    print("  MÓDULO 06 — GARCH & VOLATILIDADE")
    print("=" * 60)

    tickers = ['^BVSP', 'PETR4.SA']
    nomes = ['IBOVESPA', 'PETR4']

    dados = yf.download(tickers, start='2010-01-01', end='2025-12-31',
                        auto_adjust=True, progress=False)
    precos = dados['Close']
    precos.columns = nomes
    precos = precos.dropna()

    log_retornos = np.log(precos / precos.shift(1)).dropna()

    print(f"\n✓ {len(log_retornos)} dias | {log_retornos.index[0].strftime('%Y')} - {log_retornos.index[-1].strftime('%Y')}")
    return precos, log_retornos


def grafico_volatility_clustering(log_retornos):
    """Gráfico 1: Evidência de volatility clustering."""
    print("\n── Volatility Clustering ───────────────────────────────────")

    r = log_retornos['IBOVESPA']

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), height_ratios=[2, 1.5, 1.5])

    # Retornos
    ax = axes[0]
    colors_bar = np.where(r.values >= 0, COLORS['green'], COLORS['red'])
    ax.bar(r.index, r.values, color=colors_bar, alpha=0.6, width=1)
    ax.set_title('LOG-RETORNOS DIÁRIOS — IBOVESPA',
                 fontsize=12, fontweight='bold', color='#e8f4ff', pad=10)
    ax.set_ylabel('Log-Return')
    ax.grid(True, alpha=0.3)

    # Retornos ao quadrado (proxy de volatilidade)
    ax2 = axes[1]
    r2 = r ** 2
    ax2.fill_between(r2.index, 0, r2, alpha=0.6, color=COLORS['amber'])
    ax2.plot(r2.index, r2.rolling(21).mean(), color=COLORS['red'],
             linewidth=1.5, label='Média Móvel 21d')
    ax2.set_title('RETORNOS AO QUADRADO (Proxy de Volatilidade)',
                 fontsize=12, fontweight='bold', color='#e8f4ff', pad=10)
    ax2.set_ylabel('r²')
    ax2.legend(fontsize=9, framealpha=0.3)
    ax2.grid(True, alpha=0.3)

    # Volatilidade rolling
    ax3 = axes[2]
    vol_21 = r.rolling(21).std() * np.sqrt(252) * 100
    vol_63 = r.rolling(63).std() * np.sqrt(252) * 100
    vol_252 = r.rolling(252).std() * np.sqrt(252) * 100

    ax3.plot(vol_21.index, vol_21, color=COLORS['green'], linewidth=1, alpha=0.7, label='21d')
    ax3.plot(vol_63.index, vol_63, color=COLORS['amber'], linewidth=1.5, label='63d')
    ax3.plot(vol_252.index, vol_252, color=COLORS['red'], linewidth=2, label='252d')

    ax3.set_title('VOLATILIDADE ANUALIZADA ROLLING',
                 fontsize=12, fontweight='bold', color='#e8f4ff', pad=10)
    ax3.set_ylabel('Volatilidade (%)')
    ax3.legend(fontsize=9, framealpha=0.3)
    ax3.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, 'm06_volatility_clustering.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Gráfico salvo: {path}")

    # Teste de ARCH
    from statsmodels.stats.diagnostic import het_arch
    lm_stat, lm_pval, f_stat, f_pval = het_arch(r.dropna(), nlags=10)
    print(f"\n  Teste ARCH (Engle's LM): stat={lm_stat:.2f}, p-valor={lm_pval:.6f}")
    print(f"  → {'ARCH effects detectados!' if lm_pval < 0.05 else 'Sem ARCH effects.'}")


def ajustar_modelos_garch(log_retornos):
    """Ajusta vários modelos GARCH e compara."""
    print("\n── Ajustando Modelos GARCH ─────────────────────────────────")

    r = log_retornos['IBOVESPA'].dropna() * 100  # Escalar para estabilidade

    modelos = {}
    resultados = {}

    # GARCH(1,1) Normal
    model = arch_model(r, vol='Garch', p=1, q=1, dist='normal')
    res = model.fit(disp='off')
    modelos['GARCH(1,1) Normal'] = res
    resultados['GARCH(1,1) Normal'] = {
        'AIC': res.aic, 'BIC': res.bic, 'LogLik': res.loglikelihood
    }

    # GARCH(1,1) t-Student
    model = arch_model(r, vol='Garch', p=1, q=1, dist='t')
    res = model.fit(disp='off')
    modelos['GARCH(1,1) t-Student'] = res
    resultados['GARCH(1,1) t-Student'] = {
        'AIC': res.aic, 'BIC': res.bic, 'LogLik': res.loglikelihood
    }

    # EGARCH(1,1) t-Student
    model = arch_model(r, vol='EGARCH', p=1, q=1, dist='t')
    res = model.fit(disp='off')
    modelos['EGARCH(1,1) t-Student'] = res
    resultados['EGARCH(1,1) t-Student'] = {
        'AIC': res.aic, 'BIC': res.bic, 'LogLik': res.loglikelihood
    }

    # GJR-GARCH(1,1,1) t-Student
    model = arch_model(r, vol='Garch', p=1, o=1, q=1, dist='t')
    res = model.fit(disp='off')
    modelos['GJR-GARCH(1,1,1)'] = res
    resultados['GJR-GARCH(1,1,1)'] = {
        'AIC': res.aic, 'BIC': res.bic, 'LogLik': res.loglikelihood
    }

    # GARCH(2,2) t-Student
    model = arch_model(r, vol='Garch', p=2, q=2, dist='t')
    res = model.fit(disp='off')
    modelos['GARCH(2,2) t-Student'] = res
    resultados['GARCH(2,2) t-Student'] = {
        'AIC': res.aic, 'BIC': res.bic, 'LogLik': res.loglikelihood
    }

    df_res = pd.DataFrame(resultados).T
    df_res = df_res.sort_values('AIC')
    print("\n  Comparação de Modelos:")
    print(df_res.to_string(float_format='{:,.2f}'.format))
    print(f"\n  → Melhor modelo (AIC): {df_res.index[0]}")

    return modelos, df_res


def grafico_garch_analise(modelos, log_retornos):
    """Gráfico 2: Análise GARCH — volatilidade condicional."""
    r = log_retornos['IBOVESPA'].dropna()

    # Usar GARCH(1,1) t-Student
    res = modelos['GARCH(1,1) t-Student']
    cond_vol = res.conditional_volatility / 100  # Desfazer escala

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Retornos com bandas de volatilidade
    ax = axes[0, 0]
    ax.plot(r.index, r, alpha=0.5, linewidth=0.5, color=COLORS['blue'])
    ax.plot(r.index, 2 * cond_vol, color=COLORS['red'], linewidth=1, label='+2σ condicional')
    ax.plot(r.index, -2 * cond_vol, color=COLORS['red'], linewidth=1, label='-2σ condicional')
    ax.fill_between(r.index, -2*cond_vol, 2*cond_vol, alpha=0.1, color=COLORS['red'])
    ax.set_title('RETORNOS COM BANDAS GARCH(1,1)',
                 fontsize=11, fontweight='bold', color='#e8f4ff')
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(True, alpha=0.3)

    # Volatilidade condicional
    ax = axes[0, 1]
    vol_anual = cond_vol * np.sqrt(252) * 100
    ax.plot(vol_anual.index, vol_anual, color=COLORS['amber'], linewidth=1)
    ax.fill_between(vol_anual.index, 0, vol_anual, alpha=0.2, color=COLORS['amber'])
    ax.set_title('VOLATILIDADE CONDICIONAL ANUALIZADA (%)',
                 fontsize=11, fontweight='bold', color='#e8f4ff')
    ax.set_ylabel('Volatilidade (%)')
    ax.grid(True, alpha=0.3)

    # Resíduos padronizados
    ax = axes[1, 0]
    std_resid = res.std_resid
    ax.plot(r.index, std_resid, alpha=0.5, linewidth=0.5, color=COLORS['cyan'])
    ax.axhline(y=2, color=COLORS['red'], linestyle='--', alpha=0.5)
    ax.axhline(y=-2, color=COLORS['red'], linestyle='--', alpha=0.5)
    ax.set_title('RESÍDUOS PADRONIZADOS GARCH',
                 fontsize=11, fontweight='bold', color='#e8f4ff')
    ax.grid(True, alpha=0.3)

    # QQ-Plot dos resíduos
    ax = axes[1, 1]
    stats.probplot(std_resid, dist='norm', plot=ax)
    ax.get_lines()[0].set(color=COLORS['cyan'], markersize=2, alpha=0.5)
    ax.get_lines()[1].set(color=COLORS['red'], linewidth=1.5)
    ax.set_title('QQ-PLOT RESÍDUOS PADRONIZADOS',
                 fontsize=11, fontweight='bold', color='#e8f4ff')
    ax.grid(True, alpha=0.3)

    for ax in axes.flat[:3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, 'm06_garch_analise.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Gráfico salvo: {path}")


def grafico_comparacao_modelos(df_resultados):
    """Gráfico 3: Comparação AIC/BIC dos modelos."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    modelos_nomes = df_resultados.index.tolist()
    cores = [COLORS['green'], COLORS['blue'], COLORS['amber'], COLORS['red'], COLORS['purple']]

    # AIC
    ax = axes[0]
    bars = ax.barh(modelos_nomes, df_resultados['AIC'], color=cores, alpha=0.8)
    ax.set_title('COMPARAÇÃO AIC (menor = melhor)',
                 fontsize=11, fontweight='bold', color='#e8f4ff')
    ax.grid(True, alpha=0.3)
    for bar, val in zip(bars, df_resultados['AIC']):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                f' {val:,.0f}', va='center', fontsize=9, color='#c8d8e8')

    # BIC
    ax = axes[1]
    bars = ax.barh(modelos_nomes, df_resultados['BIC'], color=cores, alpha=0.8)
    ax.set_title('COMPARAÇÃO BIC (menor = melhor)',
                 fontsize=11, fontweight='bold', color='#e8f4ff')
    ax.grid(True, alpha=0.3)
    for bar, val in zip(bars, df_resultados['BIC']):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                f' {val:,.0f}', va='center', fontsize=9, color='#c8d8e8')

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, 'm06_comparacao_aic_bic.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Gráfico salvo: {path}")


def grafico_previsao_volatilidade(modelos, log_retornos):
    """Gráfico 4: Previsão de volatilidade futura."""
    print("\n── Previsão de Volatilidade ─────────────────────────────────")

    r = log_retornos['IBOVESPA'].dropna() * 100
    horizonte = 30

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Previsão com diferentes modelos
    ax = axes[0]
    cores_list = [COLORS['green'], COLORS['blue'], COLORS['amber'], COLORS['red']]
    modelos_plot = ['GARCH(1,1) t-Student', 'EGARCH(1,1) t-Student', 'GJR-GARCH(1,1,1)']

    for i, nome in enumerate(modelos_plot):
        res = modelos[nome]
        # EGARCH não suporta previsão analítica para horizon > 1
        # Usar method='simulation' para modelos que não suportam analítico
        try:
            forecast = res.forecast(horizon=horizonte, method='analytic')
        except ValueError:
            forecast = res.forecast(horizon=horizonte, method='simulation',
                                    simulations=1000)
        vol_forecast = np.sqrt(forecast.variance.values[-1, :]) / 100 * np.sqrt(252) * 100

        ax.plot(range(1, horizonte + 1), vol_forecast,
                color=cores_list[i], linewidth=2, marker='o', markersize=3,
                label=nome)

    ax.set_title(f'PREVISÃO DE VOLATILIDADE — PRÓXIMOS {horizonte} DIAS',
                 fontsize=11, fontweight='bold', color='#e8f4ff')
    ax.set_xlabel('Horizonte (dias)')
    ax.set_ylabel('Volatilidade Anualizada (%)')
    ax.legend(fontsize=8, framealpha=0.3, edgecolor='#1e3048')
    ax.grid(True, alpha=0.3)

    # News Impact Curve
    ax2 = axes[1]
    res_garch = modelos['GARCH(1,1) t-Student']
    params = res_garch.params

    omega = params['omega']
    alpha = params['alpha[1]']
    beta = params['beta[1]']

    # Variância de longo prazo
    long_run_var = omega / (1 - alpha - beta)

    shocks = np.linspace(-5, 5, 200)
    # Impact: σ²_t+1 = ω + α * shock² + β * σ²_t
    impact = omega + alpha * shocks**2 + beta * long_run_var

    ax2.plot(shocks, np.sqrt(impact) * np.sqrt(252), color=COLORS['green'],
             linewidth=2, label='GARCH(1,1)')
    ax2.axvline(x=0, color='#6a8aa8', linewidth=0.5, linestyle='--')

    ax2.set_title('NEWS IMPACT CURVE',
                 fontsize=11, fontweight='bold', color='#e8f4ff')
    ax2.set_xlabel('Retorno Padronizado (z)')
    ax2.set_ylabel('Volatilidade Condicional (σ anualizada)')
    ax2.legend(fontsize=9, framealpha=0.3)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, 'm06_previsao_volatilidade.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Gráfico salvo: {path}")

    # Imprimir parâmetros do melhor modelo
    res = modelos['GARCH(1,1) t-Student']
    print(f"\n  Parâmetros GARCH(1,1) t-Student:")
    print(f"    ω (omega)  = {res.params['omega']:.6f}")
    print(f"    α (alpha)  = {res.params['alpha[1]']:.4f}")
    print(f"    β (beta)   = {res.params['beta[1]']:.4f}")
    print(f"    α + β      = {res.params['alpha[1]'] + res.params['beta[1]']:.4f}")
    print(f"    Persistência: {'Alta' if res.params['alpha[1]'] + res.params['beta[1]'] > 0.95 else 'Moderada'}")


def main():
    precos, log_retornos = baixar_dados()

    print("\n── Gerando Gráficos ────────────────────────────────────────")
    grafico_volatility_clustering(log_retornos)
    modelos, df_res = ajustar_modelos_garch(log_retornos)
    grafico_garch_analise(modelos, log_retornos)
    grafico_comparacao_modelos(df_res)
    grafico_previsao_volatilidade(modelos, log_retornos)

    # Salvar resultados
    df_res.to_csv(os.path.join(DATA_DIR, 'garch_comparacao_modelos.csv'))

    print("\n" + "=" * 60)
    print("  MÓDULO 06 CONCLUÍDO ✓")
    print("  4 gráficos gerados em /graficos/")
    print("=" * 60)


if __name__ == '__main__':
    main()
