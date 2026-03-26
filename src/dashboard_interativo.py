"""
=============================================================================
DASHBOARD INTERATIVO — Finanças Quantitativas
=============================================================================
Quant Academy · Dashboard com Plotly Dash

Features:
  - Análise de ações em tempo real (yfinance)
  - Gráficos de preço com candlestick
  - Retornos e volatilidade rolling
  - VaR/CVaR dinâmico
  - Correlação entre ativos
  - Simulação Monte Carlo
  - Calculadora Black-Scholes interativa

Execute: python dashboard_interativo.py
Acesse: http://localhost:8050
=============================================================================
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from dash import Dash, html, dcc, callback, Output, Input, State
from io import StringIO
import warnings

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════════════════

TEMPLATE = 'plotly_dark'
COLORS = {
    'bg': '#080c10',
    'surface': '#0d1520',
    'border': '#1e3048',
    'green': '#00d4a0',
    'amber': '#f5a623',
    'red': '#e05252',
    'blue': '#4a9eff',
    'cyan': '#7ec8e3',
    'text': '#c8d8e8',
    'text_dim': '#6a8aa8',
    'purple': '#b07ee8',
}

TICKERS_BR = {
    'PETR4.SA': 'PETR4 — Petrobras',
    'VALE3.SA': 'VALE3 — Vale',
    'ITUB4.SA': 'ITUB4 — Itaú Unibanco',
    'BBDC4.SA': 'BBDC4 — Bradesco',
    'ABEV3.SA': 'ABEV3 — Ambev',
    'WEGE3.SA': 'WEGE3 — WEG',
    'RENT3.SA': 'RENT3 — Localiza',
    'BBAS3.SA': 'BBAS3 — Banco do Brasil',
    'SUZB3.SA': 'SUZB3 — Suzano',
    'GGBR4.SA': 'GGBR4 — Gerdau',
    '^BVSP': 'IBOVESPA',
}


# ══════════════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES
# ══════════════════════════════════════════════════════════════════════════

def black_scholes(S, K, r, sigma, T, option='call'):
    """Black-Scholes pricing."""
    if T <= 0:
        intrinsic = max(S - K, 0) if option == 'call' else max(K - S, 0)
        return intrinsic, 0, 0, 0, 0

    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    if option == 'call':
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        delta = norm.cdf(d1) - 1

    gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
    theta = (-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
    vega = S*norm.pdf(d1)*np.sqrt(T) / 100

    return price, delta, gamma, theta, vega


# ══════════════════════════════════════════════════════════════════════════
# APP DASH
# ══════════════════════════════════════════════════════════════════════════

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Quant Academy — Dashboard"

# ── LAYOUT ─────────────────────────────────────────────────────────────
app.layout = html.Div(style={
    'backgroundColor': COLORS['bg'],
    'color': COLORS['text'],
    'fontFamily': 'Consolas, monospace',
    'minHeight': '100vh',
    'padding': '20px',
}, children=[

    # Header
    html.Div(style={
        'textAlign': 'center',
        'borderBottom': f'1px solid {COLORS["border"]}',
        'paddingBottom': '20px',
        'marginBottom': '30px'
    }, children=[
        html.H1("QUANT ACADEMY", style={
            'color': COLORS['green'], 'fontSize': '32px',
            'letterSpacing': '5px', 'marginBottom': '5px'}),
        html.P("Dashboard Interativo — Finanças Quantitativas",
               style={'color': COLORS['text_dim'], 'fontSize': '14px'}),
    ]),

    # ── Controles Globais ──
    html.Div(style={
        'display': 'flex', 'gap': '20px', 'marginBottom': '30px',
        'flexWrap': 'wrap', 'alignItems': 'flex-end'
    }, children=[
        html.Div([
            html.Label("ATIVOS", style={'fontSize': '10px', 'color': COLORS['green'],
                                         'letterSpacing': '2px'}),
            dcc.Dropdown(
                id='ticker-select',
                options=[{'label': v, 'value': k} for k, v in TICKERS_BR.items()],
                value=['PETR4.SA', 'VALE3.SA', 'ITUB4.SA'],
                multi=True,
                style={'width': '500px', 'backgroundColor': COLORS['surface']},
            ),
        ]),
        html.Div([
            html.Label("PERÍODO", style={'fontSize': '10px', 'color': COLORS['green'],
                                          'letterSpacing': '2px'}),
            dcc.Dropdown(
                id='period-select',
                options=[
                    {'label': '6 meses', 'value': '6mo'},
                    {'label': '1 ano', 'value': '1y'},
                    {'label': '2 anos', 'value': '2y'},
                    {'label': '5 anos', 'value': '5y'},
                ],
                value='2y',
                style={'width': '150px', 'backgroundColor': COLORS['surface']},
            ),
        ]),
        html.Button('ATUALIZAR', id='update-btn', n_clicks=0, style={
            'backgroundColor': COLORS['green'], 'color': COLORS['bg'],
            'border': 'none', 'padding': '10px 25px', 'cursor': 'pointer',
            'fontFamily': 'monospace', 'letterSpacing': '2px', 'fontWeight': 'bold',
        }),
    ]),

    # ── Abas ──
    dcc.Tabs(id='tabs', value='tab-price', style={
        'borderBottom': f'1px solid {COLORS["border"]}',
    }, children=[
        dcc.Tab(label='📈 PREÇOS & RETORNOS', value='tab-price',
                style={'backgroundColor': COLORS['surface'], 'color': COLORS['text_dim'],
                       'border': f'1px solid {COLORS["border"]}'},
                selected_style={'backgroundColor': COLORS['bg'], 'color': COLORS['green'],
                                'borderTop': f'2px solid {COLORS["green"]}'}),
        dcc.Tab(label='📊 RISCO (VaR/CVaR)', value='tab-risk',
                style={'backgroundColor': COLORS['surface'], 'color': COLORS['text_dim'],
                       'border': f'1px solid {COLORS["border"]}'},
                selected_style={'backgroundColor': COLORS['bg'], 'color': COLORS['green'],
                                'borderTop': f'2px solid {COLORS["green"]}'}),
        dcc.Tab(label='🎯 MONTE CARLO', value='tab-mc',
                style={'backgroundColor': COLORS['surface'], 'color': COLORS['text_dim'],
                       'border': f'1px solid {COLORS["border"]}'},
                selected_style={'backgroundColor': COLORS['bg'], 'color': COLORS['green'],
                                'borderTop': f'2px solid {COLORS["green"]}'}),
        dcc.Tab(label='⚡ BLACK-SCHOLES', value='tab-bs',
                style={'backgroundColor': COLORS['surface'], 'color': COLORS['text_dim'],
                       'border': f'1px solid {COLORS["border"]}'},
                selected_style={'backgroundColor': COLORS['bg'], 'color': COLORS['green'],
                                'borderTop': f'2px solid {COLORS["green"]}'}),
    ]),

    # ── Conteúdo das Abas ──
    html.Div(id='tab-content', style={'marginTop': '20px'}),

    # ── Black-Scholes (sempre no layout, visibilidade controlada por CSS) ──
    html.Div(id='bs-section', style={'display': 'none', 'marginTop': '20px'}, children=[
        html.Div(style={
            'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap',
            'backgroundColor': COLORS['surface'],
            'border': f'1px solid {COLORS["border"]}',
            'padding': '25px', 'marginBottom': '20px',
        }, children=[
            html.Div([
                html.Label("Preço Ativo (S)", style={'fontSize': '10px', 'color': COLORS['green']}),
                dcc.Input(id='bs-S', type='number', value=100, step=1,
                          style={'width': '120px', 'backgroundColor': COLORS['bg'],
                                 'color': COLORS['text'], 'border': f'1px solid {COLORS["border"]}',
                                 'padding': '8px'}),
            ]),
            html.Div([
                html.Label("Strike (K)", style={'fontSize': '10px', 'color': COLORS['green']}),
                dcc.Input(id='bs-K', type='number', value=100, step=1,
                          style={'width': '120px', 'backgroundColor': COLORS['bg'],
                                 'color': COLORS['text'], 'border': f'1px solid {COLORS["border"]}',
                                 'padding': '8px'}),
            ]),
            html.Div([
                html.Label("Taxa (r %)", style={'fontSize': '10px', 'color': COLORS['green']}),
                dcc.Input(id='bs-r', type='number', value=10.75, step=0.25,
                          style={'width': '120px', 'backgroundColor': COLORS['bg'],
                                 'color': COLORS['text'], 'border': f'1px solid {COLORS["border"]}',
                                 'padding': '8px'}),
            ]),
            html.Div([
                html.Label("Volatilidade (σ %)", style={'fontSize': '10px', 'color': COLORS['green']}),
                dcc.Input(id='bs-sigma', type='number', value=30, step=1,
                          style={'width': '120px', 'backgroundColor': COLORS['bg'],
                                 'color': COLORS['text'], 'border': f'1px solid {COLORS["border"]}',
                                 'padding': '8px'}),
            ]),
            html.Div([
                html.Label("Maturidade (dias)", style={'fontSize': '10px', 'color': COLORS['green']}),
                dcc.Input(id='bs-T', type='number', value=90, step=1,
                          style={'width': '120px', 'backgroundColor': COLORS['bg'],
                                 'color': COLORS['text'], 'border': f'1px solid {COLORS["border"]}',
                                 'padding': '8px'}),
            ]),
        ]),
        html.Div(id='bs-output'),
    ]),

    # Store para dados
    dcc.Store(id='data-store'),
])


# ══════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════

@callback(
    Output('data-store', 'data'),
    Input('update-btn', 'n_clicks'),
    State('ticker-select', 'value'),
    State('period-select', 'value'),
)
def update_data(n_clicks, tickers, period):
    """Baixa dados do yfinance."""
    if not tickers:
        return None

    try:
        dados = yf.download(tickers, period=period, auto_adjust=True, progress=False)

        if dados.empty:
            return None

        if isinstance(dados.columns, pd.MultiIndex):
            precos = dados['Close']
        else:
            # Um único ticker
            precos = dados[['Close']]
            precos.columns = tickers if isinstance(tickers, list) else [tickers]

        precos = precos.dropna()
        # Remover colunas que ficaram todas NaN (tickers delisted)
        precos = precos.dropna(axis=1, how='all')

        if precos.empty:
            return None

        precos.index = precos.index.strftime('%Y-%m-%d')
        return precos.to_json()
    except Exception:
        return None


@callback(
    Output('bs-section', 'style'),
    Input('tabs', 'value'),
)
def toggle_bs_section(tab):
    """Mostra/esconde a seção Black-Scholes."""
    if tab == 'tab-bs':
        return {'display': 'block', 'marginTop': '20px'}
    return {'display': 'none', 'marginTop': '20px'}


@callback(
    Output('tab-content', 'children'),
    Output('tab-content', 'style'),
    Input('tabs', 'value'),
    Input('data-store', 'data'),
    State('ticker-select', 'value'),
)
def render_tab(tab, data_json, tickers):
    """Renderiza conteúdo de cada aba."""
    hidden = {'display': 'none', 'marginTop': '20px'}
    visible = {'display': 'block', 'marginTop': '20px'}

    # Black-Scholes usa seção separada (sempre no DOM)
    if tab == 'tab-bs':
        return html.Div(), hidden

    if data_json is None or not tickers:
        return html.P("Selecione ativos e clique em ATUALIZAR",
                       style={'color': COLORS['text_dim'], 'textAlign': 'center',
                              'padding': '50px'}), visible

    try:
        precos = pd.read_json(StringIO(data_json))
        precos.index = pd.to_datetime(precos.index)
    except Exception:
        return html.P("Erro ao carregar dados. Clique em ATUALIZAR novamente.",
                       style={'color': COLORS['red'], 'textAlign': 'center',
                              'padding': '50px'}), visible

    if precos.empty or len(precos) < 2:
        return html.P("Dados insuficientes. Selecione outros ativos ou período.",
                       style={'color': COLORS['amber'], 'textAlign': 'center',
                              'padding': '50px'}), visible

    if tab == 'tab-price':
        return render_price_tab(precos, tickers), visible
    elif tab == 'tab-risk':
        return render_risk_tab(precos, tickers), visible
    elif tab == 'tab-mc':
        return render_mc_tab(precos, tickers), visible

    return html.P("Selecione uma aba"), visible


def render_price_tab(precos, tickers):
    """Aba de Preços & Retornos."""
    log_ret = np.log(precos / precos.shift(1)).dropna()

    # Gráfico de preços normalizados
    precos_norm = precos / precos.iloc[0] * 100
    fig_price = go.Figure()
    for col in precos_norm.columns:
        fig_price.add_trace(go.Scatter(
            x=precos_norm.index, y=precos_norm[col],
            name=col.replace('.SA', ''), mode='lines', line=dict(width=2)))

    fig_price.update_layout(
        title='Preços Normalizados (Base 100)',
        template=TEMPLATE,
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['surface'],
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )

    # Gráfico de retornos acumulados
    cum_ret = (np.exp(log_ret.cumsum()) - 1) * 100
    fig_ret = go.Figure()
    for col in cum_ret.columns:
        fig_ret.add_trace(go.Scatter(
            x=cum_ret.index, y=cum_ret[col],
            name=col.replace('.SA', ''), mode='lines', fill='tonexty' if col == cum_ret.columns[0] else None))

    fig_ret.update_layout(
        title='Retorno Acumulado (%)',
        template=TEMPLATE,
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['surface'],
        height=350,
    )

    # Volatilidade rolling
    vol_rolling = log_ret.rolling(21).std() * np.sqrt(252) * 100
    fig_vol = go.Figure()
    for col in vol_rolling.columns:
        fig_vol.add_trace(go.Scatter(
            x=vol_rolling.index, y=vol_rolling[col],
            name=col.replace('.SA', ''), mode='lines'))

    fig_vol.update_layout(
        title='Volatilidade Rolling 21 dias (Anualizada %)',
        template=TEMPLATE,
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['surface'],
        height=300,
    )

    # Heatmap correlação
    corr = log_ret.corr()
    labels = [c.replace('.SA', '') for c in corr.columns]
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale=[[0, COLORS['red']], [0.5, COLORS['bg']], [1, COLORS['green']]],
        text=np.round(corr.values, 2), texttemplate='%{text}', textfont={'size': 12}))

    fig_corr.update_layout(
        title='Matriz de Correlação',
        template=TEMPLATE,
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['surface'],
        height=400,
    )

    # Tabela de estatísticas
    stats_data = []
    for col in log_ret.columns:
        r = log_ret[col]
        stats_data.append({
            'Ativo': col.replace('.SA', ''),
            'Retorno Anual': f"{r.mean()*252*100:.1f}%",
            'Volatilidade': f"{r.std()*np.sqrt(252)*100:.1f}%",
            'Sharpe': f"{(r.mean()*252 - 0.1075)/(r.std()*np.sqrt(252)):.2f}",
            'Max DD': f"{((1+r).cumprod() / (1+r).cumprod().cummax() - 1).min()*100:.1f}%",
            'Skew': f"{r.skew():.2f}",
            'Curtose': f"{r.kurtosis():.2f}",
        })

    return html.Div([
        dcc.Graph(figure=fig_price),
        html.Div(style={'display': 'flex', 'gap': '20px'}, children=[
            html.Div(dcc.Graph(figure=fig_ret), style={'flex': '1'}),
            html.Div(dcc.Graph(figure=fig_corr), style={'flex': '1'}),
        ]),
        dcc.Graph(figure=fig_vol),

        # Tabela
        html.Div(style={
            'backgroundColor': COLORS['surface'],
            'border': f'1px solid {COLORS["border"]}',
            'padding': '20px', 'marginTop': '20px'
        }, children=[
            html.H3("ESTATÍSTICAS", style={'color': COLORS['green'],
                     'fontSize': '12px', 'letterSpacing': '3px'}),
            html.Table(style={'width': '100%', 'borderCollapse': 'collapse'}, children=[
                html.Thead(html.Tr([
                    html.Th(h, style={
                        'padding': '10px', 'textAlign': 'left',
                        'borderBottom': f'1px solid {COLORS["border"]}',
                        'color': COLORS['text_dim'], 'fontSize': '10px',
                        'letterSpacing': '1px'})
                    for h in stats_data[0].keys()
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(v, style={
                            'padding': '8px', 'fontSize': '13px',
                            'borderBottom': f'1px solid {COLORS["border"]}30'})
                        for v in row.values()
                    ]) for row in stats_data
                ])
            ])
        ]),
    ])


def render_risk_tab(precos, tickers):
    """Aba de Risco — VaR/CVaR."""
    log_ret = np.log(precos / precos.shift(1)).dropna()

    # Usar primeiro ativo
    col = log_ret.columns[0]
    r = log_ret[col]
    nome = col.replace('.SA', '')

    # VaR & CVaR
    niveis = [0.90, 0.95, 0.99]
    var_hist = {n: np.percentile(r, (1-n)*100) for n in niveis}
    cvar_hist = {n: r[r <= var_hist[n]].mean() for n in niveis}

    mu, sigma = r.mean(), r.std()
    var_param = {n: mu + norm.ppf(1-n) * sigma for n in niveis}

    # Gráfico de distribuição com VaR
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=r, nbinsx=100, name='Retornos',
        marker_color=COLORS['blue'], opacity=0.6))

    for nivel, cor in zip(niveis, [COLORS['green'], COLORS['amber'], COLORS['red']]):
        fig_dist.add_vline(x=var_hist[nivel], line_dash='dash',
                          line_color=cor, annotation_text=f'VaR {nivel:.0%}: {var_hist[nivel]:.4f}')

    fig_dist.update_layout(
        title=f'Distribuição de Retornos & VaR — {nome}',
        template=TEMPLATE,
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['surface'],
        height=400,
    )

    # Rolling VaR
    window = 252
    rolling_var = r.rolling(window).quantile(0.05)
    rolling_cvar = r.rolling(window).apply(lambda x: x[x <= x.quantile(0.05)].mean())

    fig_rolling = go.Figure()
    fig_rolling.add_trace(go.Scatter(
        x=r.index, y=r, name='Retornos', mode='lines',
        line=dict(color=COLORS['blue'], width=0.5), opacity=0.5))
    fig_rolling.add_trace(go.Scatter(
        x=rolling_var.index, y=rolling_var, name='VaR 95% (252d)',
        line=dict(color=COLORS['red'], width=2)))
    fig_rolling.add_trace(go.Scatter(
        x=rolling_cvar.index, y=rolling_cvar, name='CVaR 95% (252d)',
        line=dict(color=COLORS['amber'], width=2, dash='dash')))

    fig_rolling.update_layout(
        title=f'VaR & CVaR Rolling (252 dias) — {nome}',
        template=TEMPLATE,
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['surface'],
        height=350,
    )

    # Tabela VaR
    investimento = 1_000_000
    table_rows = []
    for nivel in niveis:
        table_rows.append(html.Tr([
            html.Td(f'{nivel:.0%}'),
            html.Td(f'{var_hist[nivel]:.4f}', style={'color': COLORS['red']}),
            html.Td(f'{var_param[nivel]:.4f}', style={'color': COLORS['amber']}),
            html.Td(f'{cvar_hist[nivel]:.4f}', style={'color': COLORS['red']}),
            html.Td(f'R$ {-var_hist[nivel]*investimento:,.0f}'),
        ], style={'borderBottom': f'1px solid {COLORS["border"]}30'}))

    return html.Div([
        dcc.Graph(figure=fig_dist),
        dcc.Graph(figure=fig_rolling),
        html.Div(style={
            'backgroundColor': COLORS['surface'],
            'border': f'1px solid {COLORS["border"]}',
            'padding': '20px', 'marginTop': '20px'
        }, children=[
            html.H3(f"VaR / CVaR — {nome} (investimento R$ 1.000.000)",
                     style={'color': COLORS['green'], 'fontSize': '12px',
                            'letterSpacing': '2px', 'marginBottom': '15px'}),
            html.Table(style={'width': '100%', 'borderCollapse': 'collapse'}, children=[
                html.Thead(html.Tr([
                    html.Th(h, style={
                        'padding': '10px', 'textAlign': 'left',
                        'borderBottom': f'1px solid {COLORS["border"]}',
                        'color': COLORS['text_dim'], 'fontSize': '10px'})
                    for h in ['Nível', 'VaR Histórico', 'VaR Paramétrico',
                              'CVaR Histórico', 'Perda Máxima (R$)']
                ])),
                html.Tbody(table_rows)
            ])
        ]),
    ])


def render_mc_tab(precos, tickers):
    """Aba Monte Carlo."""
    col = precos.columns[0]
    nome = col.replace('.SA', '')
    log_ret = np.log(precos[col] / precos[col].shift(1)).dropna()

    mu = log_ret.mean()
    sigma = log_ret.std()
    S0 = precos[col].iloc[-1]

    np.random.seed(42)
    n_paths = 500
    n_days = 252
    dt = 1/252

    Z = np.random.standard_normal((n_paths, n_days))
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    paths = S0 * np.exp(np.cumsum(log_returns, axis=1))
    paths = np.column_stack([np.full(n_paths, S0), paths])

    days = list(range(n_days + 1))

    fig_mc = go.Figure()

    # Caminhos individuais (sample)
    for i in range(min(100, n_paths)):
        fig_mc.add_trace(go.Scatter(
            x=days, y=paths[i], mode='lines',
            line=dict(color=COLORS['cyan'], width=0.3),
            opacity=0.3, showlegend=False))

    # Percentis
    p5 = np.percentile(paths, 5, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    fig_mc.add_trace(go.Scatter(
        x=days, y=p95, mode='lines', name='P95',
        line=dict(color=COLORS['green'], width=0, dash='dash')))
    fig_mc.add_trace(go.Scatter(
        x=days, y=p5, mode='lines', name='Intervalo 90%',
        fill='tonexty', fillcolor=f'rgba(0,212,160,0.15)',
        line=dict(color=COLORS['green'], width=0)))
    fig_mc.add_trace(go.Scatter(
        x=days, y=p50, mode='lines', name='Mediana',
        line=dict(color=COLORS['amber'], width=3)))

    fig_mc.update_layout(
        title=f'Simulação Monte Carlo (GBM) — {nome} | {n_paths} caminhos | 1 ano',
        xaxis_title='Dias',
        yaxis_title='Preço (R$)',
        template=TEMPLATE,
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['surface'],
        height=500,
    )

    # Distribuição do preço final
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=paths[:, -1], nbinsx=50, name='Preço Final',
        marker_color=COLORS['blue'], opacity=0.7))

    fig_hist.add_vline(x=S0, line_dash='dash', line_color=COLORS['amber'],
                      annotation_text=f'Preço Atual: R${S0:.2f}')
    fig_hist.add_vline(x=np.mean(paths[:, -1]), line_dash='dash',
                      line_color=COLORS['green'],
                      annotation_text=f'Média: R${np.mean(paths[:, -1]):.2f}')

    fig_hist.update_layout(
        title='Distribuição do Preço Final (1 ano)',
        template=TEMPLATE,
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['surface'],
        height=350,
    )

    # Stats
    retornos_mc = (paths[:, -1] / S0 - 1) * 100
    stats_mc = {
        'Preço Atual': f'R$ {S0:.2f}',
        'Média (1 ano)': f'R$ {np.mean(paths[:, -1]):.2f}',
        'Mediana (1 ano)': f'R$ {np.median(paths[:, -1]):.2f}',
        'P5 (pessimista)': f'R$ {np.percentile(paths[:, -1], 5):.2f}',
        'P95 (otimista)': f'R$ {np.percentile(paths[:, -1], 95):.2f}',
        'Retorno Médio': f'{np.mean(retornos_mc):.1f}%',
        'Prob. Lucro': f'{(retornos_mc > 0).mean()*100:.1f}%',
    }

    return html.Div([
        dcc.Graph(figure=fig_mc),
        html.Div(style={'display': 'flex', 'gap': '20px'}, children=[
            html.Div(dcc.Graph(figure=fig_hist), style={'flex': '2'}),
            html.Div(style={
                'flex': '1', 'backgroundColor': COLORS['surface'],
                'border': f'1px solid {COLORS["border"]}',
                'padding': '20px',
            }, children=[
                html.H3("ESTATÍSTICAS MC", style={
                    'color': COLORS['green'], 'fontSize': '11px',
                    'letterSpacing': '2px', 'marginBottom': '15px'}),
                *[html.Div([
                    html.Span(k, style={'color': COLORS['text_dim'], 'fontSize': '11px'}),
                    html.Br(),
                    html.Span(v, style={'color': COLORS['text'], 'fontSize': '16px',
                                         'fontWeight': 'bold'}),
                    html.Hr(style={'borderColor': COLORS['border'], 'margin': '8px 0'}),
                ]) for k, v in stats_mc.items()]
            ]),
        ]),
    ])


@callback(
    Output('bs-output', 'children'),
    Input('bs-S', 'value'),
    Input('bs-K', 'value'),
    Input('bs-r', 'value'),
    Input('bs-sigma', 'value'),
    Input('bs-T', 'value'),
)
def update_bs(S, K, r_pct, sigma_pct, T_days):
    """Atualiza calculadora Black-Scholes."""
    try:
        S = float(S or 100)
        K = float(K or 100)
        r_pct = float(r_pct or 10.75)
        sigma_pct = float(sigma_pct or 30)
        T_days = float(T_days or 90)

        if any(v <= 0 for v in [S, K, sigma_pct, T_days]):
            return html.P("Insira valores positivos válidos", style={'color': COLORS['red']})
    except (TypeError, ValueError):
        return html.P("Insira valores numéricos válidos", style={'color': COLORS['red']})

    r = r_pct / 100
    sigma = sigma_pct / 100
    T = T_days / 365

    c_price, c_delta, c_gamma, c_theta, c_vega = black_scholes(S, K, r, sigma, T, 'call')
    p_price, p_delta, p_gamma, p_theta, p_vega = black_scholes(S, K, r, sigma, T, 'put')

    # Gráfico de payoff
    S_range = np.linspace(S * 0.6, S * 1.4, 200)
    call_payoff = np.maximum(S_range - K, 0) - c_price
    put_payoff = np.maximum(K - S_range, 0) - p_price

    fig_payoff = go.Figure()
    fig_payoff.add_trace(go.Scatter(
        x=S_range.tolist(), y=call_payoff.tolist(), name=f'Call (${c_price:.2f})',
        line=dict(color=COLORS['green'], width=2.5)))
    fig_payoff.add_trace(go.Scatter(
        x=S_range.tolist(), y=put_payoff.tolist(), name=f'Put (${p_price:.2f})',
        line=dict(color=COLORS['red'], width=2.5)))
    fig_payoff.add_hline(y=0, line_dash='dash', line_color=COLORS['text_dim'])
    fig_payoff.add_vline(x=K, line_dash='dash', line_color=COLORS['amber'],
                        annotation_text=f'K={K}')

    fig_payoff.update_layout(
        title='Payoff na Expiração',
        xaxis_title='Preço do Ativo (S_T)',
        yaxis_title='Lucro/Prejuízo ($)',
        template=TEMPLATE,
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['surface'],
        height=350,
    )

    # Greeks cards
    def greek_card(name, call_val, put_val, color):
        return html.Div(style={
            'backgroundColor': COLORS['surface'],
            'border': f'1px solid {COLORS["border"]}',
            'padding': '15px', 'textAlign': 'center', 'flex': '1',
        }, children=[
            html.Div(name, style={'color': color, 'fontSize': '10px',
                                   'letterSpacing': '2px', 'marginBottom': '8px'}),
            html.Div(f'C: {call_val:+.4f}', style={'fontSize': '14px', 'color': COLORS['green']}),
            html.Div(f'P: {put_val:+.4f}', style={'fontSize': '14px', 'color': COLORS['red']}),
        ])

    return html.Div([
        # Preços
        html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}, children=[
            html.Div(style={
                'flex': '1', 'backgroundColor': COLORS['surface'],
                'border': f'2px solid {COLORS["green"]}',
                'padding': '20px', 'textAlign': 'center',
            }, children=[
                html.Div("CALL", style={'color': COLORS['green'], 'fontSize': '11px',
                                         'letterSpacing': '3px'}),
                html.Div(f'${c_price:.4f}', style={'fontSize': '28px',
                                                      'fontWeight': 'bold',
                                                      'color': COLORS['green']}),
            ]),
            html.Div(style={
                'flex': '1', 'backgroundColor': COLORS['surface'],
                'border': f'2px solid {COLORS["red"]}',
                'padding': '20px', 'textAlign': 'center',
            }, children=[
                html.Div("PUT", style={'color': COLORS['red'], 'fontSize': '11px',
                                        'letterSpacing': '3px'}),
                html.Div(f'${p_price:.4f}', style={'fontSize': '28px',
                                                     'fontWeight': 'bold',
                                                     'color': COLORS['red']}),
            ]),
        ]),

        # Greeks
        html.Div(style={'display': 'flex', 'gap': '10px', 'marginBottom': '20px'}, children=[
            greek_card('DELTA (Δ)', c_delta, p_delta, COLORS['green']),
            greek_card('GAMMA (Γ)', c_gamma, p_gamma, COLORS['cyan']),
            greek_card('THETA (Θ)', c_theta, p_theta, COLORS['amber']),
            greek_card('VEGA (ν)', c_vega, p_vega, COLORS['purple']),
        ]),

        dcc.Graph(figure=fig_payoff),
    ])


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("  QUANT ACADEMY — DASHBOARD INTERATIVO")
    print("  Acesse: http://localhost:8050")
    print("=" * 60)
    app.run(debug=False, port=8050)
