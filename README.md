# 📊 Quant Academy — Finanças Quantitativas com Python

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-00d4a0?style=flat-square)
![Modules](https://img.shields.io/badge/Módulos-12-f5a623?style=flat-square)
![ML Models](https://img.shields.io/badge/ML_Models-8+-e05252?style=flat-square)

**Curso completo de Finanças Quantitativas: da teoria matemática à implementação prática com dados reais do mercado brasileiro.**

</div>

---

## 🏗 Estrutura do Projeto

```
quantitative-finance/
│
├── curso_quant_financas.html      # Curso interativo (abrir no navegador)
├── requirements.txt               # Dependências Python
├── README.md                      # Este arquivo
│
├── src/                           # Código-fonte (12 módulos + dashboard)
│   ├── modulo01_retornos_financeiros.py
│   ├── modulo02_estatistica_financeira.py
│   ├── modulo03_probabilidade_distribuicoes.py
│   ├── modulo04_series_temporais.py
│   ├── modulo05_arima_sarima.py
│   ├── modulo06_garch_volatilidade.py
│   ├── modulo07_previsao_backtesting.py
│   ├── modulo08_var_cvar.py
│   ├── modulo09_monte_carlo.py
│   ├── modulo10a_random_forest_xgboost.py
│   ├── modulo10b_ridge_lasso_regression.py
│   ├── modulo10c_lstm_previsao.py
│   ├── modulo10d_pca_clustering.py
│   ├── modulo10e_svm_knn.py
│   ├── modulo11_portfolio_quantitativo.py
│   ├── modulo12_derivativos_opcoes.py
│   └── dashboard_interativo.py
│
├── data/                          # Dados gerados (CSV)
│   ├── precos_acoes_br.csv
│   ├── estatisticas_retornos.csv
│   ├── matriz_correlacao.csv
│   └── ...
│
└── graficos/                      # Gráficos gerados (PNG)
    ├── m01_precos_normalizados.png
    ├── m02_heatmap_correlacao.png
    ├── m06_garch_analise.png
    ├── m12_greeks.png
    └── ...
```

---

## 📚 Módulos

### Bloco 1 — Fundamentos (Módulos 01-03)

| Módulo | Tema | Conteúdo |
|--------|------|----------|
| **01** | Retornos Financeiros | Retorno simples vs log-return, retorno acumulado, volatilidade, Sharpe Ratio |
| **02** | Estatística Financeira | 4 momentos, covariância/correlação, GBM (Geometric Brownian Motion), risco de portfólio |
| **03** | Probabilidade & Distribuições | Normal, t-Student, Log-Normal, TLC, VaR paramétrico, QQ-Plot |

### Bloco 2 — Séries Temporais (Módulos 04-06)

| Módulo | Tema | Conteúdo |
|--------|------|----------|
| **04** | Séries Temporais | Estacionariedade, ADF test, ACF/PACF, decomposição, rolling stats |
| **05** | ARIMA & SARIMA | Identificação de ordem, fitting, diagnóstico de resíduos, previsão |
| **06** | GARCH & Volatilidade | Volatility clustering, GARCH(1,1), EGARCH, GJR-GARCH, previsão de vol |

### Bloco 3 — Projeções & Risco (Módulos 07-09)

| Módulo | Tema | Conteúdo |
|--------|------|----------|
| **07** | Previsão & Backtesting | Walk-forward, expanding window, métricas de previsão, sinais de trading |
| **08** | VaR & CVaR | VaR histórico/paramétrico/Monte Carlo, CVaR, backtesting de VaR |
| **09** | Monte Carlo | Simulação GBM, precificação de opções, análise de cenários, convergência |

### Bloco 4 — Avançado (Módulos 10-12)

| Módulo | Tema | Conteúdo |
|--------|------|----------|
| **10a** | Random Forest & XGBoost | Classificação de direção de mercado, feature importance, tuning |
| **10b** | Ridge & Lasso Regression | Regularização, seleção de features, cross-validation |
| **10c** | LSTM (Deep Learning) | Previsão de séries temporais com redes neurais recorrentes |
| **10d** | PCA & Clustering | Redução de dimensionalidade, K-Means, regime detection |
| **10e** | SVM & KNN | Classificação não-linear, kernel trick, validação |
| **11** | Gestão de Portfólio | Markowitz, fronteira eficiente, Black-Litterman, Sharpe/Sortino/Calmar |
| **12** | Derivativos & Opções | Black-Scholes, Greeks, estratégias (Straddle, Butterfly, Iron Condor) |

### Dashboard Interativo

Dashboard completo em **Plotly Dash** com:
- Análise de preços em tempo real
- Retornos e volatilidade rolling
- VaR/CVaR dinâmico
- Simulação Monte Carlo
- Calculadora Black-Scholes interativa

---

## 🚀 Instalação & Uso

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Executar módulos individuais

```bash
# Cada módulo é independente — baixa dados reais e gera gráficos
python src/modulo01_retornos_financeiros.py
python src/modulo02_estatistica_financeira.py
python src/modulo06_garch_volatilidade.py
python src/modulo10a_random_forest_xgboost.py
python src/modulo12_derivativos_opcoes.py
```

### 3. Executar o Dashboard

```bash
python src/dashboard_interativo.py
# Acesse: http://localhost:8050
```

---

## 📈 Dados Reais Utilizados

Todos os módulos usam **dados reais** do mercado brasileiro via `yfinance`:

| Ativo | Empresa | Setor |
|-------|---------|-------|
| PETR4 | Petrobras | Petróleo & Gás |
| VALE3 | Vale | Mineração |
| ITUB4 | Itaú Unibanco | Bancos |
| BBDC4 | Bradesco | Bancos |
| ABEV3 | Ambev | Bebidas |
| WEGE3 | WEG | Industrial |
| RENT3 | Localiza | Locação |
| BBAS3 | Banco do Brasil | Bancos |
| SUZB3 | Suzano | Papel & Celulose |
| JBSS3 | JBS | Alimentos |
| ^BVSP | IBOVESPA | Índice |

---

## 🛠 Stack Tecnológica

- **Dados**: `yfinance`, `pandas`, `numpy`
- **Estatística**: `scipy`, `statsmodels`, `arch`
- **Séries Temporais**: `statsmodels`, `pmdarima`
- **Machine Learning**: `scikit-learn`, `xgboost`, `tensorflow/keras`
- **Visualização**: `matplotlib`, `plotly`, `dash`
- **Finanças**: `scipy.optimize` (otimização de portfólio), `scipy.stats` (Black-Scholes)

---

## 📊 Exemplos de Gráficos

Cada módulo gera múltiplos gráficos com tema escuro profissional:

- Fronteira Eficiente de Markowitz
- Heatmaps de correlação
- Superfície de volatilidade implícita
- Simulações Monte Carlo (GBM)
- Volatilidade condicional GARCH
- Greeks de opções
- Confusion matrices e ROC curves (ML)
- Decomposição de séries temporais

---

## 📖 Referências

- **Markowitz, H.** (1952). Portfolio Selection. *Journal of Finance*
- **Black, F. & Scholes, M.** (1973). The Pricing of Options and Corporate Liabilities
- **Engle, R.** (1982). Autoregressive Conditional Heteroscedasticity (ARCH)
- **Bollerslev, T.** (1986). Generalized ARCH (GARCH)
- **Black, F. & Litterman, R.** (1992). Global Portfolio Optimization
- **Hull, J.** — Options, Futures, and Other Derivatives
- **Tsay, R.** — Analysis of Financial Time Series

---

<div align="center">

*Desenvolvido como material educacional com ajuda do Claude para meu estudo*

</div>
