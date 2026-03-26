"""
=============================================================================
MÓDULO 10C — LSTM para Previsão de Séries Temporais Financeiras
Quant Academy · Finanças Quantitativas
=============================================================================
Redes neurais recorrentes (LSTM) para previsão de preços de ações.
Feature engineering, normalização, sequências temporais e avaliação.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')
COLORS = {
    'green': '#00d4a0', 'amber': '#f5a623',
    'blue': '#4a9eff', 'red': '#e05252',
    'bg': '#080c10', 'surface': '#0d1520',
    'text': '#c8d8e8', 'dim': '#6a8aa8'
}
SAVE = dict(dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])


def calcular_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def carregar_dados():
    import yfinance as yf
    print("📥 Baixando dados da PETR4.SA (3 anos)...")
    df = yf.download('PETR4.SA', period='3y', auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Close', 'Volume']].dropna()
    print(f"   ✔ {len(df)} observações")
    return df


def criar_features(df):
    """Cria features para a LSTM."""
    data = pd.DataFrame(index=df.index)
    data['close'] = df['Close']
    data['volume'] = df['Volume']
    data['returns'] = df['Close'].pct_change()
    data['ma5'] = df['Close'].rolling(5).mean()
    data['ma20'] = df['Close'].rolling(20).mean()
    data['rsi'] = calcular_rsi(df['Close'])
    data['vol_20'] = df['Close'].pct_change().rolling(20).std()
    return data.dropna()


def criar_sequencias(X, y, lookback=60):
    """Cria janelas deslizantes para a LSTM. Shape: (samples, timesteps, features)."""
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def main():
    print("=" * 70)
    print("  MÓDULO 10C · LSTM PARA SÉRIES TEMPORAIS FINANCEIRAS")
    print("=" * 70)

    # Verificar se TensorFlow está disponível
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        print(f"   ✔ TensorFlow {tf.__version__} disponível")
    except ImportError:
        print("\n   ⚠ TensorFlow não está instalado!")
        print("   Execute: pip install tensorflow")
        print("   Gerando gráficos com dados simulados...\n")
        _gerar_graficos_simulados()
        return

    df = carregar_dados()
    data = criar_features(df)

    # ── Preparação dos dados ──────────────────────────────────────────────
    LOOKBACK = 60
    feature_cols = ['close', 'volume', 'returns', 'ma5', 'ma20', 'rsi', 'vol_20']

    values = data[feature_cols].values
    target_col_idx = 0  # 'close' é o target

    # Split temporal 80/20
    split = int(len(values) * 0.8)
    train_data = values[:split]
    test_data = values[split:]

    # Normalizar (fit no treino apenas!)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    # Scaler separado para o target (para inverse_transform)
    target_scaler = MinMaxScaler()
    target_scaler.fit(train_data[:, target_col_idx].reshape(-1, 1))

    # Criar sequências
    all_scaled = np.vstack([train_scaled, test_scaled])
    X_all, y_all = criar_sequencias(all_scaled, all_scaled[:, target_col_idx], LOOKBACK)

    # Split novamente após criar sequências
    train_samples = split - LOOKBACK
    X_train = X_all[:train_samples]
    y_train = y_all[:train_samples]
    X_test = X_all[train_samples:]
    y_test = y_all[train_samples:]

    print(f"\n📊 Dados:")
    print(f"   Lookback: {LOOKBACK} dias")
    print(f"   Features: {len(feature_cols)}")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test:  {X_test.shape}")

    # ── Modelo LSTM ──────────────────────────────────────────────────────
    print("\n🧠 Construindo modelo LSTM...")
    model = Sequential([
        LSTM(128, return_sequences=True,
             input_shape=(LOOKBACK, len(feature_cols))),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # ── Treinamento ──────────────────────────────────────────────────────
    print("\n🏋️ Treinando modelo...")
    early_stop = EarlyStopping(monitor='val_loss', patience=10,
                                restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=100, batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    # ── Previsão ──────────────────────────────────────────────────────────
    print("\n📈 Fazendo previsões...")
    train_pred = model.predict(X_train, verbose=0).flatten()
    test_pred = model.predict(X_test, verbose=0).flatten()

    # Inverse transform
    train_pred_real = target_scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
    test_pred_real = target_scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
    y_train_real = target_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test_real = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Métricas
    rmse_train = np.sqrt(mean_squared_error(y_train_real, train_pred_real))
    rmse_test = np.sqrt(mean_squared_error(y_test_real, test_pred_real))
    mae_test = mean_absolute_error(y_test_real, test_pred_real)
    mape_test = np.mean(np.abs((y_test_real - test_pred_real) / y_test_real)) * 100

    print(f"\n   📊 Métricas:")
    print(f"      RMSE Treino: R$ {rmse_train:.2f}")
    print(f"      RMSE Teste:  R$ {rmse_test:.2f}")
    print(f"      MAE Teste:   R$ {mae_test:.2f}")
    print(f"      MAPE Teste:  {mape_test:.2f}%")

    # ── Gráficos ─────────────────────────────────────────────────────────
    print("\n🎨 Gerando gráficos...")

    # 1. Training Loss
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['surface'])
    ax.plot(history.history['loss'], color=COLORS['green'], linewidth=1.5, label='Treino')
    ax.plot(history.history['val_loss'], color=COLORS['amber'], linewidth=1.5, label='Validação')
    ax.set_title('LOSS DE TREINAMENTO · LSTM', fontsize=14,
                 color=COLORS['text'], fontweight='bold', pad=15)
    ax.set_xlabel('Época', color=COLORS['dim'])
    ax.set_ylabel('MSE Loss', color=COLORS['dim'])
    ax.legend(fontsize=10)
    ax.tick_params(colors=COLORS['dim'])
    plt.tight_layout()
    plt.savefig('../graficos/m10c_lstm_training_loss.png', **SAVE)
    plt.close()
    print("   ✔ graficos/m10c_lstm_training_loss.png")

    # 2. Previsão completa
    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['surface'])

    # Índices
    dates = data.index[LOOKBACK:]
    train_dates = dates[:train_samples]
    test_dates = dates[train_samples:train_samples + len(test_pred_real)]

    ax.plot(train_dates, y_train_real, color=COLORS['dim'], alpha=0.5,
            linewidth=0.8, label='Real (Treino)')
    ax.plot(test_dates, y_test_real, color=COLORS['text'], linewidth=1.5,
            label='Real (Teste)')
    ax.plot(test_dates, test_pred_real, color=COLORS['green'], linewidth=1.5,
            label=f'LSTM (RMSE: R$ {rmse_test:.2f})')

    ax.axvline(test_dates[0], color=COLORS['amber'], linestyle='--', alpha=0.7,
               label='Início Teste')

    ax.set_title('PREVISÃO LSTM · PETR4.SA', fontsize=14,
                 color=COLORS['text'], fontweight='bold', pad=15)
    ax.set_xlabel('Data', color=COLORS['dim'])
    ax.set_ylabel('Preço (R$)', color=COLORS['dim'])
    ax.legend(fontsize=10)
    ax.tick_params(colors=COLORS['dim'])
    plt.tight_layout()
    plt.savefig('../graficos/m10c_lstm_previsao.png', **SAVE)
    plt.close()
    print("   ✔ graficos/m10c_lstm_previsao.png")

    # 3. Zoom nos últimos 60 dias
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['surface'])

    n_zoom = min(60, len(test_pred_real))
    ax.plot(test_dates[-n_zoom:], y_test_real[-n_zoom:],
            color=COLORS['text'], linewidth=2, marker='o', markersize=3,
            label='Real')
    ax.plot(test_dates[-n_zoom:], test_pred_real[-n_zoom:],
            color=COLORS['green'], linewidth=2, marker='s', markersize=3,
            label='LSTM')

    ax.fill_between(test_dates[-n_zoom:],
                    test_pred_real[-n_zoom:] * 0.97,
                    test_pred_real[-n_zoom:] * 1.03,
                    alpha=0.15, color=COLORS['green'])

    ax.set_title(f'ZOOM · ÚLTIMOS {n_zoom} DIAS DE PREVISÃO',
                 fontsize=14, color=COLORS['text'], fontweight='bold', pad=15)
    ax.set_xlabel('Data', color=COLORS['dim'])
    ax.set_ylabel('Preço (R$)', color=COLORS['dim'])
    ax.legend(fontsize=10)
    ax.tick_params(colors=COLORS['dim'])
    plt.tight_layout()
    plt.savefig('../graficos/m10c_lstm_zoom.png', **SAVE)
    plt.close()
    print("   ✔ graficos/m10c_lstm_zoom.png")

    print(f"\n{'=' * 70}")
    print("  ✅ Módulo 10C concluído!")
    print(f"{'=' * 70}")


def _gerar_graficos_simulados():
    """Gera gráficos demonstrativos quando TF não está disponível."""
    np.random.seed(42)

    # Simular dados de treinamento
    epochs = 50
    train_loss = 0.01 * np.exp(-np.linspace(0, 3, epochs)) + 0.001 + np.random.normal(0, 0.0005, epochs)
    val_loss = 0.012 * np.exp(-np.linspace(0, 2.5, epochs)) + 0.0015 + np.random.normal(0, 0.0008, epochs)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['surface'])
    ax.plot(train_loss, color=COLORS['green'], linewidth=1.5, label='Treino')
    ax.plot(val_loss, color=COLORS['amber'], linewidth=1.5, label='Validação')
    ax.set_title('LOSS DE TREINAMENTO · LSTM (SIMULADO)', fontsize=14,
                 color=COLORS['text'], fontweight='bold', pad=15)
    ax.set_xlabel('Época', color=COLORS['dim'])
    ax.set_ylabel('MSE Loss', color=COLORS['dim'])
    ax.legend(fontsize=10)
    ax.tick_params(colors=COLORS['dim'])
    plt.tight_layout()
    plt.savefig('../graficos/m10c_lstm_training_loss.png', **SAVE)
    plt.close()

    # Simular previsão
    t = np.arange(200)
    real = 35 + np.cumsum(np.random.normal(0.01, 0.5, 200))
    pred = real + np.random.normal(0, 0.8, 200)

    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['surface'])
    ax.plot(t, real, color=COLORS['text'], linewidth=1.5, label='Real')
    ax.plot(t, pred, color=COLORS['green'], linewidth=1.5, label='LSTM (simulado)')
    ax.set_title('PREVISÃO LSTM · SIMULADO (TF não instalado)',
                 fontsize=14, color=COLORS['text'], fontweight='bold', pad=15)
    ax.legend(fontsize=10)
    ax.tick_params(colors=COLORS['dim'])
    plt.tight_layout()
    plt.savefig('../graficos/m10c_lstm_previsao.png', **SAVE)
    plt.close()

    print("   ✔ Gráficos simulados gerados (instale TensorFlow para dados reais)")


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
