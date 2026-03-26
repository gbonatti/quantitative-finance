"""
=============================================================================
MÓDULO 10D — PCA & Clustering de Ativos
Quant Academy · Finanças Quantitativas
=============================================================================
Aprendizado não-supervisionado: PCA para redução de dimensionalidade,
K-Means e Clustering Hierárquico para agrupamento de ações brasileiras.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')
COLORS = {
    'green': '#00d4a0', 'amber': '#f5a623',
    'blue': '#4a9eff', 'red': '#e05252',
    'purple': '#b388ff', 'pink': '#ff6b9d',
    'bg': '#080c10', 'surface': '#0d1520',
    'text': '#c8d8e8', 'dim': '#6a8aa8'
}
SAVE = dict(dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
CLUSTER_COLORS = ['#00d4a0', '#4a9eff', '#f5a623', '#e05252', '#b388ff']


def carregar_dados():
    import yfinance as yf

    tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA',
               'WEGE3.SA', 'ABEV3.SA', 'RENT3.SA', 'SUZB3.SA', 'JBSS3.SA',
               'MGLU3.SA', 'LREN3.SA', 'RADL3.SA', 'GGBR4.SA', 'CSNA3.SA']

    print(f"📥 Baixando dados de {len(tickers)} ações (2 anos)...")
    dados = yf.download(tickers, period='2y', auto_adjust=True, progress=False)
    precos = dados['Close'].dropna(axis=1, how='all').dropna()

    # Limpar nomes
    nomes_limpos = [t.replace('.SA', '') for t in precos.columns]
    precos.columns = nomes_limpos

    retornos = np.log(precos / precos.shift(1)).dropna()
    print(f"   ✔ {len(retornos)} dias, {retornos.shape[1]} ativos")
    print(f"   ✔ Ativos: {', '.join(nomes_limpos)}")

    return precos, retornos


def analise_pca(retornos):
    """PCA na matriz de retornos."""
    print("\n📊 Análise PCA...")

    scaler = StandardScaler()
    ret_scaled = scaler.fit_transform(retornos)

    pca_full = PCA()
    pca_full.fit(ret_scaled)

    var_explicada = pca_full.explained_variance_ratio_
    var_acumulada = np.cumsum(var_explicada)

    print(f"   PC1: {var_explicada[0]:.2%} da variância")
    print(f"   PC2: {var_explicada[1]:.2%} da variância")
    print(f"   PC1+PC2: {var_acumulada[1]:.2%} da variância")

    n_95 = np.argmax(var_acumulada >= 0.95) + 1
    print(f"   Componentes para 95%: {n_95}")

    # PCA com 2 componentes para visualização
    pca_2d = PCA(n_components=2)
    coords_2d_dias = pca_2d.fit_transform(ret_scaled)

    # PCA na transposta para projetar ATIVOS em 2D (para gráfico de clusters)
    pca_ativos = PCA(n_components=2)
    coords_2d_ativos = pca_ativos.fit_transform(scaler.fit_transform(retornos.T))

    return pca_full, pca_2d, coords_2d_ativos, var_explicada, var_acumulada


def analise_kmeans(retornos, coords_2d):
    """K-Means clustering com método do cotovelo."""
    print("\n📊 K-Means Clustering...")

    scaler = StandardScaler()
    # Transpor: cada ATIVO vira uma linha (features = dias de retorno)
    ret_scaled = scaler.fit_transform(retornos.T)

    # Método do cotovelo
    inertias = []
    K_range = range(2, 8)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(ret_scaled)
        inertias.append(km.inertia_)

    # Treinar com k=4 (tipicamente bom para mercado BR)
    k_final = 4
    km_final = KMeans(n_clusters=k_final, random_state=42, n_init=10)
    labels = km_final.fit_predict(ret_scaled)

    for i in range(k_final):
        ativos_cluster = retornos.columns[labels == i].tolist()
        print(f"   Cluster {i+1}: {', '.join(ativos_cluster)}")

    return labels, inertias, K_range


def analise_hierarquica(retornos):
    """Clustering hierárquico com dendrograma."""
    print("\n📊 Clustering Hierárquico...")

    corr = retornos.corr()
    dist = np.sqrt(2 * (1 - corr))  # Distância de correlação

    # Converter para formato condensado
    dist_condensed = squareform(dist.values, checks=False)
    linkage_matrix = linkage(dist_condensed, method='ward')

    return linkage_matrix, corr


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════
def grafico_pca(var_explicada, var_acumulada, coords_2d, nomes, labels):
    """Scree plot + PCA 2D com clusters."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(COLORS['bg'])

    # ── Scree Plot ──
    ax = axes[0]
    ax.set_facecolor(COLORS['surface'])

    n = min(len(var_explicada), 15)
    x = range(1, n + 1)

    ax.bar(x, var_explicada[:n] * 100, color=COLORS['blue'], alpha=0.7)
    ax.plot(x, var_acumulada[:n] * 100, color=COLORS['green'],
            marker='o', linewidth=2, markersize=6, label='Acumulada')
    ax.axhline(95, color=COLORS['amber'], linestyle='--', alpha=0.5, label='95%')

    ax.set_xlabel('Componente Principal', color=COLORS['dim'])
    ax.set_ylabel('Variância Explicada (%)', color=COLORS['dim'])
    ax.set_title('SCREE PLOT · PCA', fontsize=13, color=COLORS['text'],
                 fontweight='bold', pad=15)
    ax.legend(fontsize=9)
    ax.tick_params(colors=COLORS['dim'])

    # ── PCA 2D com clusters ──
    ax2 = axes[1]
    ax2.set_facecolor(COLORS['surface'])

    for i in range(max(labels) + 1):
        mask = labels == i
        ax2.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                    c=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                    s=100, alpha=0.8, label=f'Cluster {i+1}',
                    edgecolors='white', linewidths=0.5)

    for j, nome in enumerate(nomes):
        ax2.annotate(nome, (coords_2d[j, 0], coords_2d[j, 1]),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8, color=COLORS['text'])

    ax2.set_xlabel('PC1', color=COLORS['dim'])
    ax2.set_ylabel('PC2', color=COLORS['dim'])
    ax2.set_title('ATIVOS NO ESPAÇO PCA · K-MEANS',
                  fontsize=13, color=COLORS['text'], fontweight='bold', pad=15)
    ax2.legend(fontsize=9)
    ax2.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m10d_pca_clusters.png', **SAVE)
    plt.close()
    print("   ✔ graficos/m10d_pca_clusters.png")


def grafico_cotovelo(inertias, K_range):
    """Método do cotovelo para K-Means."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['surface'])

    ax.plot(list(K_range), inertias, color=COLORS['green'],
            marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Número de Clusters (k)', color=COLORS['dim'], fontsize=12)
    ax.set_ylabel('Inércia (SSE)', color=COLORS['dim'], fontsize=12)
    ax.set_title('MÉTODO DO COTOVELO · K-MEANS',
                 fontsize=14, color=COLORS['text'], fontweight='bold', pad=15)
    ax.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m10d_elbow_method.png', **SAVE)
    plt.close()
    print("   ✔ graficos/m10d_elbow_method.png")


def grafico_dendrograma(linkage_matrix, nomes):
    """Dendrograma de clustering hierárquico."""
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['surface'])

    dendrogram(linkage_matrix, labels=list(nomes),
               leaf_rotation=45, leaf_font_size=10,
               color_threshold=linkage_matrix[-3, 2],
               above_threshold_color=COLORS['dim'], ax=ax)

    ax.set_title('DENDROGRAMA · CLUSTERING HIERÁRQUICO (WARD)',
                 fontsize=14, color=COLORS['text'], fontweight='bold', pad=15)
    ax.set_ylabel('Distância', color=COLORS['dim'])
    ax.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m10d_dendrograma.png', **SAVE)
    plt.close()
    print("   ✔ graficos/m10d_dendrograma.png")


def grafico_correlacao(corr):
    """Heatmap de correlação."""
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor(COLORS['bg'])

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, vmin=-1, vmax=1, ax=ax,
                annot_kws={'size': 8},
                linewidths=0.5, linecolor=COLORS['surface'])

    ax.set_title('MATRIZ DE CORRELAÇÃO · RETORNOS',
                 fontsize=14, color=COLORS['text'], fontweight='bold', pad=15)
    ax.tick_params(colors=COLORS['dim'])

    plt.tight_layout()
    plt.savefig('../graficos/m10d_correlacao.png', **SAVE)
    plt.close()
    print("   ✔ graficos/m10d_correlacao.png")


def main():
    print("=" * 70)
    print("  MÓDULO 10D · PCA & CLUSTERING DE ATIVOS")
    print("=" * 70)

    precos, retornos = carregar_dados()

    # PCA
    pca_full, pca_2d, coords_2d, var_exp, var_acum = analise_pca(retornos)

    # K-Means
    labels, inertias, K_range = analise_kmeans(retornos, coords_2d)

    # Hierárquico
    linkage_mat, corr = analise_hierarquica(retornos)

    # Gráficos
    print("\n🎨 Gerando gráficos...")
    grafico_pca(var_exp, var_acum, coords_2d, retornos.columns, labels)
    grafico_cotovelo(inertias, K_range)
    grafico_dendrograma(linkage_mat, retornos.columns)
    grafico_correlacao(corr)

    print(f"\n{'=' * 70}")
    print("  ✅ Módulo 10D concluído!")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
