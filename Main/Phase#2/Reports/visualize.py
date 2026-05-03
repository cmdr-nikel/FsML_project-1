import os
import json
import pandas as pd
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "cluster_distribution.json")


def plot_clusters():
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        return

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    for cid, info in data.items():
        label_text = info['label']
        if isinstance(label_text, dict):
            label_text = label_text.get('label', 'unknown')

        counts = info['counts']
        rows.append({
            "Cluster": cid,
            "Label": label_text,
            "Size": info['size'],
            "BMW": counts.get('bmw', 0),
            "Mercedes": counts.get('mercedes', 0),
            "VAG": counts.get('vag', 0)
        })

    df = pd.DataFrame(rows)

    print("DataFrame head:")
    print(df.head())
    print("DataFrame columns:", df.columns)

    # 1. Bar chart
    fig = px.bar(df, x="Label", y="Size", title="Cluster distribution by assigned label")
    fig.show()


def create_visuals():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    for cid, info in data.items():
        label_text = info['label']
        # Handle label format correctly
        if isinstance(label_text, dict):
            label_text = label_text.get('label', 'unknown')

        counts = info['counts']
        total = info['size']
        rows.append({
            "Cluster": int(cid),
            "Label": label_text,
            "BMW": counts.get('bmw', 0) / total,
            "Mercedes": counts.get('mercedes', 0) / total,
            "VAG": counts.get('vag', 0) / total,
            "Size": total
        })

    df = pd.DataFrame(rows).sort_values("Cluster")

    # 1. Heatmap: Brand distribution per cluster
    heatmap_df = df.set_index("Cluster")[['BMW', 'Mercedes', 'VAG']]
    fig_heat = px.imshow(
        heatmap_df.T,
        title="Heatmap: Brand concentration by Cluster",
        aspect="auto",
        color_continuous_scale='Viridis'
    )
    fig_heat.show()

    # 2. Scatterplot: Purity (max brand share) vs Cluster Index
    df['Purity'] = df[['BMW', 'Mercedes', 'VAG']].max(axis=1)
    fig_scatter = px.scatter(
        df, x="Cluster", y="Purity",
        size="Size", color="Label",
        title="Scatterplot: Cluster Purity vs Cluster Index",
        hover_data=["BMW", "Mercedes", "VAG"]
    )
    fig_scatter.show()


def print_dirtiest_clusters(json_path, top_n=5):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dirty_list = []
    for cid, info in data.items():
        total = info['size']
        counts = info['counts']
        purity = max(counts.values()) / total

        dirty_list.append({
            "Cluster": int(cid),
            "Purity": purity,
            "Size": total,
            "Counts": counts
        })

    # Sort by Purity (asc) and Size (desc)
    dirty_list.sort(key=lambda x: (x['Purity'], -x['Size']))

    print(f"\n--- Top {top_n} Dirtiest Clusters ---")
    for cluster in dirty_list[:top_n]:
        print(f"Cluster {cluster['Cluster']}: Purity={cluster['Purity']:.2f}, Size={cluster['Size']}")
        print(f"  Counts: {cluster['Counts']}")




create_visuals(DATA_PATH)
print_dirtiest_clusters(DATA_PATH)

if __name__ == "__main__":
    plot_clusters()