import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"d:\BizOwn\000_Biw2\51_googleantigravity\007_kaggle_Biohub-Cell_Tracking_During_Development\s5\github")
WORKING_DIR = BASE_DIR / "working"
DATA_DIR = BASE_DIR / "s5_analysys_data"

p_edge = WORKING_DIR / "s5_015DYNRADIUS_04_check_edges_summary_blobdog_lgbm.csv"
p_node = WORKING_DIR / "s5_015DYNRADIUS_02_check_nodes_summary_blobdog_lgbm.csv"
p_pe   = WORKING_DIR / "s5_015DYNRADIUS_05_pe_ratio_summary.csv"

df_e = pd.read_csv(p_edge)
df_n = pd.read_csv(p_node)
df_pe = pd.read_csv(p_pe)

df_m = df_e.merge(df_n[['dataset', 'recall', 'total_gt_nodes', 'mean_distance_um']], on='dataset', suffixes=('_edge', '_node'))
df_m = df_m.merge(df_pe[['dataset', 'pe_ratio_final', 'raw_nodes', 'final_nodes', 'cull_rate_pct']], on='dataset')

df_m['edge_recall'] = df_m['official_edge_tp'] / (df_m['official_edge_tp'] + df_m['official_edge_fn'])
sorted_df = df_m.sort_values('edge_recall').reset_index(drop=True)

out_csv = DATA_DIR / "s5_016_edge_recall_full_ranking.csv"
cols = ['dataset', 'edge_recall', 'recall_node', 'total_gt_edges', 'total_gt_nodes', 'official_edge_tp', 'official_edge_fn', 'pe_ratio_final', 'cull_rate_pct']
sorted_df[cols].to_csv(out_csv, index=False)
print(f"Total datasets: {len(sorted_df)}")
print("Distribution of Edge Recall:")
print(f"  < 0.50     : {(sorted_df['edge_recall'] < 0.50).sum()} datasets")
print(f"  0.50 - 0.60: {((sorted_df['edge_recall'] >= 0.50) & (sorted_df['edge_recall'] < 0.60)).sum()} datasets")
print(f"  0.60 - 0.70: {((sorted_df['edge_recall'] >= 0.60) & (sorted_df['edge_recall'] < 0.70)).sum()} datasets")
print(f"  0.70 - 0.80: {((sorted_df['edge_recall'] >= 0.70) & (sorted_df['edge_recall'] < 0.80)).sum()} datasets")
print(f"  >= 0.80    : {(sorted_df['edge_recall'] >= 0.80).sum()} datasets")

print("\n=== TOP 25 LOWEST EDGE RECALL DATASETS ===")
print(f"{'No':<3} {'Dataset':<16} {'EdgeRec':<8} {'NodeRec':<8} {'GT_Edges':<9} {'TP':<6} {'FN':<6} {'P/E':<6} {'Primary Bottleneck'}")
print("-" * 85)

for i, r in sorted_df.head(25).iterrows():
    # Diagnose bottleneck
    if r['recall_node'] < 0.65:
        bottleneck = f"Node Detection Failure (NodeRec={r['recall_node']:.2f})"
    elif r['recall_node'] - r['edge_recall'] > 0.35:
        bottleneck = f"Tracking Cut / Disconnect (Gap={r['recall_node'] - r['edge_recall']:.2f})"
    elif r['cull_rate_pct'] > 25:
        bottleneck = f"Heavy Culling (Cull={r['cull_rate_pct']:.1f}%)"
    else:
        bottleneck = "Combined Loss"
        
    print(f"{i+1:<3} {r['dataset']:<16} {r['edge_recall']:<8.3f} {r['recall_node']:<8.3f} {int(r['total_gt_edges']):<9} {int(r['official_edge_tp']):<6} {int(r['official_edge_fn']):<6} {r['pe_ratio_final']:<6.3f} {bottleneck}")
