# Customer Segmentation — Unsupervised Learning (Project 3)

> DecodeLabs Data Science Industrial Training Kit — Batch 2026

Segment unlabeled retail customers into actionable business personas using
**PCA + K-Means**, with the choice of cluster count mathematically justified
via the **Elbow Method** and **Silhouette Score**.

![Cluster scatter](outputs/cluster_scatter_pca.png)

## Overview

Enterprise retail datasets routinely capture 20+ behavioral and demographic
columns per customer (purchase frequency, spend, engagement, channel
preference, etc.) with no ready-made labels. This project builds a full,
reproducible pipeline that:

1. **Scales** every feature with `StandardScaler` so no single column (e.g.
   income) dominates distance calculations.
2. **Compresses** the feature space with **PCA**, keeping only as many
   components as needed to retain 95% of cumulative variance.
3. **Clusters** the compressed data with **K-Means**, choosing K by combining
   the Elbow Method (WCSS inflection point) and Silhouette Score (cluster
   cohesion vs. separation).
4. **Translates** the resulting clusters back into human-readable centroids
   and four actionable **business personas**, each paired with a recommended
   marketing action.

## Repository structure

```
customer-segmentation/
├── data/
│   └── customer_data.csv          # synthetic, unlabeled retail dataset (800 customers, 24 columns)
├── notebooks/
│   └── customer_segmentation.ipynb  # full walkthrough with narrative + charts (start here)
├── src/
│   ├── generate_data.py           # generates the synthetic dataset
│   └── segmentation.py            # reusable pipeline: scale → PCA → K-Means → personas
├── outputs/
│   ├── pca_explained_variance.png
│   ├── elbow_method.png
│   ├── silhouette_scores.png
│   ├── cluster_scatter_pca.png
│   ├── customers_with_segments.csv
│   └── persona_summary.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

`data/customer_data.csv` contains 800 synthetic customers with 24 columns,
including:

- **Demographics:** `age`, `income`, `gender`, `region`
- **Spend behavior:** `spending_score`, `annual_purchases`, `avg_order_value`,
  `discount_usage_rate`, `premium_category_pct`
- **Engagement:** `website_visits_month`, `app_sessions_month`,
  `email_open_rate`, `social_engagement`, `wishlist_items`
- **Loyalty/risk:** `loyalty_points`, `tenure_years`, `return_rate`,
  `cart_abandon_rate`, `support_tickets`, `referrals_made`

No label or cluster column is included — that's the point of an
*unsupervised* task. The data is synthetic (generated from four latent
archetypes with realistic noise) so the pipeline can be run and validated
end-to-end without a proprietary dataset; swap in your own CSV with the same
numeric-feature convention to use real data.

## Quickstart

```bash
# 1. Clone and set up
git clone <your-repo-url>
cd customer-segmentation
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# 2. (Optional) regenerate the synthetic dataset
python src/generate_data.py

# 3. Run the full pipeline from the command line
python src/segmentation.py

# 4. Or open the annotated notebook
jupyter notebook notebooks/customer_segmentation.ipynb
```

Running `src/segmentation.py` prints the chosen number of PCA components,
the final K, and a persona summary, and writes all charts + result CSVs into
`outputs/`.

## Methodology

| Stage | Technique | Why |
|---|---|---|
| Scale | `StandardScaler` (z-score) | Prevents high-magnitude columns (e.g. income) from swallowing smaller-scale behavioral signals in Euclidean distance |
| Compress | PCA, 95% variance retained | Counters the curse of dimensionality across 20+ correlated columns |
| Cluster | K-Means | Minimizes within-cluster sum of squares (WCSS) to find spherical, cohesive groupings |
| Choose K | Elbow Method + Silhouette Score | Elbow finds the inflection point of diminishing returns; Silhouette confirms cohesion vs. separation. The pipeline reconciles both, preferring the simpler (elbow) K when its silhouette score is competitive |
| Translate | Inverse-mapped cluster means | Reports persona metrics in original units (age, $, %) rather than abstract PCA coordinates |

## Results

The pipeline recovers **4 clusters** with a strong silhouette score, matching
the four latent archetypes used to generate the data:

| Persona | Profile | Recommended Action |
|---|---|---|
| **Affluent Conservatives** | Older, high income, low spending score | High-touch support, extended warranties, loyalty programs |
| **High-Value Trendsetters** | Mid-30s, high income, high spending score | Exclusive perks, early access, experiential marketing |
| **Budget-Conscious Explorers** | Young, lower income, high spending score | Influencer campaigns, flash sales, buy-now-pay-later |
| **Conservative Minimizers** | Older, lower income, low spending score | Minimize acquisition spend, clear price/value messaging |

Full per-cluster metrics are in `outputs/persona_summary.csv`, and
per-customer segment assignments are in `outputs/customers_with_segments.csv`.

## Key skills demonstrated

Dimensionality reduction (PCA), K-Means clustering, distance metrics
(Euclidean/WCSS), unsupervised model selection (Elbow Method, Silhouette
Score), and business-intelligence translation of statistical output into
actionable personas.

## License

MIT — feel free to fork, adapt, and reuse for your own portfolio.
