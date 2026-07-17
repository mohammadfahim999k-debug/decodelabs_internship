"""
generate_data.py
-----------------
Generates a synthetic, unlabeled retail customer dataset with 20+ behavioral
and demographic columns. The data is built from four latent (hidden) customer
archetypes so that the clustering pipeline has real structure to recover —
but no cluster/label column is included in the final CSV, since the whole
point of Project 3 is *unsupervised* discovery.

Run:
    python src/generate_data.py
Output:
    data/customer_data.csv
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_CUSTOMERS = 800

rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. Define four latent archetypes (this is what K-Means should rediscover)
#    Each archetype has its own distribution for each behavioral feature.
# ---------------------------------------------------------------------------
archetypes = {
    "affluent_conservative": dict(
        weight=0.25,
        age=(41, 8), income=(88000, 12000), spending_score=(18, 8),
        annual_purchases=(6, 2), avg_order_value=(140, 30),
        website_visits_month=(3, 1.5), app_sessions_month=(2, 1),
        cart_abandon_rate=(0.55, 0.1), discount_usage_rate=(0.10, 0.05),
        loyalty_points=(1200, 300), tenure_years=(7, 2),
        support_tickets=(1, 1), email_open_rate=(0.35, 0.1),
        social_engagement=(0.10, 0.05), return_rate=(0.05, 0.03),
        wishlist_items=(2, 1), referrals_made=(1, 1),
        premium_category_pct=(0.60, 0.15), weekend_purchase_pct=(0.30, 0.1),
        mobile_purchase_pct=(0.20, 0.1),
    ),
    "high_value_trendsetter": dict(
        weight=0.25,
        age=(33, 6), income=(86000, 15000), spending_score=(82, 8),
        annual_purchases=(28, 5), avg_order_value=(160, 35),
        website_visits_month=(18, 4), app_sessions_month=(22, 5),
        cart_abandon_rate=(0.15, 0.05), discount_usage_rate=(0.20, 0.08),
        loyalty_points=(4200, 600), tenure_years=(4, 1.5),
        support_tickets=(1, 1), email_open_rate=(0.55, 0.1),
        social_engagement=(0.65, 0.12), return_rate=(0.08, 0.04),
        wishlist_items=(9, 3), referrals_made=(4, 2),
        premium_category_pct=(0.70, 0.12), weekend_purchase_pct=(0.55, 0.1),
        mobile_purchase_pct=(0.70, 0.1),
    ),
    "budget_conscious_explorer": dict(
        weight=0.30,
        age=(25, 5), income=(25700, 6000), spending_score=(79, 9),
        annual_purchases=(22, 6), avg_order_value=(35, 10),
        website_visits_month=(25, 6), app_sessions_month=(30, 7),
        cart_abandon_rate=(0.45, 0.1), discount_usage_rate=(0.75, 0.1),
        loyalty_points=(1800, 400), tenure_years=(1.5, 1),
        support_tickets=(2, 1.5), email_open_rate=(0.40, 0.1),
        social_engagement=(0.75, 0.12), return_rate=(0.18, 0.06),
        wishlist_items=(12, 4), referrals_made=(3, 2),
        premium_category_pct=(0.10, 0.08), weekend_purchase_pct=(0.60, 0.1),
        mobile_purchase_pct=(0.85, 0.08),
    ),
    "conservative_minimizer": dict(
        weight=0.20,
        age=(45, 9), income=(26300, 7000), spending_score=(21, 9),
        annual_purchases=(3, 1.5), avg_order_value=(28, 8),
        website_visits_month=(2, 1), app_sessions_month=(1, 1),
        cart_abandon_rate=(0.65, 0.1), discount_usage_rate=(0.55, 0.1),
        loyalty_points=(300, 150), tenure_years=(5, 2.5),
        support_tickets=(3, 1.5), email_open_rate=(0.15, 0.07),
        social_engagement=(0.08, 0.05), return_rate=(0.10, 0.05),
        wishlist_items=(1, 1), referrals_made=(0, 1),
        premium_category_pct=(0.15, 0.1), weekend_purchase_pct=(0.20, 0.1),
        mobile_purchase_pct=(0.15, 0.1),
    ),
}

feature_names = [k for k in archetypes["affluent_conservative"].keys() if k != "weight"]

rows = []
labels_for_validation = []  # kept ONLY for internal sanity-checking, not exported

weights = [a["weight"] for a in archetypes.values()]
names = list(archetypes.keys())

for _ in range(N_CUSTOMERS):
    archetype_name = rng.choice(names, p=weights)
    params = archetypes[archetype_name]
    row = {}
    for feat in feature_names:
        mean, std = params[feat]
        val = rng.normal(mean, std)
        row[feat] = val
    rows.append(row)
    labels_for_validation.append(archetype_name)

df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 2. Clean up to realistic ranges/types
# ---------------------------------------------------------------------------
df["age"] = df["age"].clip(18, 75).round(0).astype(int)
df["income"] = df["income"].clip(8000, 200000).round(0).astype(int)
df["spending_score"] = df["spending_score"].clip(1, 100).round(1)
df["annual_purchases"] = df["annual_purchases"].clip(0, None).round(0).astype(int)
df["avg_order_value"] = df["avg_order_value"].clip(5, None).round(2)
df["website_visits_month"] = df["website_visits_month"].clip(0, None).round(0).astype(int)
df["app_sessions_month"] = df["app_sessions_month"].clip(0, None).round(0).astype(int)
df["cart_abandon_rate"] = df["cart_abandon_rate"].clip(0, 1).round(3)
df["discount_usage_rate"] = df["discount_usage_rate"].clip(0, 1).round(3)
df["loyalty_points"] = df["loyalty_points"].clip(0, None).round(0).astype(int)
df["tenure_years"] = df["tenure_years"].clip(0.1, None).round(2)
df["support_tickets"] = df["support_tickets"].clip(0, None).round(0).astype(int)
df["email_open_rate"] = df["email_open_rate"].clip(0, 1).round(3)
df["social_engagement"] = df["social_engagement"].clip(0, 1).round(3)
df["return_rate"] = df["return_rate"].clip(0, 1).round(3)
df["wishlist_items"] = df["wishlist_items"].clip(0, None).round(0).astype(int)
df["referrals_made"] = df["referrals_made"].clip(0, None).round(0).astype(int)
df["premium_category_pct"] = df["premium_category_pct"].clip(0, 1).round(3)
df["weekend_purchase_pct"] = df["weekend_purchase_pct"].clip(0, 1).round(3)
df["mobile_purchase_pct"] = df["mobile_purchase_pct"].clip(0, 1).round(3)

# ---------------------------------------------------------------------------
# 3. Add categorical / identifier columns (also realistic, non-numeric noise)
# ---------------------------------------------------------------------------
df.insert(0, "customer_id", [f"CUST{100000+i}" for i in range(N_CUSTOMERS)])
df["gender"] = rng.choice(["Female", "Male"], size=N_CUSTOMERS, p=[0.52, 0.48])
df["region"] = rng.choice(
    ["North", "South", "East", "West", "Central"], size=N_CUSTOMERS
)
df["preferred_channel"] = rng.choice(
    ["Mobile App", "Website", "In-Store"], size=N_CUSTOMERS, p=[0.5, 0.35, 0.15]
)

# ---------------------------------------------------------------------------
# 4. Inject a small amount of missingness and light noise, like real data
# ---------------------------------------------------------------------------
for col in ["income", "email_open_rate", "support_tickets"]:
    mask = rng.random(N_CUSTOMERS) < 0.02
    df.loc[mask, col] = np.nan

df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

df.to_csv("data/customer_data.csv", index=False)
print(f"Saved {df.shape[0]} rows x {df.shape[1]} columns to data/customer_data.csv")
print(df.head())
