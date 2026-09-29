import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

BLOCK_SIZE = 500  # metre
TEST_FRACTION = 0.2
SEED = 42

data = np.load("outputs/recovery_dataset.npz")
feature_cols = ["baseline_vigor", "dnbr_value", "elevation", "slope_deg", "ns_gradient"]
feature_names_tr = ["yangın öncesi NBR", "dNBR (şiddet)", "yükseklik", "eğim (derece)", "kuzey-güney eğim"]

df = pd.DataFrame({col: data[col] for col in feature_cols})
df["nbr_2026"] = data["nbr_2026"]
df["block_x"] = (data["x_coord"] // BLOCK_SIZE).astype(int)
df["block_y"] = (data["y_coord"] // BLOCK_SIZE).astype(int)


def evaluate(X_train, X_test, y_train, y_test, label):
    scaler = StandardScaler()
    Xtr_s, Xte_s = scaler.fit_transform(X_train), scaler.transform(X_test)

    lr = LinearRegression().fit(Xtr_s, y_train)
    preds_lr = lr.predict(Xte_s)

    rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=SEED)
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_test)

    print(f"\n--- {label} (train n={len(X_train)}, test n={len(X_test)}) ---")
    print(f"  Linear Regression   R²={r2_score(y_test, preds_lr):.3f}  RMSE={root_mean_squared_error(y_test, preds_lr):.3f}")
    print(f"  Random Forest       R²={r2_score(y_test, preds_rf):.3f}  RMSE={root_mean_squared_error(y_test, preds_rf):.3f}")
    return rf


# --- 1) NAİF (rastgele piksel) ayrım — karşılaştırma için, SIZINTILI ---
rng = np.random.RandomState(SEED)
idx = rng.permutation(len(df))
split = int(len(df) * (1 - TEST_FRACTION))
train_idx, test_idx = idx[:split], idx[split:]
X = df[feature_cols].values
y = df["nbr_2026"].values
evaluate(X[train_idx], X[test_idx], y[train_idx], y[test_idx], "NAİF rastgele piksel ayrımı (sızıntılı, referans)")

# --- 2) TAMPONLU MEKANSAL BLOK ayrımı — doğru değerlendirme ---
unique_blocks = df[["block_x", "block_y"]].drop_duplicates().reset_index(drop=True)
n_test_blocks = int(TEST_FRACTION * len(unique_blocks))
test_pos = rng.choice(len(unique_blocks), size=n_test_blocks, replace=False)
test_blocks_df = unique_blocks.iloc[test_pos].copy()
test_blocks_df["is_test_block"] = True

buffer_rows = [
    (bx + dx, by + dy)
    for bx, by in test_blocks_df[["block_x", "block_y"]].itertuples(index=False)
    for dx in (-1, 0, 1) for dy in (-1, 0, 1)
]
buffer_blocks_df = pd.DataFrame(buffer_rows, columns=["block_x", "block_y"]).drop_duplicates()
buffer_blocks_df["is_buffer_block"] = True

df = df.merge(test_blocks_df[["block_x", "block_y", "is_test_block"]], on=["block_x", "block_y"], how="left")
df = df.merge(buffer_blocks_df[["block_x", "block_y", "is_buffer_block"]], on=["block_x", "block_y"], how="left")
df["is_test_block"] = df["is_test_block"].fillna(False)
df["is_buffer_block"] = df["is_buffer_block"].fillna(False)

df["split"] = "train"
df.loc[df["is_buffer_block"] & ~df["is_test_block"], "split"] = "buffer"
df.loc[df["is_test_block"], "split"] = "test"

print(f"\nBlok sayısı: {len(unique_blocks)} (test için seçilen: {n_test_blocks})")
print(f"Piksel dağılımı — train: {(df['split']=='train').sum()}  test: {(df['split']=='test').sum()}  tampon (dışlanan): {(df['split']=='buffer').sum()}")

train_mask = (df["split"] == "train").values
test_mask = (df["split"] == "test").values
X = df[feature_cols].values
y = df["nbr_2026"].values
rf_spatial = evaluate(X[train_mask], X[test_mask], y[train_mask], y[test_mask], "TAMPONLU MEKANSAL BLOK ayrımı (doğru değerlendirme)")

print("\nÖzellik önemi (mekansal blok ayrımlı Random Forest):")
for name, importance in zip(feature_names_tr, rf_spatial.feature_importances_):
    print(f"  {name:20s}: {importance:.3f}")