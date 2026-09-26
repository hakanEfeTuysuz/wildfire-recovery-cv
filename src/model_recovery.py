import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

data = np.load("outputs/recovery_dataset.npz")
X = np.column_stack([
    data["baseline_vigor"], data["dnbr_value"],
    data["elevation"], data["slope_deg"], data["ns_gradient"],
])
y = data["nbr_2026"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

feature_names = ["yangın öncesi NBR", "dNBR (şiddet)", "yükseklik", "eğim (derece)", "kuzey-güney eğim"]

# Linear Regression: özellikleri standartlaştırıp (ölçek farkını gidererek) kullan
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
preds_lr = lr.predict(X_test_scaled)
print(f"{'Linear Regression':20s}  R²={r2_score(y_test, preds_lr):.3f}  RMSE={root_mean_squared_error(y_test, preds_lr):.3f}")

# Random Forest: ölçeklemeye ihtiyaç duymuyor, ham veriyle
rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
preds_rf = rf.predict(X_test)
print(f"{'Random Forest':20s}  R²={r2_score(y_test, preds_rf):.3f}  RMSE={root_mean_squared_error(y_test, preds_rf):.3f}")

print("\nÖzellik önemi (Random Forest):")
for name, importance in zip(feature_names, rf.feature_importances_):
    print(f"  {name:20s}: {importance:.3f}")

print("\nÖzellik katsayıları (Linear Regression, standartlaştırılmış):")
for name, coef in zip(feature_names, lr.coef_):
    print(f"  {name:20s}: {coef:.4f}")