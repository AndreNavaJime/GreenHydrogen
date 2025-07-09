import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load data
base = r"C:\Users\Navar\OneDrive\Documentos\USA\TECHNO-ECONOMIC SIMULATION"
dfs = []
for tech, fname in [("ALKALINE", "USAALKALINE.csv"), ("PEM", "USAPEM.csv"), ("SOEC", "USASOEC.csv")]:
    df0 = pd.read_csv(os.path.join(base, fname))
    df0 = df0.rename(columns={c: "LCOH_$kg" for c in df0.columns if c.startswith("LCOH_")})
    df0["Tech"] = tech
    dfs.append(df0)
df = pd.concat(dfs, ignore_index=True)

# Train/test split
X = pd.get_dummies(df.drop(columns=["State", "LCOH_$kg"]), columns=["Tech"], drop_first=True)
y = df["LCOH_$kg"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# Train models
models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = {
        "model": model,
        "r2": r2_score(y_test, preds),
        "rmse": np.sqrt(mean_squared_error(y_test, preds)),
        "mae": mean_absolute_error(y_test, preds),
        "preds": preds,
        "X_train": X_train
    }

# Save best model
best_name = max(results, key=lambda k: results[k]["r2"])
joblib.dump(results[best_name]["model"], "best_model.pkl")
joblib.dump(results, "results.pkl")
print(f"✅ Saved best model: {best_name}")
