import streamlit as st
import pandas as pd
import plotly.express as px
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
from fpdf import FPDF
import os

# --- Load & preprocess data ---
base_path = r"C:\Users\Navar\OneDrive\Documentos\USA\TECHNO-ECONOMIC SIMULATION"
files_info = [
    ("USAALKALINE.csv", "LCOH_Alk_$kg", "ALKALINE"),
    ("USAPEM.csv", "LCOH_PEM_$kg", "PEM"),
    ("USASOEC.csv", "LCOH_SOEC_$kg", "SOEC"),
]
df_list = []
for filename, lcoh_col, tech in files_info:
    path = os.path.join(base_path, filename)
    d = pd.read_csv(path)
    d["LCOH"] = d[lcoh_col]
    d["Technology"] = tech
    df_list.append(d)
df = pd.concat(df_list, ignore_index=True)
df.drop(columns=[col for _, col, _ in files_info], inplace=True, errors="ignore")
df.dropna(subset=["LCOH"], inplace=True)

# Features and target
X = pd.get_dummies(df.drop(columns=["LCOH"]), drop_first=True)
y = df["LCOH"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = CatBoostRegressor(verbose=0, random_state=42)
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)

# Save results
results = {
    "CatBoost": {
        "r2": r2_score(y_test, y_test_pred),
        "rmse": ((y_test - y_test_pred) ** 2).mean() ** 0.5,
        "mae": (y_test - y_test_pred).abs().mean(),
        "Actual": y_test.values,
        "Pred": y_test_pred,
        "df": X_test.copy()
    }
}
results["CatBoost"]["df"]["Actual"] = y_test.values
results["CatBoost"]["df"]["Pred"] = y_test_pred
results["CatBoost"]["df"]["Residual"] = y_test.values - y_test_pred

# --- Streamlit Interface ---
st.title("🧠 LCOH Prediction Dashboard")

# Sidebar filters
year = st.sidebar.slider("Select Year", int(df["Year"].min()), int(df["Year"].max()), int(df["Year"].min()))
tech = st.sidebar.selectbox("Select Technology", df["Technology"].unique())

# Filtered data
df_filtered = df[(df["Year"] == year) & (df["Technology"] == tech)]

# Display prediction example
pred = model.predict(X_test.iloc[[0]])[0]
st.markdown(f"### 🧪 Predicted LCOH → **${pred:.2f}/kg** using **CatBoost**")

tabs = st.tabs(["🏷️ Metrics", "📊 Residuals", "🗂️ Top Errors", "🌍 Visuals", "📥 Export"])

with tabs[0]:
    stats_df = pd.DataFrame({
        "CatBoost": {
            "R²": results["CatBoost"]["r2"],
            "RMSE": results["CatBoost"]["rmse"],
            "MAE": results["CatBoost"]["mae"]
        }
    }).T.round(3)
    st.dataframe(stats_df.style.background_gradient(cmap="viridis"))

with tabs[1]:
    df_err = results["CatBoost"]["df"]
    tech_cols = [col for col in df_err.columns if f"Technology_{tech}" in col]
    fig = px.histogram(df_err, x="Residual", title=f"Residuals Histogram ({tech})", nbins=50)
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    top5 = df_err.assign(Error=abs(df_err["Residual"])).nlargest(5, "Error")
    st.write("### 🔻 Top 5 Predictions with Highest Error")
    st.dataframe(top5[["Actual", "Pred", "Residual"]])

with tabs[3]:
    heat_df = df.groupby(["Technology", "Year"])["LCOH"].mean().reset_index()
    pivot = heat_df.pivot(index="Technology", columns="Year", values="LCOH")
    fig_heat = px.imshow(pivot, color_continuous_scale="Viridis",
                         title="LCOH Trend Heatmap", template="plotly_dark")
    st.plotly_chart(fig_heat, use_container_width=True)

with tabs[4]:
    # CSV Export
    csv_buffer = io.StringIO()
    df_err.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download CSV", csv_buffer.getvalue(), "predictions.csv")

    # PDF Export
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "LCOH Prediction Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    for k, v in stats_df.iloc[0].items():
        pdf.cell(0, 8, f"{k}: {v:.3f}", ln=True)

# Place this import at the very top with the others:
import tempfile

# Create residual plot and save to a temporary PNG
with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df_err["Pred"], y=df_err["Residual"])
    plt.title("Residuals vs Predictions")
    plt.tight_layout()
    plt.savefig(tmp_img.name, format="PNG")
    plt.close()
    image_path = tmp_img.name  # Save temp path

# Insert image into PDF (after the with block)
pdf.image(image_path, x=10, y=60, w=190)

# Finalize PDF
pdf_output = pdf.output(dest='S').encode('latin1')
st.download_button("📥 Download PDF", pdf_output, "report.pdf", mime="application/pdf")
