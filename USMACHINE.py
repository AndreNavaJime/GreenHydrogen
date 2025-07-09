import streamlit as st
import pandas as pd
import plotly.express as px
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
from fpdf import FPDF
import tempfile
import os

# --- Load & preprocess data ---
USE_LOCAL = os.path.exists(r"C:\Users\Navar\OneDrive\Documentos\USA\TECHNO-ECONOMIC SIMULATION")

if USE_LOCAL:
    base = r"C:\Users\Navar\OneDrive\Documentos\USA\TECHNO-ECONOMIC SIMULATION"
    files_info = [
        (os.path.join(base, "USAALKALINE.csv"), "LCOH_Alk_$kg", "ALKALINE"),
        (os.path.join(base, "USAPEM.csv"), "LCOH_PEM_$kg", "PEM"),
        (os.path.join(base, "USASOEC.csv"), "LCOH_SOEC_$kg", "SOEC"),
    ]
else:
    base_url = "https://raw.githubusercontent.com/AndreNavaJime/greenhydrogen/main/"
    files_info = [
        (base_url + "USAALKALINE.csv", "LCOH_Alk_$kg", "ALKALINE"),
        (base_url + "USAPEM.csv", "LCOH_PEM_$kg", "PEM"),
        (base_url + "USASOEC.csv", "LCOH_SOEC_$kg", "SOEC"),
    ]

df_list = []
for source, lcoh_col, tech in files_info:
    d = pd.read_csv(source)
    d["LCOH"] = d[lcoh_col]
    d["Technology"] = tech
    df_list.append(d)

df = pd.concat(df_list, ignore_index=True)
df.drop(columns=[lcoh_col for _, lcoh_col, _ in files_info], inplace=True, errors="ignore")
df.dropna(subset=["LCOH"], inplace=True)


dfs = []
for path, col, tech in files_info:
    d = pd.read_csv(path)
    d["LCOH"] = d[col]
    d["Technology"] = tech
    dfs.append(d)

df = pd.concat(dfs, ignore_index=True)
df = df.drop(columns=[c for c, _, _ in files_info if c.endswith(".csv")], errors="ignore")
df.dropna(subset=["LCOH"], inplace=True)

# Features & target
X = pd.get_dummies(df.drop(columns=["LCOH"]), drop_first=True)
y = df["LCOH"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = CatBoostRegressor(verbose=0, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics & residuals
df_err = X_test.copy()
df_err["Actual"] = y_test.values
df_err["Pred"] = y_pred
df_err["Residual"] = df_err["Actual"] - df_err["Pred"]
r2 = r2_score(y_test, y_pred)
rmse = ((y_test - y_pred) ** 2).mean() ** 0.5
mae = (y_test - y_pred).abs().mean()

# --- Streamlit UI ---
st.title("🧠 LCOH Prediction Dashboard")

year = st.sidebar.slider("Year", int(df["Year"].min()), int(df["Year"].max()), int(df["Year"].min()))
tech = st.sidebar.selectbox("Technology", df["Technology"].unique())

# Show example prediction
pred_example = model.predict(X_test.iloc[[0]])[0]
st.markdown(f"### Example prediction: **${pred_example:.2f}/kg**")

tabs = st.tabs(["🏷️ Metrics", "📊 Residuals", "🗂️ Top Errors", "🌍 Trends", "📥 Export"])

with tabs[0]:
    stats = pd.DataFrame({
        "Metric": ["R²", "RMSE", "MAE"],
        "CatBoost": [r2, rmse, mae]
    }).round(3).set_index("Metric")
    st.dataframe(stats.style.background_gradient("viridis"))

with tabs[1]:
    fig = px.histogram(df_err, x="Residual", nbins=50, title="Residuals Distribution")
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    top5 = df_err.assign(abs_err=df_err["Residual"].abs()).nlargest(5, "abs_err")
    st.dataframe(top5[["Actual", "Pred", "Residual"]])

with tabs[3]:
    pivot = df.groupby(["Technology", "Year"])["LCOH"].mean().unstack()
    fig = px.imshow(pivot, color_continuous_scale="viridis", title="LCOH Trends")
    st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    # CSV export
    csv_buf = io.StringIO()
    df_err.to_csv(csv_buf, index=False)
    st.download_button("Download CSV", csv_buf.getvalue(), "predictions.csv")

    # Prepare PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "LCOH Prediction Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(5)
    pdf.cell(0, 8, f"R²: {r2:.3f}")
    pdf.ln(5)
    pdf.cell(0, 8, f"RMSE: {rmse:.3f}")
    pdf.ln(5)
    pdf.cell(0, 8, f"MAE: {mae:.3f}")
    pdf.ln(10)

    # Plot residual vs pred
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        plt.figure(figsize=(6,4))
        sns.scatterplot(data=df_err, x="Pred", y="Residual")
        plt.title("Residual vs Prediction")
        plt.tight_layout()
        plt.savefig(tmp.name)
        plt.close()
    pdf.image(tmp.name, x=10, w=190)
    os.unlink(tmp.name)

    pdf_out = pdf.output(dest="S").encode("latin1")
    st.download_button("Download PDF", pdf_out, "report.pdf", mime="application/pdf")
