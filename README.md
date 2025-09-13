🧪 Hydrogen LCOH Dashboard

Explore U.S. green hydrogen techno-economic analysis (TEA) and machine-learning (ML) projections (2022–2060) across Alkaline, PEM, and SOEC electrolysis.
Try it live 👉 Launch the Dashboard Online

⚡ Quick Start

For experienced users:

pip install -r requirements.txt
# or manually:
pip install streamlit pandas numpy geopandas scikit-learn xgboost lightgbm catboost plotly matplotlib fpdf
streamlit run USMACHINE.py


Make sure the CSV files (USAALKALINE.csv, USAPEM.csv, USASOEC.csv) exist in the working directory (or update load_data() paths).

🌐 Or just open it online:
👉 https://greenhydrogen-xg5o4rotmarima9zughosq.streamlit.app

🔧 Full Installation Guide
1️⃣ Clone the repo
git clone https://github.com/your-username/greenhydrogen.git
cd greenhydrogen

2️⃣ Python version

Requires Python 3.12
(see requires-python>=3.12,<3.13 in requirements.txt).

3️⃣ Create environment

Conda (recommended):

conda create -n lcoh python=3.12 -y
conda activate lcoh


venv:

python3.12 -m venv lcoh
source lcoh/bin/activate   # Linux/Mac
lcoh\Scripts\activate      # Windows

4️⃣ Install dependencies
pip install -r requirements.txt


Or install manually (see list inside requirements.txt).

⚠️ Geospatial dependencies (geopandas, fiona, gdal) may fail on Windows/Streamlit Cloud:

Conda:

conda install -c conda-forge geopandas gdal proj shapely fiona


Streamlit Cloud: add packages.txt with:

gdal-bin
libgdal-dev
proj-bin

5️⃣ Generate input data

Run the techno-economic models to produce CSV inputs:

python "ALKALINE USA.py"
python "PEM USA.py"
python "SOEC USA.py"


This generates:

MACHINE LEARNING/
 ├─ USAALKALINE.csv
 ├─ USAPEM.csv
 └─ USASOEC.csv

6️⃣ Launch dashboard
streamlit run USMACHINE.py

🖥️ Usage Guide

Once launched, the app opens in your browser (http://localhost:8501 by default).
Or skip installation and use the live version:
👉 https://greenhydrogen-xg5o4rotmarima9zughosq.streamlit.app

Sidebar Controls

Prediction Mode: ML Model (regressors + SHAP explanations) or Baseline TEA (deterministic techno-economics).

Technology: Alkaline, PEM, or SOEC.

Year: 2022 → 2060 (decade steps).

State: one of the 48 contiguous U.S. states.

Sliders: adjust CAPEX, efficiency, and electricity price.

Parameter source:

DOE targets → resets to DOE reference values.

Fix inputs across years → keep your adjustments consistent.

Tabs

Prediction → scenario inputs + LCOH results.

Stats → R², RMSE, MAE, and cross-validation metrics (ML mode).

Feature Importance & Trends → SHAP plots and LCOH trend heatmaps.

Map → interactive U.S. choropleths of LCOH.

State Comparison → compare LCOH sensitivity across states.

U.S. Summary → national averages by decade.

Download Center → export results as PDF or ZIP bundles.

Outputs

Interactive Plotly charts (exportable as PNG).

Scenario reports (PDF/ZIP).

SHAP explanations to interpret model predictions.

🌐 Deploy Online Yourself

Push this repo to GitHub.

On Streamlit Cloud
, create a new app and set entry point to USMACHINE.py.

Ensure you have:

requirements.txt (already included).

packages.txt with system libs.

Deploy → your app is served at a URL like:

https://greenhydrogen-xxxx.streamlit.app

