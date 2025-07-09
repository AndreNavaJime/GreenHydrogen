# GreenHydrogen
🧪 Hydrogen LCOH Dashboard – Instructions
🔧 Requirements
To run this dashboard locally, install the required packages:

bash
Copiar
Editar
pip install -r requirements.txt
Or install manually:

bash
Copiar
Editar
pip install streamlit pandas numpy geopandas scikit-learn xgboost lightgbm catboost plotly matplotlib fpdf
🚀 Launch the Dashboard
To start the dashboard locally, run:

bash
Copiar
Editar
streamlit run USMACHINE.py
Ensure all required CSV input files (USAALKALINE.csv, USAPEM.csv, USASOEC.csv, etc.) are located in the same directory or adjust the load_data() function paths accordingly.

🌐 Run Online with Streamlit Cloud
You can also run the dashboard via Streamlit Cloud:https://streamlit.io/cloud
