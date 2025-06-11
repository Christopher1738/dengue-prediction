DENGUE OUTBREAK PREDICTION SYSTEM
---------------------------------

📅 Week 2 Assignment: AI for Sustainable Development  
📘 Theme: "Machine Learning Meets the UN Sustainable Development Goals (SDGs)"  
👨‍💻 Student: Christopher  
🎯 SDG Focus: Goal 3 – Good Health and Well-being  

🧠 PROJECT OVERVIEW
-------------------
This project is a machine learning-based system designed to predict dengue outbreaks using environmental and demographic factors. It aligns with the United Nations Sustainable Development Goal 3 (Good Health and Well-being), aiming to support proactive disease prevention in vulnerable regions.

The application allows users to input data such as temperature, rainfall, humidity, vegetation index, and population density, and then provides:
- Expected number of dengue cases
- Probability of a dengue outbreak
- Suggested actions based on risk level

✅ MACHINE LEARNING APPROACH
----------------------------
- Supervised Learning
- Trained a **regression model** to predict dengue cases
- Trained a **classification model** to estimate outbreak probability
- Used features: temperature, rainfall, humidity, vegetation index, population density, past cases, week of the year

📊 TOOLS & LIBRARIES USED
--------------------------
- Python
- Pandas, NumPy, Scikit-learn
- Matplotlib
- Streamlit
- Joblib (for saving/loading models)
- Jupyter Notebook (for training and preprocessing)

🧹 FILE STRUCTURE
------------------
- `app.py` – Streamlit application file
- `data_preprocessing.py` – Cleans and prepares the dataset
- `data_generation.py` – Simulates or augments data (if needed)
- `train_model.py` – Trains ML models (regression + classification)
- `case_predictor.pkl` – Saved regression model
- `outbreak_classifier.pkl` – Saved classification model
- `scaler.pkl` – StandardScaler used for preprocessing
- `README.txt` – This file
- `screenshots/` – Folder for demo screenshots (to be added)
  
⚙️ HOW TO RUN LOCALLY
---------------------
1. Clone the repo:
git clone https://github.com/Christopher1738/dengue-prediction
cd dengue-prediction

markdown
Copy
Edit

2. Install dependencies:
pip install -r requirements.txt

markdown
Copy
Edit

3. Run the app:
streamlit run app.py

makefile
Copy
Edit

📁 Requirements.txt should include:
streamlit
pandas
numpy
scikit-learn
matplotlib
joblib

pgsql
Copy
Edit

🧪 ETHICAL CONSIDERATIONS
--------------------------
- Models are only as reliable as the data; regional bias or missing data may affect accuracy.
- No personally identifiable information is used.
- This tool is for **educational and planning purposes only**, not a replacement for medical professionals or public health authorities.

📌 IMPACT
---------
By forecasting dengue risks, this tool enables early interventions, improves resource allocation, and supports public health decision-making in regions prone to outbreaks—contributing directly to SDG 3.

🖼️ SCREENSHOTS
--------------
Screenshots of the app, input form, and results will be added to the `screenshots/` folder.

🔗 GitHub Repository
--------------------
https://github.com/Christopher1738/dengue-prediction











