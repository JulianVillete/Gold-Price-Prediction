---

# 🪙 Gold Price Prediction

**Gold Price Prediction** is a machine learning project that uses historical data to forecast future gold prices.
This project applies data preprocessing, visualization, and regression modeling to understand trends and predict price movement with reasonable accuracy.

> 📊 *Built for learning, data exploration, and model experimentation.*

---

## ⚙️ Tech Stack

### 🐍 **Language / Runtime**

* Python 3

### 🧠 **Environment / Notebook**

* Jupyter Notebook
* Google Colab compatible
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### 📦 **Data Handling**

* `pandas` – data manipulation and analysis
* `numpy` – numerical operations and array handling

### 📈 **Visualization**

* `matplotlib` – plotting graphs and data distributions
* `seaborn` – advanced and aesthetic data visualization

### 🤖 **Machine Learning**

* `scikit-learn`

  * `RandomForestRegressor` for predictive modeling
  * `train_test_split` for model validation
  * `metrics` for performance evaluation

### 📂 **Data Source**

* CSV file loaded via:

  ```python
  pd.read_csv('gld_price_data.csv')
  ```

---

## 🚀 Features

* Data cleaning and preprocessing using pandas
* Exploratory Data Analysis (EDA) with visual insights
* Correlation heatmap between gold and other market indicators
* Machine learning model for price prediction
* Performance evaluation using regression metrics

---

## 🧱 Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/Gold-Price-Prediction.git
   cd Gold-Price-Prediction
   ```

2. **Install dependencies**
   (Recommended: use a virtual environment)

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. **Run the notebook**
   Open `gold_price_prediction.ipynb` in:

   * **Jupyter Notebook**, or
   * **Google Colab** (upload the `.ipynb` file)

4. **Load the dataset**
   Ensure `gld_price_data.csv` is in the same directory or update the file path in the notebook.

---

## 📊 Example Workflow

```python
# Load dataset
import pandas as pd
data = pd.read_csv('gld_price_data.csv')

# Visualize correlations
import seaborn as sns
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')

# Train model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X = data.drop(['GLD'], axis=1)
y = data['GLD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)
```

---

## 💡 Future Plans

* Add feature importance visualization
* Experiment with additional regression models (XGBoost, Linear Regression)
* Integrate time-series forecasting models (ARIMA, LSTM)
* Deploy model as a web app (Streamlit or Flask)

---

## 🧑‍💻 Author

**Julian Villete**

> Exploring the intersection of data, simplicity, and intelligent prediction.

---
