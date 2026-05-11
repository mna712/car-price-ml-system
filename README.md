
# Car Price Prediction - Multi Model ML System

This project is a **machine learning regression system** that predicts the selling price of cars based on various features such as mileage, engine power, fuel type, transmission, and car age.

It includes **multiple ML models**, full data preprocessing, feature engineering, and a simple UI for predictions.

---

## 📊 Project Overview

The goal of this project is to compare different regression models and select the best-performing model for predicting car prices.

### 🔍 Key Steps:
- Data cleaning and preprocessing
- Feature engineering (car age, encoding categorical variables)
- Handling missing values
- Data visualization (EDA)
- Training multiple ML models
- Model evaluation and comparison
- Saving trained models for deployment

---

## 🧠 Machine Learning Models Used

- Linear Regression
- Polynomial Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Decision Tree Regressor
- Random Forest Regressor ⭐ (Best Performing)
- K-Nearest Neighbors (KNN)
- Support Vector Regressor (SVR)

---

## 📈 Evaluation Metrics

- R² Score
- Root Mean Squared Error (RMSE)

Models were compared and ranked based on performance.

---

## 🛠️ Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Joblib (model saving)
- Jupyter Notebook
- Simple Frontend UI (for prediction)

---

## ⚙️ How It Works

1. Load dataset
2. Clean and preprocess data
3. Encode categorical features
4. Feature engineering (car age)
5. Train multiple regression models
6. Compare performance
7. Save best models using `joblib`
8. Use UI for real-time prediction

---

## 🧪 How to Run

### 1. Clone Repository

```bash
git clone https://github.com/your-username/car-price-prediction-ml-multi-model
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Notebook / Script

```bash
jupyter notebook
```

or

```bash
python app.py
```

---

## 🖥️ UI Feature

The project includes a **simple user interface** that allows users to:

* Input car details
* Get instant price prediction
* Compare outputs from trained models

---

## 📊 Best Model

After evaluation:

🏆 **Random Forest Regressor** performed best in most cases based on:

* High R² Score
* Low RMSE

---

## 🚀 Future Improvements

* Deploy as Flask / FastAPI web app
* Improve UI with React or Streamlit
* Add deep learning regression model
* Deploy on cloud (Render / AWS)
* Add real-time car dataset updates

---

## 👨‍💻 Author

Developed as part of ML practice project for learning regression and model comparison.
