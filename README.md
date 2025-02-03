# Customer Churn Prediction using ANN

![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow%20%7C%20Keras-blue)

## 📌 Overview
This project focuses on predicting customer churn using an **Artificial Neural Network (ANN)** built with **Keras** and **TensorFlow**. The dataset used is from [Kaggle - Credit Card Customer Churn Prediction](https://www.kaggle.com/datasets/rjmanoj/credit-card-customer-churn-prediction/data). The trained model achieves an **accuracy of 86%**.

## 📊 Dataset
The dataset contains information about credit card customers, including demographics, account information, and transaction history. The goal is to predict whether a customer will churn (leave the service) or not.

### 🔹 Features:
- **Customer_Age**
- **Gender**
- **Dependent_count**
- **Education_Level**
- **Marital_Status**
- **Income_Category**
- **Card_Category**
- **Months_on_book** (tenure of the customer)
- **Total_Relationship_Count**
- **Total_Revolving_Bal**
- **Avg_Open_To_Buy**
- **Total_Amt_Chng_Q4_Q1** (change in transaction amount)
- **Total_Trans_Ct**
- **Attrition_Flag** (Target Variable: Churn or Not)

## 📌 Technologies Used
- **Python** 🐍
- **TensorFlow & Keras** 🔥
- **NumPy, Pandas, Matplotlib, Seaborn** 📊
- **Scikit-learn** 🔍
- **Google Colab / Jupyter Notebook**

## 🏗 Model Pipeline
1. **Data Preprocessing**
   - Load dataset using Pandas
   - Handle missing values (if any)
   - Perform encoding of categorical variables
   - Scale numerical features
2. **Model Building**
   - ANN with multiple dense layers and ReLU activation
   - Dropout layers to prevent overfitting
   - Sigmoid activation in the output layer
3. **Model Training & Evaluation**
   - Train with Adam optimizer & binary cross-entropy loss
   - Evaluate using accuracy, precision, recall, and F1-score

## 📌 Installation & Usage
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/customer-churn-ann.git
   cd customer-churn-ann
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**
   ```bash
   jupyter notebook churn-prediction-using-ann.ipynb
   ```

## 📈 Results
- **Accuracy:** 86%
- **Precision, Recall, and F1-Score**: Evaluated using a classification report
- **Confusion Matrix**: Visualized for performance assessment

## 📌 Project Structure
```
📂 customer-churn-ann
│── dataset/
│── models/
│── notebooks/
│── churn-prediction-using-ann.ipynb
│── train.py
│── preprocess.py
│── requirements.txt
│── README.md
```

## 📜 Future Improvements
- Hyperparameter tuning for better accuracy
- Implementing other deep learning architectures (e.g., CNN, LSTM)
- Deploying the model using Flask/Django

## 📌 References
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Kaggle Dataset](https://www.kaggle.com/datasets/rjmanoj/credit-card-customer-churn-prediction/data)

🚀 **Happy Learning!**

