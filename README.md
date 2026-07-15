 # Heart Disease Risk Prediction

A machine learning web app that predicts the risk of heart disease based on a patient's health information.

## 🎯 About This Project
Heart disease is one of the leading causes of death worldwide. This project uses machine learning to predict whether a person is at risk of heart disease, based on medical data like age, blood pressure, cholesterol, and other health indicators.

## 🔧 Tools and Technologies Used:
1.Python
2.Pandas and NumPy (data handling)
3.Scikit-learn (machine learning models)
4.Streamlit (web app)

## 📊 Dataset
The UCI Heart Disease Dataset was used. It contains 920 patient records from four hospitals: Cleveland, Hungary, Switzerland, and VA Long Beach.

## 🚀 Project Steps
1. **Data Cleaning**: Handled missing values in the dataset.
2. **Feature Encoding**: Converted text data (like Male/Female) into numbers so the model can understand it.
3. **Model Training**: Trained and compared 5 different models: Logistic Regression, KNN, Decision Tree, Random Forest, and SVM.
4. **Model Selection**: Used cross-validation to pick the best and most reliable model.
5. **Final Model**: SVM (Support Vector Machine) gave the best results.
6. **Deployment**: Built a simple web app using Streamlit so anyone can use the model.

## 📈 Model Performance
| Metric | Score |
|---|---|
| Accuracy | 85.3% |
| Precision | 83.8% |
| Recall | 91.2% |
| F1 Score | 87.3% |
| ROC-AUC | 92.2% |

## 💻 How to Run This Project
1. Install the required libraries: