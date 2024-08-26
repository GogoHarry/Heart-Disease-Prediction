# Heart-Disease-Prediction

![hrt](https://github.com/user-attachments/assets/50089e39-3b7d-472a-947d-27bb74bf84a5)

## Project Overview

This project aims to develop a machine-learning model that predicts the likelihood of heart disease in individuals based on specific health-related features. The project involves data preprocessing, model training, and evaluation of multiple machine learning algorithms to identify the best-performing model.

## Dataset Description

The dataset used for this project contains health-related features such as age, cholesterol levels, blood pressure, and more, along with a target variable indicating the presence or absence of heart disease. The data was cleaned and preprocessed to ensure accuracy in model predictions.

**Dataset Features:**
- `Age`: Age of the individual
- `Sex`: Gender of the individual
- `ChestPainType`: Type of chest pain experienced
- `RestingBP`: Resting blood pressure
- `Cholesterol`: Serum cholesterol in mg/dl
- `FastingBS`: Fasting blood sugar
- `RestingECG`: Resting electrocardiographic results
- `MaxHR`: Maximum heart rate achieved
- `ExerciseAngina`: Exercise-induced angina
- `Oldpeak`: ST depression induced by exercise
- `ST_Slope`: Slope of the peak exercise ST segment
- `Target`: Presence of heart disease (1 = Yes, 0 = No)

## Libraries and Tools

The following libraries were used in this project:

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib**: For creating static, animated, and interactive visualizations.
- **Seaborn**: For statistical data visualization.
- **Scikit-learn**: For implementing machine learning models and evaluation metrics.

## Data Preprocessing

Data preprocessing steps included:

1. **Handling Missing Values and duplicates**: There were no missing values in the dataset, however, 1 duplicate was found and handled appropriately.
2. **Feature Scaling**: A Min-Max scaler was applied to normalize resting_blood_pressure', 'cholesterol', 'thalassemia', 'max_heart_rate_achieved features.
3. **Data Splitting**: The dataset was split into training and testing sets with a ratio of 80:20.

## Model Selection

Several machine learning models were trained and evaluated for this task:

- **Logistic Regression**
- **SGD Classifier**
- **K-Nearest Neighbors (KNN)**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**
- **Naive Bayes**
- **Decision Tree**

## Model Evaluation

The models were evaluated using the following metrics:

- **Accuracy**: The ratio of correctly predicted instances to the total instances.
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: The ratio of correctly predicted positive observations to all observations in the actual class.
- **ROC-AUC Score**: A performance measurement for classification problems at various threshold settings.
- **Confusion Matrix**: A summary of prediction results on a classification problem.

## Results

The best-performing model was the **SGD Classifier**, which achieved an accuracy of 92%, precision of 96%, and recall of 88%. The model also had an ROC-AUC score of 92, indicating strong predictive power.

```python
# 7 Machine Learning Algorithm will be applied to the dataset

classifiers = [[RandomForestClassifier(), 'Random Forest'],
              [KNeighborsClassifier(), 'K-Nearest Neighbors'],
              [SGDClassifier(), 'SGD Classifier'],
              [SVC(), 'SVC'],
              [GaussianNB(), 'Naive_Bayes'],
              [DecisionTreeClassifier(random_state = 42), "Decision Tree"],
              [LogisticRegression(), 'LogisticRegression']
             ]
```
```python
acc_list = {}
precision_list = {}
recall_list = {}
roc_list = {}

for classifier in classifiers:
    model = classifier[0]
    model.fit(X_train, y_train)
    model_name = classifier[1]
    
    pred = model.predict(X_test)
    
    a_score = accuracy_score(y_test, pred)
    p_score = precision_score(y_test, pred)
    r_score = recall_score(y_test, pred)
    roc_score = roc_auc_score(y_test, pred)
    
# Converting into a percentage
    acc_list[model_name] = ([str(round(a_score*100, 2))  + '%'])
    precision_list[model_name] = ([str(round(p_score*100, 2)) + '%'])
    recall_list[model_name] = ([str(round(r_score*100, 2)) + '%'])
    roc_list[model_name] = ([str(round(roc_score*100, 2)) + '%'])
    
    if model_name != classifier[-1][1]:
        print('')
```

![image](https://github.com/user-attachments/assets/64f19a54-4c1c-430b-af64-dbb83ede79f2)



## Conclusion

The SGD Classifier proved to be the most effective model in predicting heart disease based on the given features. Future improvements could include using a larger dataset, experimenting with feature engineering techniques, and exploring deep learning models for potentially better results.

Based on our analysis, doctors and hospital nurses aim to minimize the number of individuals who are predicted to have heart disease but do not have it. Misdiagnosing healthy patients as having heart disease could result in unnecessary and potentially harmful treatments. Therefore, it is crucial to focus on the precision metric to evaluate the accuracy of predictions and reduce the risk of incorrect diagnoses.

## How to Run the Project

To run this project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/GogoHarry/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook:
   ```bash
   jupyter notebook Heart_Disease_Modeling.ipynb
   ```

4. Follow the steps in the notebook to preprocess the data, train the models, and evaluate their performance.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

## License
This project is licensed under the MIT License - See the [LICENSE](LICENSE) file for details.
