# Stroke Prediction Classification Problem

## Overview
Stroke is a leading cause of death and disability worldwide, and early prediction of stroke risk can help prevent or reduce the severity of the disease. However, predicting stroke risk is a challenging task due to the complex interplay of various risk factors, such as age, gender, lifestyle, and medical history. Existing stroke prediction models have limitations in terms of accuracy, interpretability, and generalizability, and there is a need for more robust and reliable models.

## Data
**Dataset Source:** [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data)

**Dataset Information:** 
- 12 features
- 5110 rows

## Feature Explanation
Here are explanations for the dataset features:

| Feature          | Explanation                                             |
|------------------|---------------------------------------------------------|
| **id**           | Identifier                                              |
| **gender**       | Male, Female, or Other                                 |
| **age**          | Age                                                     |
| **hypertension** | 0 (No hypertension) or 1 (Hypertension)                |
| **heart_disease**| 0 (No heart disease) or 1 (Heart disease)              |
| **ever_married** | Yes or No                                              |
| **work_type**    | Job                                                    |
| **Residence_type**| Rural or Urban                                         |
| **avg_glucose_level** | Average glucose level                              |
| **bmi**          | Body Mass Index                                        |
| **smoking_status**| Smoking                                              |
| **stroke**       | 1 (Had a stroke) or 0 (No stroke)                     |

This dataset is intended for building a predictive model to determine stroke risk based on these attributes.

## Dependencies

To install the required dependencies, use the following command:

```bash
pip install -r requirements.txt
```

## Installation
Explain how to set up and install the project on a local machine:
[Python Virtual Environment Documentation](https://docs.python.org/3/library/venv.html)

## Resampling

Given the severe class imbalance in the stroke dataset, with a significantly larger number of samples in the "no stroke" class (0) compared to the "stroke" class (1), we employed resampling techniques to address this issue.

**Original Dataset:**

- Class 0 (No Stroke): 4,861 samples
- Class 1 (Stroke): 249 samples

We used the following resampling techniques to balance the classes and improve the predictive performance of our models:

1. **Over-sampling with SMOTE (Synthetic Minority Over-sampling Technique):**

   - SMOTE generates synthetic samples for the minority class (stroke) to match the number of samples in the majority class (no stroke).
   - It creates synthetic data points by interpolating between neighboring samples, effectively expanding the size of the minority class.
   
2. **Under-sampling with ENN (Edited Nearest Neighbors):**

   - ENN is an under-sampling technique that removes some of the noisy samples from the majority class.
   - It identifies samples in the majority class that are misclassified as the minority class based on their nearest neighbors and eliminates them.

The combination of SMOTE for oversampling and ENN for undersampling helps create a balanced dataset while eliminating noisy data points from the majority class. This balanced dataset is used as input for our machine learning models to enhance their predictive accuracy and robustness, particularly for detecting strokes. The improved class balance plays a crucial role in achieving better model performance, as evident from the results section.

## Model

For the stroke prediction task, we employed a variety of machine learning models and techniques to build a predictive model. Our aim was to find the model that offered the best performance for this classification problem.

**Models Used:**
- Random Forest Classifier
- Gradient Boosting Classifier
- AdaBoost Classifier
- Support Vector Machine (SVC)
- Logistic Regression
- Stochastic Gradient Descent (SGD) Classifier
- K-Nearest Neighbors (KNN) Classifier
- Gaussian Naive Bayes Classifier
- Decision Tree Classifier
- Multi-Layer Perceptron (MLP) Classifier
- Linear Discriminant Analysis (LDA)

## Model Evaluation

We conducted a comprehensive evaluation of various machine learning classifiers, optimizing their hyperparameters with GridSearchCV. The results are summarized in the table below. The ROC AUC score served as the primary evaluation metric for distinguishing positive and negative classes.

| Classifier               | Best Parameters                                        | ROC AUC Score     |
|--------------------------|-------------------------------------------------------|-------------------|
| RandomForest              | {'max_depth': None, 'n_estimators': 200}             | 0.9988            |
| GradientBoosting         | {'learning_rate': 0.1, 'n_estimators': 200}          | 0.9891            |
| SVM (Support Vector Machine) | {'C': 10, 'kernel': 'rbf'}                         | 0.9922            |
| KNeighbors               | {'n_neighbors': 7, 'weights': 'distance'}           | 0.9966            |
| MLP (Multi-layer Perceptron) | {'activation': 'relu', 'hidden_layer_sizes': (100, 50, 25)} | 0.9969 |

Based on the highest ROC AUC score of 0.9988, we selected the **RandomForest** classifier as the best model. Its exceptional performance in stroke prediction makes it the preferred choice.

**Visualizations:** 

- ROC Curve on Random Forest

![ROC Curve - Random Forest](https://github.com/RaflyQowi/End-to-end-ML-Stroke-Prediction/blob/main/image/ROC%20Curve%20Random%20Forest.png?raw=true)

## Choosing the Best Model
## Choosing the Best Model

In the final stage of model selection, we compared the performance of two RandomForest models: one without a specific threshold and one with a threshold of 0.19. Here's a concise summary of their evaluation results:

### RandomForest

|                    | Precision | Recall | F1-Score | Support |
|--------------------|-----------|--------|----------|---------|
| 0                  | 0.97      | 0.87   | 0.92     | 976     |
| 1                  | 0.12      | 0.39   | 0.19     | 46      |
| Accuracy           |           |        | 0.85     | 1022    |
| Macro Avg          | 0.55      | 0.63   | 0.55     | 1022    |
| Weighted Avg       | 0.93      | 0.85   | 0.88     | 1022    |

**Confusion Matrix**:

![Confusion Matrix Random Forest](https://github.com/RaflyQowi/End-to-end-ML-Stroke-Prediction/blob/main/image/Heatmap%20Random%20Forest.png)

### RandomForest with Threshold (0.19)

|                    | Precision | Recall | F1-Score | Support |
|--------------------|-----------|--------|----------|---------|
| 0                  | 0.99      | 0.72   | 0.83     | 976     |
| 1                  | 0.12      | 0.78   | 0.20     | 46      |
| Accuracy           |           |        | 0.72     | 1022    |
| Macro Avg          | 0.55      | 0.75   | 0.52     | 1022    |
| Weighted Avg       | 0.95      | 0.72   | 0.80     | 1022    |

**Confusion Matrix**:
![Confusion Matrix Random Forest with Threshold](https://github.com/RaflyQowi/End-to-end-ML-Stroke-Prediction/blob/main/image/Heatmap%20Random%20Forest.png)

Ultimately, we chose the RandomForest model with a threshold of 0.19. This model exhibits improved fairness in predicting true positives, as reflected by the increased recall and F1 score for the positive class. Although accuracy decreases slightly, the trade-off is well-justified by the enhancement in the model's ability to identify instances of stroke accurately.

## Results

Our model achieved promising results in stroke prediction, and we observed several key findings and insights:

- The combination of resampling techniques, SMOTE and ENN, helped mitigate class imbalance and improved model performance.
- The RandomForestClassifier exhibited the highest accuracy and F1-score among the models we tested.
- Visualizations, including ROC curves and precision-recall curves, helped us understand the trade-offs between precision and recall for different threshold levels.

For a more detailed summary of results, refer to the performance metrics and visualizations in the project documentation.

<!-- Continue with the rest of the README structure as previously provided -->
