# Steel Plate Fault Classification

## 1. Problem Statement

Steel manufacturing industries need to identify surface defects in steel plates to maintain product quality and reduce manufacturing defects.

This project uses Machine Learning classification algorithms to identify steel plate fault categories based on measurable surface characteristics.

The objective is to implement and compare five classification models and identify the best-performing model using multiple evaluation metrics.

## 2. Dataset Description

- **Dataset Name:** Steel Plates Faults
- **Source:** UCI Machine Learning Repository
- **Dataset URL:** https://archive.ics.uci.edu/dataset/198/steel+plates+faults
- **Total Instances:** 1,941
- **Input Features:** 27
- **Classification Classes:** 7
- **Problem Type:** Multiclass Classification

The dataset contains numerical measurements describing steel plate characteristics. The target variable `Fault_Type` represents the defect category.

The seven fault categories are:

- Pastry
- Z_Scratch
- K_Scratch
- Stains
- Dirtiness
- Bumps
- Other_Faults

The dataset was divided into training and testing sets using an 80:20 stratified split with `random_state=42`.

## 3. GitHub Repository

**Repository URL:** [https://github.com/PavansaiPodichervu/Steel-Plate-Fault-Classification]

The repository contains:

- Python source code
- Dataset preprocessing script
- Model training script
- Model evaluation script
- Streamlit application
- Saved trained classification models
- Test dataset
- Model comparison results
- Requirements file

### Project Structure

```text
steel-plate-classification/
├── app.py
├── data_preprocessing.py
├── train_models.py
├── evaluate_models.py
├── run_project.py
├── requirements.txt
├── README.md
├── data/
│   ├── steel_plates_faults.csv
│   ├── train_data.csv
│   └── test_data.csv
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   └── feature_names.joblib
└── model_comparison.csv
```

## 4. Machine Learning Models

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

StandardScaler was applied to Logistic Regression, KNN, and SVM through machine learning pipelines.

### Evaluation Metrics

The models were evaluated using:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

Precision, Recall, and F1 Score were calculated using macro averaging to give equal importance to all seven fault classes.

AUC was calculated using the One-vs-Rest multiclass strategy with macro averaging.

### Model Comparison

The following results were obtained from the model evaluation experiment.


### Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7275 | 0.9379 | 0.7613 | 0.7305 | 0.7418 | 0.6487 |
| Decision Tree | 0.7429 | 0.8534 | 0.7531 | 0.7559 | 0.7530 | 0.6688 |
| KNN | 0.7275 | 0.9184 | 0.7458 | 0.7482 | 0.7416 | 0.6548 |
| Naive Bayes | 0.4524 | 0.8386 | 0.4315 | 0.4795 | 0.3830 | 0.3789 |
| SVM | 0.7455 | 0.9361 | 0.7910 | 0.7334 | 0.7551 | 0.6701 |


### Model Performance Observations

#### Logistic Regression

Logistic Regression achieved an accuracy of 72.75% and a macro F1 score of 0.7418. It demonstrated strong AUC performance, indicating good class discrimination. However, its performance was slightly lower than SVM and Decision Tree on the overall classification metrics.

#### Decision Tree

Decision Tree achieved an accuracy of 74.29% and a macro F1 score of 0.7530. It demonstrated balanced precision and recall and achieved the highest macro recall among the five models. Its AUC was comparatively lower, indicating differences in probability-based class discrimination.

#### K-Nearest Neighbors

KNN achieved an accuracy of 72.75% and a macro F1 score of 0.7416. It provided reasonably balanced precision and recall. Its performance was competitive with Logistic Regression but lower than Decision Tree and SVM.

#### Naive Bayes

Naive Bayes achieved an accuracy of 45.24% and a macro F1 score of 0.3830. It performed substantially worse than the other classifiers. This may be related to its conditional independence assumption and the characteristics of the dataset.

#### Support Vector Machine

SVM achieved the highest accuracy of 74.55%, macro precision of 0.7910, macro F1 score of 0.7551, and MCC of 0.6701. It also demonstrated strong AUC performance of 0.9361.

### Overall Winner

**Best-performing model: Support Vector Machine (SVM)**

SVM was selected as the overall best-performing model based on its highest accuracy, macro precision, macro F1 score, and MCC among the evaluated classifiers.

Decision Tree achieved the highest macro recall, while Logistic Regression achieved the highest AUC. Nevertheless, SVM provided the strongest overall performance across the principal classification metrics.

## 5. Streamlit Application

The Streamlit application provides an interactive interface for model comparison and test dataset evaluation.

### Application Features

- Model comparison table
- Performance comparison chart
- Classification model selection
- Test dataset CSV upload
- Evaluation metric display
- Confusion matrix visualization
- Classification report
- Prediction results download

The application loads previously trained models from the `model/` directory.

Uploaded test datasets are evaluated using the selected trained model without retraining.

### Running the Application Locally

Clone the repository:

Clone the repository:

```bash
git clone https://github.com/PavansaiPodichervu/Steel-Plate-Fault-Classification.git
cd Steel-Plate-Fault-Classification
```

Alternatively, using GitHub CLI:

```bash
gh repo clone PavansaiPodichervu/Steel-Plate-Fault-Classification
cd Steel-Plate-Fault-Classification
```


Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the complete project:

```bash
python run_project.py
```

Alternatively, launch the Streamlit application:

```bash
streamlit run app.py
```

## 6. Streamlit Deployment

**Live Application URL:** [https://steel-plate-fault-classification-qx9angjwpwdxnzv4u6jx8v.streamlit.app/]

The deployed application allows users to select a trained classification model, upload test data, and evaluate classification performance interactively.

## 7. Conclusion

This project implemented and compared five Machine Learning classification algorithms for steel plate fault identification using the UCI Steel Plates Faults dataset.

The models were evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and MCC.

The experimental results showed that Support Vector Machine achieved the strongest overall classification performance, while Decision Tree demonstrated the highest macro recall.

The project also provides an interactive Streamlit application that enables model selection and evaluation of uploaded test datasets without retraining.

The results demonstrate the importance of evaluating classification models using multiple metrics rather than relying exclusively on accuracy.
