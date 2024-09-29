# Comparative Analysis of Machine Learning Techniques for Hepatitis C Virus Infection Staging

For the given dataset, the goal is to perform data preprocessing and implement machine learning techniques to classify the stages of Hepatitis C Virus infection.

## Link to the Dataset:
[HCV Dataset](https://archive.ics.uci.edu/dataset/571/hcv+data)

## Dataset Overview:
The dataset contains laboratory and demographic values (such as age) for blood donors and Hepatitis C patients. The target variable includes different stages of Hepatitis C progression.

- **Dataset Characteristics**: Multivariate
- **Subject Area**: Health and Medicine
- **Associated Tasks**: Classification, Clustering
- **Feature Type**: Integer, Real
- **Instances**: 615
- **Features**: 12

### Dataset Information:
- **Instances**: Each row represents a patient.
- **Target Attribute**: Category (Blood donors vs. Hepatitis C stages: Hepatitis C, Fibrosis, Cirrhosis).
- **Missing Values**: Yes, some data points are missing.
  
### Additional Variable Information:
1. **X**: Patient ID (Identifier, not used for modeling)
2. **Category**: Diagnosis (0 = Blood Donor, 0s = Suspect Blood Donor, 1 = Hepatitis, 2 = Fibrosis, 3 = Cirrhosis)
3. **Age**: Age of the patient in years
4. **Sex**: Gender (f = Female, m = Male)
5. **ALB**: Albumin levels
6. **ALP**: Alkaline Phosphatase levels
7. **ALT**: Alanine Transaminase levels
8. **AST**: Aspartate Transaminase levels
9. **BIL**: Bilirubin levels
10. **CHE**: Cholinesterase levels
11. **CHOL**: Cholesterol levels
12. **CREA**: Creatinine levels
13. **GGT**: Gamma-Glutamyl Transferase levels
14. **PROT**: Protein levels

## Project Overview:

### 1. **Data Loading**:
   - Import the dataset (`hcvdat0.csv`).

### 2. **Data Preprocessing**:
   - **Data Cleaning**:
     - Remove unnecessary columns such as Patient ID.
     - Handle missing values (either by imputing or removing).
   - **Feature Encoding**:
     - Convert categorical columns like 'Sex' and 'Category' into numerical values using label encoding.

### 3. **Exploratory Data Analysis (EDA)**:
   - **Summary Statistics**: Generate descriptive statistics (mean, median, etc.).
   - **Visualizations**:
     - Plot histograms for numerical features to understand distributions.
     - Create a count plot for the target variable to observe class distributions.
   - **Correlation Analysis**:
     - Use a heatmap to visualize correlations between numerical features.

### 4. **Data Filtering and Preparation**:
   - Filter data based on specific Hepatitis C categories if necessary.
   - Split the dataset into features (X) and target (y).
   - **Train/Test Split**:
     - Split the data into training and test sets (e.g., 80/20 split).
   - **Feature Scaling**:
     - Apply scaling to normalize the feature values.

### 5. **Model Implementation**:
   - Implement the following models:
     - **Support Vector Machine (SVM)** for classification.
     - **K-Means Clustering** for grouping similar instances.
     - **Artificial Neural Network (ANN)** for complex pattern recognition.

### 6. **Model Training and Evaluation**:
   - **Training**: Train each model on the training dataset.
   - **Evaluation**:
     - Calculate accuracy on both the training and test sets.
     - Compare model performance using metrics like accuracy, precision, and recall.

### 7. **Results Visualization**:
   - Display model results, including training and test accuracy.
   - **Bar Chart Comparison**:
     - Compare the performance of SVM, K-Means, and ANN models using a bar chart.

## Results:

| Model  | Training Accuracy | Test Accuracy |
|--------|-------------------|---------------|
| **SVM** | 85.00%            | 80.00%        |
| **K-Means** | 58.33%         | 73.33%        |
| **ANN** | 96.67%            | 80.00%        |

### Key Observations:
- The **ANN** achieved the highest training accuracy, indicating strong learning capability, but its test accuracy suggests potential overfitting.
- **SVM** performed consistently well on both the training and test sets.
- **K-Means** showed lower training accuracy but improved test accuracy, which is typical for an unsupervised learning algorithm in classification tasks.

