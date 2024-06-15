# Comparative Analysis of Machine Learning Techniques for Hepatitis C Virus Infection Staging![image](https://github.com/Roopa2002/hcv/assets/121749709/34ea0f31-c6ac-4bcc-a087-95305d2fffc2)


For the given dataset, perform the data-preprocessing

Link to the Dataset:

https://archive.ics.uci.edu/dataset/571/hcv+data

The data set contains laboratory values of blood donors and Hepatitis C patients and demographic values like age.

Dataset Characteristics:Multivariate

Subject Area:Health and Medicine

Associated Tasks:Classification, Clustering

Feature Type:Integer, Real
# Instances:615
# Features:12

Dataset Information:
What do the instances in this dataset represent?
Instances are patients

Additional Information:
The target attribute for classification is Category (blood donors vs. Hepatitis C, including its progress: 'just' Hepatitis C, Fibrosis, Cirrhosis).

Has Missing Values?
Yes

Additional Variable Information
All attributes except Category and Sex are numerical. The laboratory data are the attributes 5-14.
	 1) X (Patient ID/No.)
	 2) Category (diagnosis) (values: '0=Blood Donor', '0s=suspect Blood Donor', '1=Hepatitis', '2=Fibrosis', '3=Cirrhosis')
	 3) Age (in years)
	 4) Sex (f,m)
	 5) ALB
	 6) ALP
	 7) ALT
	 8) AST
	 9) BIL
	10) CHE
	11) CHOL
	12) CREA
	13) GGT
	14) PROT

Project Overview:

1. **Data Loading**: Importing the dataset (`hcvdat0.csv`).

2. **Data Preprocessing**:
   - Data Cleaning (removing unnecessary columns, handling missing values).
   - Feature Encoding (encoding 'Sex' and 'Category' columns).

3. **Exploratory Data Analysis (EDA)**:
   - Generating summary statistics.
   - Visualizing distributions of features (histograms).
   - Visualizing the target variable distribution (count plot).
   - Correlation analysis (heatmap).

4. **Data Filtering and Preparation**:
   - Filtering data for specific HCV categories.
   - Splitting into features (X) and target (y).
   - Splitting into training and test sets.
   - Feature scaling.

5. **Model Implementation**:
   - Support Vector Machine (SVM).
   - K-Means Clustering.
   - Artificial Neural Network (ANN).

6. **Model Training and Evaluation**:
   - Training each model on the training set.
   - Evaluating accuracy on both training and test sets.

7. **Results Visualization**:
   - Displaying accuracy results.
   - Bar chart comparison of model performances.


Results:

SVM Training Accuracy: 85.00%
SVM Test Accuracy: 80.00%
K-Means Training Accuracy: 58.33%
K-Means Test Accuracy: 73.33%
ANN Training Accuracy: 96.67%
ANN Test Accuracy: 80.00%
