import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('hcvdat0.csv')
# Display the first few rows of the dataset

print(data.head())

data.drop('Unnamed: 0', axis=1, inplace=True)

# Check for missing values in the dataset
missing_values = data.isnull().sum()

print(missing_values)

# Impute missing values using the median of the respective columns
for column in missing_values.index:
    if missing_values[column] > 0:
        data[column].fillna(data[column].median(), inplace=True)

# Verify if missing values have been handled
missing_values_after_imputation = data.isnull().sum()

print(missing_values_after_imputation)

# Encode the 'Sex' column as binary variables
data['Sex'] = data['Sex'].map({'m': 1, 'f': 0})

# Verify the transformation
data['Sex'].head()

# Inspect unique values in the 'Category' column
unique_categories = data['Category'].unique()
print(unique_categories)

#Encode the 'Category' column with ordinal encoding
category_encoding = {
    '0=Blood Donor': 0,
    '0s=suspect Blood Donor': 1,
    '1=Hepatitis': 2,
    '2=Fibrosis': 3,
    '3=Cirrhosis': 4
}

data['Category'] = data['Category'].map(category_encoding)

# Verify the transformation
print(data[['Category']].head())


# Generate summary statistics for the numerical features
summary_statistics = data.describe()
category_counts = data['Category'].value_counts()
# Let's take a look at the summary statistics
print(summary_statistics)

print(category_counts)
# Set the aesthetics for the plots
sns.set_style("whitegrid")

# Plot histograms for the numerical features
num_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
num_features.remove('Category')  # Remove the target variable from the list for separate plotting

plt.figure(figsize=(15, 10))
for i, feature in enumerate(num_features, 1):
    plt.subplot(3, 4, i)
    sns.histplot(data[feature], kde=False)
    plt.tight_layout()
    plt.title(f'Distribution of {feature}')

# Plot the distribution of the target variable 'Category'
plt.figure(figsize=(6, 4))
sns.countplot(x='Category', data=data)
plt.title('Distribution of Category')
plt.show()

# Calculate the correlation matrix
corr_matrix = data.corr()

# # Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix of HCV Data Features')
plt.show()

# Filtering for C1, C2, C3 categories
data_filtered = data[data['Category'].isin([2, 3, 4])]

# Splitting the data into features and target
X = data_filtered.drop('Category', axis=1)
y = data_filtered['Category']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Implementing the models
# SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_train_accuracy = accuracy_score(y_train, svm_model.predict(X_train))
svm_test_accuracy = accuracy_score(y_test, svm_model.predict(X_test))

# K-Means Clustering
kmeans = KMeans(n_clusters=6, random_state=2302, n_init='auto')
kmeans.fit(X_train)
y_train_pred_kmeans = kmeans.predict(X_train)
y_test_pred_kmeans = kmeans.predict(X_test)
kmeans_train_accuracy = accuracy_score(y_train, y_train_pred_kmeans)
kmeans_test_accuracy = accuracy_score(y_test, y_test_pred_kmeans)

# ANN
ann_model = MLPClassifier(hidden_layer_sizes=(100,), random_state=42)
ann_model.fit(X_train, y_train)
ann_train_accuracy = accuracy_score(y_train, ann_model.predict(X_train))
ann_test_accuracy = accuracy_score(y_test, ann_model.predict(X_test))

accuracy_results = {
    "SVM Training Accuracy": svm_train_accuracy,
    "SVM Test Accuracy": svm_test_accuracy,
    "K-Means Training Accuracy": kmeans_train_accuracy,
    "K-Means Test Accuracy": kmeans_test_accuracy,
    "ANN Training Accuracy": ann_train_accuracy,
    "ANN Test Accuracy": ann_test_accuracy
}

for model, accuracy in accuracy_results.items():
    print(f"{model}: {accuracy * 100:.2f}%")

# Graphing the results
xLabels = ["SVM Training", "SVM Test", "K-Means Training", "K-Means Test", "ANN Training", "ANN Test"]
values = [svm_train_accuracy, svm_test_accuracy, kmeans_train_accuracy, kmeans_test_accuracy, ann_train_accuracy, ann_test_accuracy]

plt.bar(xLabels, values, width=.6)
plt.xlabel("Algorithm/Set")
plt.ylabel("Accuracy")
plt.title("ML Results")
plt.show()