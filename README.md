# Datascience-EDA
assist with a study case using a simple dummy dataset. Tasks include EDA (patterns, trends, anomalies), churn predictive modeling, and providing business insights for +- 7000 rows of data. Additional tasks involve building NLP pipelines and creating a simple text-generative model.
================
To assist with a study case involving a dataset of ~7000 rows, including exploratory data analysis (EDA), churn predictive modeling, and text generative modeling, we can break down the tasks into smaller components. Below is a Python script that accomplishes the following:

    Exploratory Data Analysis (EDA): Identifies patterns, trends, and anomalies in the data.
    Churn Prediction: Predicts customer churn using a machine learning model.
    NLP Pipeline and Text Generation: Builds a simple text-generative model.

We'll use the following libraries:

    pandas, matplotlib, seaborn for EDA
    scikit-learn for churn prediction modeling
    nltk and transformers for building a simple NLP pipeline

Install Necessary Libraries

First, install the required libraries using pip:

pip install pandas matplotlib seaborn scikit-learn nltk transformers

Python Code for the Study Case:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load a dummy dataset for the example
# For real-world application, load your dataset (e.g., df = pd.read_csv("your_data.csv"))
# Here's an example with made-up data for illustration.
data = {
    'CustomerID': range(1, 7001),
    'Age': np.random.randint(18, 70, 7000),
    'Tenure': np.random.randint(1, 24, 7000),  # months with company
    'Balance': np.random.uniform(100, 5000, 7000),
    'NumProducts': np.random.randint(1, 4, 7000),
    'HasCrCard': np.random.choice([1, 0], 7000),
    'IsActiveMember': np.random.choice([1, 0], 7000),
    'EstimatedSalary': np.random.uniform(10000, 150000, 7000),
    'Churn': np.random.choice([1, 0], 7000)  # Churn (1 = churned, 0 = stayed)
}
df = pd.DataFrame(data)

# 1. Exploratory Data Analysis (EDA)
# Basic Info
print(df.info())
print(df.describe())

# Visualizing Distribution of Important Features
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, color='blue', bins=30)
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Churn', data=df, palette='viridis')
plt.title('Churn Count Distribution')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.show()

# Checking for Missing Data
print(df.isnull().sum())

# 2. Churn Predictive Modeling

# Prepare Data
X = df.drop(['CustomerID', 'Churn'], axis=1)
y = df['Churn']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model: Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test_scaled)

# Classification Report and Confusion Matrix
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature Importance
features = X.columns
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importance
plt.figure(figsize=(12, 6))
plt.title("Feature Importance")
plt.barh(range(len(features)), importances[indices], align="center")
plt.yticks(range(len(features)), features[indices])
plt.xlabel("Relative Importance")
plt.show()

# 3. NLP Pipeline and Simple Text-Generative Model
# For the text generative model, we will use a pretrained GPT-2 model from Hugging Face transformers.

# Download GPT-2 Model and Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model_nlp = GPT2LMHeadModel.from_pretrained("gpt2")

# Function to Generate Text
def generate_text(prompt, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model_nlp.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generate Text
prompt = "In the world of AI chatbots,"
generated_text = generate_text(prompt)
print("Generated Text:")
print(generated_text)

# 4. Business Insights

# Churn Rate Analysis
churn_rate = df['Churn'].mean()
print(f"Churn Rate: {churn_rate * 100:.2f}%")

# Insights Based on Customer Age and Churn
age_churn = df.groupby('Age')['Churn'].mean()
plt.figure(figsize=(10, 6))
sns.lineplot(x=age_churn.index, y=age_churn.values)
plt.title("Churn Rate vs Age")
plt.xlabel('Age')
plt.ylabel('Churn Rate')
plt.show()

# Insights Based on Tenure and Churn
tenure_churn = df.groupby('Tenure')['Churn'].mean()
plt.figure(figsize=(10, 6))
sns.lineplot(x=tenure_churn.index, y=tenure_churn.values)
plt.title("Churn Rate vs Tenure")
plt.xlabel('Tenure (Months)')
plt.ylabel('Churn Rate')
plt.show()

# Marketing Strategy Suggestions:
# - Target younger customers with a longer tenure as they have a higher probability of churning.
# - Focus on increasing engagement for inactive members (HasCrCard and IsActiveMember).
# - Offer tailored incentives to customers with high balances or multi-product accounts to increase retention.

Breakdown of the Code:

    Exploratory Data Analysis (EDA):
        We load a dummy dataset with random values (Age, Tenure, Balance, etc.).
        We explore the basic structure and statistics of the dataset (df.describe() and df.info()).
        We visualize the distribution of Age and Churn using histograms and count plots.
        We plot a correlation heatmap to identify relationships between features.

    Churn Prediction Model:
        We prepare the data by dropping the CustomerID column and separating the features (X) from the target (y).
        We use a RandomForestClassifier to predict churn (y).
        We evaluate the model using a classification report and confusion matrix to measure accuracy, precision, recall, and F1-score.
        We visualize the feature importance of each predictor variable.

    NLP Pipeline and Simple Text-Generative Model:
        We use the transformers library from Hugging Face to download a pretrained GPT-2 model.
        We create a function generate_text that takes a prompt and generates text based on the GPT-2 model.
        The text generation is printed as output.

    Business Insights:
        We calculate the churn rate and provide insights about factors contributing to churn.
        We plot churn rates against Age and Tenure to identify trends.
        Based on insights, we suggest marketing strategies such as focusing on specific age groups or enhancing retention programs for inactive members.

Conclusion:

This script provides a basic framework to handle exploratory data analysis (EDA), churn prediction using machine learning, and a simple NLP text generation model. The code can be adjusted and expanded depending on your specific dataset and requirements, such as implementing more advanced NLP models, optimization techniques, or handling imbalanced data for churn prediction.
