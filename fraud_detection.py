import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns

# Load and sample data
df = pd.read_csv("creditcard.csv")
df = df.sample(frac=0.1, random_state=42)  # Use 10% for speed

# Normalize 'Amount' and drop 'Time'
df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
df.drop(['Time'], axis=1, inplace=True)

# Split features and labels
X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Show class balance before resampling
print("Before Oversampling:\n", y_train.value_counts())

# Oversample minority class
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

# Show class balance after resampling
print("\nAfter Oversampling:\n", y_train_res.value_counts())

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_res, y_train_res)
y_pred_lr = lr.predict(X_test)

# Balanced, fast Random Forest
rf = RandomForestClassifier(
    n_estimators=20,     # Enough trees for good performance
    max_depth=10,        # Prevent overfitting, fast training
    random_state=42,
    n_jobs=-1            # Use all CPU cores
)
rf.fit(X_train_res, y_train_res)
y_pred_rf = rf.predict(X_test)

# Evaluation function
def evaluate_model(y_true, y_pred, model_name):
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n{cm}\n")

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Evaluate models
evaluate_model(y_test, y_pred_lr, "Logistic Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest (Balanced)")

# Final output
print("\n✅ Done! This version gives fast training + good accuracy.")
