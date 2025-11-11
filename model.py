import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report, roc_auc_score

df = pd.read_csv('creditcard.csv')
print(df.head())
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,           # limit depth
    min_samples_leaf=10,     # require at least 5 samples in leaf
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)
# Predict labels
y_test_pred = model.predict(X_test)

# Predict probabilities for threshold tuning
y_test_proba = model.predict_proba(X_test)[:, 1]
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print("=== Classification Report ===")
print(classification_report(y_test, y_test_pred, digits=4))

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

print("\nROC-AUC Score:", roc_auc_score(y_test, y_test_proba))
threshold = 0.4
y_test_pred_thresh = (y_test_proba >= threshold).astype(int)

print("\n=== Classification Report (Threshold = 0.4) ===")
print(classification_report(y_test, y_test_pred_thresh, digits=4))
import pickle

# Save the model to a file
with open('fraud_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved as 'fraud_model.pkl'")
