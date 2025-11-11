# Credit Card Fraud Detection 🛡️

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/) 
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2.2-orange)](https://scikit-learn.org/) 


A machine learning project to detect fraudulent credit card transactions using a **Random Forest Classifier**.  
The dataset is highly imbalanced, so special techniques were applied to improve fraud detection performance.

---

## Dataset 

Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)  

- **Features**: Numerical and anonymized attributes (`V1`–`V28`, `Time`, `Amount`)  
- **Target**: `Class` (0 = Legitimate, 1 = Fraud)  
- **Challenge**: Imbalanced dataset – very few fraud cases  

---

## Installation ⚙️

```bash
pip install pandas numpy scikit-learn flask
```

## Approach 
Due to the imbalanced dataset, a Random Forest Classifier was used with class weighting and regularization:
```bash
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,        
    max_depth=10,            
    min_samples_leaf=10,     
    class_weight='balanced',
    random_state=42
)
```
```bash
class_weight='balanced'
```
<b>What it does</b>: 

This automatically calculates weights for you. It tells the model: "Hey, the 'fraud' class is rare, so when you make a mistake on a fraud transaction, penalize that mistake much more heavily than a mistake on a normal transaction."

<b>The result</b>:

 The model becomes much more sensitive to spotting fraud. It would rather accidentally flag a few extra normal transactions as suspicious (false positives) than miss actual fraud (false negatives).

<b>Regularization</b>:

The other parameters used don't directly handle imbalance, but they prevent overfitting and make the model more robust, which is especially important when working with imbalanced data:

```bash
max_depth=10:
```

 Prevents the trees from growing too deep and becoming overly complex. A complex model might just memorize the few fraud examples instead of learning general patterns.

```bash
min_samples_leaf=10: 
```

Ensures that each leaf node (the end of a branch) has at least 10 samples. This stops the tree from creating tiny, specific leaves that only contain one or two fraud cases, which is a classic overfitting trap with rare classes.

```bash
n_estimators=200:
```

 Uses 200 individual trees. More trees generally make the model more stable and reliable.</p>

<h4>Confusion Matrix </h4>
<img src="images\Confusion Matrix.png" alt="Feature Importance" style="width:600px; border:1px solid #ddd; border-radius:8px;">

<h4>Feature Importance </h4>
<img src="images\Feature Importance.png" alt="Feature Importance" style="width:600px; border:1px solid #ddd; border-radius:8px;">

<h2>Classification Report (Threshold = 0.4)</h2>

<table style="width:80%; border-collapse: collapse; text-align:center; margin-bottom:30px;">
  <tr style="background-color:#4CAF50; color:white;">
    <th>Class</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
    <th>Support</th>
  </tr>
  <tr style="background-color:#f2f2f2; color:black;">
    <td>0 (Legitimate)</td>
    <td>0.9998</td>
    <td>0.9992</td>
    <td>0.9995</td>
    <td>56864</td>
  </tr>
  <tr>
    <td>1 (Fraud)</td>
    <td>0.6641</td>
    <td>0.8673</td>
    <td>0.7522</td>
    <td>98</td>
  </tr>
  <tr style="background-color:#f2f2f2; color:black;">
    <td><b>Accuracy</b></td>
    <td colspan="3"></td>
    <td>0.9990</td>
  </tr>
  <tr>
    <td><b>Macro Avg</b></td>
    <td>0.8319</td>
    <td>0.9333</td>
    <td>0.8759</td>
    <td>56962</td>
  </tr>
  <tr style="background-color:#f2f2f2; color:black;">
    <td><b>Weighted Avg</b></td>
    <td>0.9992</td>
    <td>0.9990</td>
    <td>0.9991</td>
    <td>56962</td>
  </tr>
</table>



<h4>Streamlit Front End  </h4>
<img src="[images\streamlit frontend.png](https://github.com/khuzaima2182/FraudDetectPro/blob/main/images/Confusion%20Matrix.png)" alt="Feature Importance" style="width:600px; border:1px solid #ddd; border-radius:8px;">

<h4> Flask API Response </h4>
<img src="images\Flask API.png" alt="Feature Importance" style="width:600px; border:1px solid #ddd; border-radius:8px;">


