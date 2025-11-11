# import streamlit as st
# import pandas as pd
# import pickle

# # Load trained model
# with open('C:\\Users\\Tree\\Desktop\\Fraud Detection\\fraud_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# # Title
# st.title("💳 Fraud Detection App")
# st.write("Enter transaction details to predict fraud.")

# # Sidebar inputs for features
# st.sidebar.header("Transaction Features")
# def user_input_features():
#     data = {
#         'Time': st.sidebar.number_input('Time', value=406.0),
#         'V1': st.sidebar.number_input('V1', value=-2.312227),
#         'V2': st.sidebar.number_input('V2', value=1.951992),
#         'V3': st.sidebar.number_input('V3', value=-1.609851),
#         'V4': st.sidebar.number_input('V4', value=3.997906),
#         'V5': st.sidebar.number_input('V5', value=-0.522188),
#         'V6': st.sidebar.number_input('V6', value=-1.426545),
#         'V7': st.sidebar.number_input('V7', value=-2.537387),
#         'V8': st.sidebar.number_input('V8', value=1.391657),
#         'V9': st.sidebar.number_input('V9', value=-2.770089),
#         'V10': st.sidebar.number_input('V10', value=-2.772272),
#         'V11': st.sidebar.number_input('V11', value=3.202033),
#         'V12': st.sidebar.number_input('V12', value=-2.899907),
#         'V13': st.sidebar.number_input('V13', value=-0.595222),
#         'V14': st.sidebar.number_input('V14', value=-4.289254),
#         'V15': st.sidebar.number_input('V15', value=0.389724),
#         'V16': st.sidebar.number_input('V16', value=-1.140747),
#         'V17': st.sidebar.number_input('V17', value=-2.830056),
#         'V18': st.sidebar.number_input('V18', value=-0.016822),
#         'V19': st.sidebar.number_input('V19', value=0.416956),
#         'V20': st.sidebar.number_input('V20', value=0.126911),
#         'V21': st.sidebar.number_input('V21', value=0.517232),
#         'V22': st.sidebar.number_input('V22', value=-0.035049),
#         'V23': st.sidebar.number_input('V23', value=-0.465211),
#         'V24': st.sidebar.number_input('V24', value=0.320198),
#         'V25': st.sidebar.number_input('V25', value=0.044519),
#         'V26': st.sidebar.number_input('V26', value=0.177840),
#         'V27': st.sidebar.number_input('V27', value=0.261145),
#         'V28': st.sidebar.number_input('V28', value=-0.143276),
#         'Amount': st.sidebar.number_input('Amount', value=0.0)
#     }
#     features = pd.DataFrame(data, index=[0])
#     return features

# input_df = user_input_features()

# # Predict probability
# pred_proba = model.predict_proba(input_df)[:,1][0]

# # Apply threshold
# threshold = 0.4
# pred_class = int(pred_proba >= threshold)

# # Show results
# st.subheader("Prediction Result")
# if pred_class == 1:
#     st.error(f"⚠️ Fraud Detected! (Probability: {pred_proba:.4f})")
# else:
#     st.success(f"✅ Non-Fraud (Probability: {pred_proba:.4f})")

# st.write("Prediction threshold:", threshold)


import streamlit as st
import pandas as pd
import pickle

# Load trained pickle model
with open('fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

# App title
st.title("💳 Fraud Detection App")
st.write("Enter transaction details to predict fraud.")

# Sidebar: Dynamic input for features
st.sidebar.header("Transaction Features")

data = {
    'Time': st.sidebar.number_input('Time', value=406.0),
    **{f'V{i}': st.sidebar.number_input(f'V{i}', value=0.0) for i in range(1,29)},
    'Amount': st.sidebar.number_input('Amount', value=0.0)
}

input_df = pd.DataFrame([data])
feature_order = ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']
input_df = input_df[feature_order]

# Threshold slider
threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.4)

# Predict probability
pred_proba = model.predict_proba(input_df)[:,1][0]
pred_class = int(pred_proba >= threshold)

# Show results
st.subheader("Prediction Result")
if pred_class == 1:
    st.error(f"⚠️ Fraud Detected! (Probability: {pred_proba:.4f})")
else:
    st.success(f"✅ Non-Fraud (Probability: {pred_proba:.4f})")

st.write(f"Prediction threshold: {threshold}")
