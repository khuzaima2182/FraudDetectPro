from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load pickle model
with open('fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the correct feature order (must match training)
feature_order = ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']

# Default threshold
THRESHOLD = 0.4

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input from request
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No JSON data provided"}), 400

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Ensure correct column order
        input_df = input_df[feature_order]

        # Predict probability
        prob = model.predict_proba(input_df)[:,1][0]
        pred_class = int(prob >= THRESHOLD)

        # Return JSON response
        return jsonify({
            "pred_class": pred_class,
            "probability": prob,
            "threshold": THRESHOLD
        })

    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
