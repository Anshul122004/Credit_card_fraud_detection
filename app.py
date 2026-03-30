from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('credit_card_fraud_rf_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get all required features
        features = ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']
        data = {f: float(request.form.get(f, 0)) for f in features}
        
        # Create DataFrame with all features in correct order
        input_data = pd.DataFrame([data.values()], columns=features)
        
        # Make prediction with adjusted threshold (0.3 instead of 0.5)
        probability = model.predict_proba(input_data)
        fraud_prob = probability[0][1]
        prediction = 1 if fraud_prob > 0.3 else 0  # More sensitive threshold
        
        return jsonify({
            'prediction': prediction,
            'probability_fraud': float(fraud_prob),
            'probability_legitimate': float(probability[0][0])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)