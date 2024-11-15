from flask import Flask, request, jsonify, render_template
from functions import  calculate_turnover_risk, combine_probabilities

import joblib

# Charger le modèle de machine learning
model = joblib.load('rf_model.pkl')

# Fonction pour appliquer le modèle de machine learning
def apply_model(data, model):
    features = [
        data['satisfaction'], data['evaluation'], data['projectCount'],
        data['averageMonthlyHours'], data['yearsAtCompany'], data['workAccident'],
        data['promotion'], data['department'], data['salary'], data['gender']
    ]
    features = [features]  # Format attendu pour le modèle
    model_probability = model.predict_proba(features)[0][1] * 100  # Probabilité de turnover
    return model_probability

# Création de l'application Flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer les données de la requête
    data = request.get_json()
    print(f"Received data: {data}")  # Afficher les données reçues pour débogage

    # Appliquer le modèle pour obtenir la probabilité
    model_probability = apply_model(data, model)

    # Calculer le risque ajusté
    adjusted_probability = calculate_turnover_risk(
        data['satisfaction'], data['evaluation'], data['projectCount'],
        data['averageMonthlyHours'], data['yearsAtCompany'], data['workAccident'],
        data['promotion'], data['department'], data['salary'], data['gender']
    )

    # Combiner les deux probabilités
    combined_probability = combine_probabilities(model_probability, adjusted_probability)

    # Retourner les résultats sous forme JSON
    return jsonify({
        "model_probability": model_probability,
        "adjusted_probability": adjusted_probability,
        "combined_probability": combined_probability
    })

# Exécuter l'application Flask
if __name__ == '__main__':
    app.run(debug=True)
