from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np

# Initialiser l'application Flask
app = Flask(__name__)

# Charger le modèle
model_path = "Modéle\\xgboost_optimized_model.pkl"
model = joblib.load(model_path)

# Page d'accueil
@app.route('/')
def home():
    return render_template('index.html')

# Page pour prédire
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Récupérer les valeurs du formulaire
            age = float(request.form['age'])
            job = int(request.form['job'])
            housing = request.form['housing']
            saving_accounts = request.form['saving_accounts']
            checking_account = float(request.form['checking_account'])
            credit_amount = float(request.form['credit_amount'])
            duration = int(request.form['duration'])
            purpose = request.form['purpose']

            # Encodage simple pour les données catégoriques
            housing_mapping = {'own': 0, 'rent': 1, 'free': 2}
            purpose_mapping = {'car': 0, 'furniture/equipment': 1, 'radio/TV': 2,
                               'domestic appliances': 3, 'repairs': 4, 'education': 5, 'business': 6, 'vacation/others': 7}
            saving_mapping = {'little': 0, 'moderate': 1, 'quite rich': 2, 'rich': 3}

            # Convertir en tableau d'entrée pour le modèle
            input_data = np.array([
                age,
                job,
                housing_mapping[housing],
                saving_mapping[saving_accounts],
                checking_account,
                credit_amount,
                duration,
                purpose_mapping[purpose]
            ]).reshape(1, -1)

            # Prédiction avec le modèle
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            # Renvoyer les résultats
            return render_template('result.html', prediction=prediction, probability=probability)

        except Exception as e:
            return render_template('predict.html', error=str(e))
    return render_template('predict.html')

# Lancer l'application
if __name__ == "__main__":
    app.run(debug=True)
