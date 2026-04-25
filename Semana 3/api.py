from flask import Flask, request, redirect
from flask_restx import Api, Resource, fields
import pandas as pd
import joblib
import os

# === Cargar modelo ===
modelo = joblib.load(os.path.join(os.path.dirname(__file__), 'modelo_catboost.pkl'))
# === Variables originales esperadas ===
variables_originales = [
    'duration_ms', 'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'explicit', 'key', 'mode', 'time_signature',
    'artists', 'album_name', 'track_name', 'track_genre'
]

variables_categoricas = ['artists', 'album_name', 'track_name', 'track_genre']

# === Configuración de Flask ===
app = Flask(__name__)
api = Api(app, version='1.0', title='🎵 API de Popularidad Spotify',
          description='Predicción de popularidad con CatBoost',
          doc='/')

@app.route('/')
def index():
    return redirect('/docs')

ns = api.namespace('predict', description='Operaciones de predicción')
# === Modelo de entrada ===\n",
entrada_modelo = api.model('EntradaModelo', {
    var: fields.String(required=True, description=f'Valor de {var}') if var in variables_categoricas
    else fields.Float(required=True, description=f'Valor de {var}')
    for var in variables_originales
})

# === Endpoint POST: predicción vía JSON ===
@ns.route('/')
class Prediccion(Resource):
    @ns.expect(entrada_modelo)
    def post(self):
        datos = [request.json[var] for var in variables_originales]
        df = pd.DataFrame([datos], columns=variables_originales)
        pred = modelo.predict(df)[0]
        return {'popularidad_estimada': round(pred, 2)}, 200

# === Endpoint GET: predicción por URL (opcional) ===
@ns.route('/params')
@ns.doc(params={var: f'Valor de {var}' for var in variables_originales})
class PrediccionConParametros(Resource):
    def get(self):
        try:
            datos = [
                request.args.get(var) if var in variables_categoricas
                else float(request.args.get(var, 0))
                for var in variables_originales
            ]
            df = pd.DataFrame([datos], columns=variables_originales)
            pred = modelo.predict(df)[0]
            return {'popularidad_estimada': round(pred, 2)}, 200
        except Exception as e:
            return {'error': str(e)}, 400

# === Ejecutar la aplicación ===
if __name__ == '__main__':
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)

