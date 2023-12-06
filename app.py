from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model from pickle file
file_path = 'hi_random_forest.pickle'
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Extract model from loaded data
model = data['model']
scaler = data['scaler']
hi_map = data['hi_map']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        # Dapatkan data input dari formulir
        kapasitas = float(request.form['kapasitas'])
        prabayar_daya = float(request.form['prabayar_daya'])
        prabayar_transaksi = float(request.form['prabayar_transaksi'])
        prabayar_kwh = float(request.form['prabayar_kwh'])
        paska_daya = float(request.form['paska_daya'])
        paska_plgn = float(request.form['paska_plgn'])
        paska_kwh = float(request.form['paska_kwh'])

        # input data to list
        input_data = [[kapasitas, prabayar_daya, prabayar_transaksi, prabayar_kwh, paska_daya, paska_plgn, paska_kwh]]

        # Use dataframe to add feature name/column
        X = pd.DataFrame(
            input_data,
            columns=['kapasitas', 'prabayar_daya', 'prabayar_transaksi', 'prabayar_kwh', 'paska_daya', 'paska_plgn', 'paska_kwh']
        )

        # transform using fitted StandardScaler
        x_scaled = scaler.transform(X)

        # use model to predict
        y_preds = model.predict(x_scaled)
        prediction = y_preds[0]

        print('Prediction Result:', prediction)
        print('Kapasitas:', kapasitas)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
