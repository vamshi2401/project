from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle 


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


def predict_from_csv(csv_file):
    # Load the trained model
    model = pickle.load(open('model_1.pickle', 'rb'))

    # Load the CSV file
    df = pd.read_csv(csv_file)
    x=df.iloc[:,:-1]

    # Make predictions
    predictions = model.predict(x)

    return predictions.tolist()

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a CSV file is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the file is CSV
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and file.filename.endswith('.csv'):
        predictions = predict_from_csv(file)
        return jsonify({'predictions': predictions})
    else:
        return jsonify({'error': 'File must be in CSV format'})

if __name__ == '__main__':
    app.run(debug=True)
