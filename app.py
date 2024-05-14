# Import necessary libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
url = 'https://raw.githubusercontent.com/Yashchaudhary05/pima/main/aims_diabetes_data.csv'
df = pd.read_csv(url)
print()
print("< < < < < < < < < < < < < < <  Data Imported Succesfully > > > > > > > > > > > > > > > ")
print()

# Preprocess the data
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train different models
models = {
    
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} accuracy: {accuracy:.2f}')

# Create a user interface to input data and get predictions
def predict_diabetes(data):
    # Convert input data to a numpy array
    data_array = np.asarray(data)

    # Data Standardisation
    data_scaled = scaler.transform(data_array.reshape(1, -1))

    # Making predictions using different models
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(data_scaled)[0]

    return predictions


# Creating a Flask app
from flask import Flask, request, render_template


app = Flask(__name__)

#  main route
@app.route('/', methods=['GET', 'POST'])
def index():
    #user input
    if request.method == 'POST':
        data = [float(value) for value in request.form.values()]
        predictions = predict_diabetes(data)
        return render_template('results.html', predictions=predictions)

    # Pass X to the template context
    return render_template('index.html', X=X.columns)


#  results route
@app.route('/results')
def results():
    # Get predictions
    predictions = request.args.get('predictions')

    # Render results 
    return render_template('results.html', predictions=predictions)

# Run app
if __name__ == '__main__':
    app.run(debug=True)

#python  app.py to run