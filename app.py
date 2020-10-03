from flask import Flask, url_for, render_template, request, jsonify
import joblib
import pandas as pd
from utils import FeatureSelector
import json


app = Flask(__name__)

model = joblib.load('smoke_model.joblib')

smokeData = pd.read_csv('smokeData.csv')

df = smokeData[['Age','Sex','Grade','Race','RuralUrban','ever_cigarettes','ever_cigars_cigarillos_or',
                'Ever_chewing_tobacco_snuf']].copy()

# Dropping missing values
df.dropna(subset=['ever_cigarettes'], inplace = True)
df.dropna(subset=['Sex'], inplace = True)

# Filling in missing values
df.fillna(value = {'Race':'White'}, inplace = True)
df.fillna(value = {'Age': 14, 'Grade': 4}, inplace = True)

# drop all respondents under age of 12 who have smoked before
df = df.drop(df[(df['Age'] < 12) & (df['ever_cigarettes'] == True)].index)
# need to reset index b/c we dropped rows or some rows will just be missing and it'll create errors
df.reset_index(drop = True, inplace = True)

# replace White with white as well as urban and rural so it's consistent
df = df.replace('White', 'white')
df = df.replace('Rural', 'rural')
df = df.replace('Urban', 'urban')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods = ['POST'])
def predict():
    age = request.form['age']
    sex = request.form['sex']
    race = request.form['race']
    rurality = request.form['rurality']

    X = pd.DataFrame([[int(age), sex, race, rurality]], columns = ['Age', 'Sex', 'Race', 'RuralUrban'])

    y_pred = model.predict(X)

    return render_template('predict.html', prediction = y_pred)
    

@app.route('/api', methods=['GET'])
def api_id():

    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."

    y_pred = model.predict(df.iloc[[id], :])

    result = (df.iloc[[id], :]).copy()
    if y_pred[0] == 0:
        result['Ever_Smoked'] = ['No']
    else:
        result['Ever_Smoked'] = ['Yes']
    
    result.insert(0, 'id', result.index)

    jsonfile = jsonify(result.to_json(orient='records'))

    return jsonfile


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)


