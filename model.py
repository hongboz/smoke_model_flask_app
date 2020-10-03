import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion

from utils import FeatureSelector
from joblib import dump

# import data and condensing it a bit
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

# TODO: Process missing data in pipeline

categorical_pipeline = Pipeline(steps = [('cat_selector', FeatureSelector(['Sex', 'Race', 'RuralUrban'])),
                                            ('one_hot_enc', OneHotEncoder(sparse = False))])

numerical_pipeline = Pipeline(steps = [('num_selector', FeatureSelector(['Age']))])

feature_pipeline = FeatureUnion(transformer_list = [('numerical_pipeline', numerical_pipeline), 
                                                    ('categorical_pipeline', categorical_pipeline)])

final_pipeline = Pipeline(steps = [('feature_pipeline', feature_pipeline), ('model', LogisticRegression(C = 0.001))])

le = LabelEncoder()
y = df[['ever_cigarettes']].to_numpy()
y = le.fit_transform(y.ravel())

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = 0.2)

final_pipeline.fit(X_train, y_train)

# final_pipeline.score(X_test, y_test)

dump(final_pipeline, "smoke_model.joblib")