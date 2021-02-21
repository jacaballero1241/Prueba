import streamlit as st
from PIL import Image

import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import LabelEncoder
import numpy as np

from xgboost import XGBClassifier

from sklearn.metrics import classification_report

st.title('Live or die at Titanic?')

st.write('Want to know whether you would live of die at Titanic? '
         'Input your data in the sidebar and get the predictions!')
st.image(Image.open('Titanic.jpg'), use_column_width=True)

def predictor_variables ():
    columns = ['pclass','age','sibsp', 'parch', 'fare', 'embarked']
    pclass = st.sidebar.slider('Passenger Class (socio-economic): ',1,3)
    female = st.sidebar.slider('Are you male (0) or female (1)? :',0,1)
    age = st.sidebar.slider('Age:',1,99)
    sibsp = st.sidebar.slider('Number of siblings aboard: ',0,10)
    parch = st.sidebar.slider('Number of parents and children aboard: ',0,10)
    fare = st.sidebar.slider('How much would you pay for the ticket? ',0,199)
    embarked = st.sidebar.selectbox('Which port would you embark?', ('C','Q','S'))

    return pd.DataFrame([pclass,age,sibsp,parch,fare,embarked,female],
                        ['pclass', 'age', 'sibsp', 'parch', 'fare', 'embarked','female'])

df = pd.DataFrame.transpose(predictor_variables())

df['pclass'] = df['pclass'].astype(int)
df['age'] = df['age'].astype(float)
df['sibsp'] = df['sibsp'].astype(int)
df['parch'] = df['parch'].astype(int)
df['fare'] = df['fare'].astype(float)
#df['embarked'].astype(object)
df['female'] = df['female'].astype(int)

st.dataframe(df)

train = pd.read_csv('train.csv')
def clean(df):
    dfc = df.copy()
    dfc.drop(columns=['PassengerId','Name', 'Ticket', 'Cabin'], inplace = True)
    dfc.columns = dfc.columns.str.lower()
    dfc.dropna(subset=['embarked'], inplace = True)
    dfc = dfc.assign(female = dfc['sex'].apply(lambda x: 1 if x == 'female' else 0)).drop(columns = ['sex'])
    
    return dfc
train = clean(train)

ct_embarked = ColumnTransformer(
        [('onehot', OneHotEncoder(drop='first'), ['embarked'])],
        remainder='passthrough')

ct_embarked.fit(train[['embarked']])

ct_pclass = ColumnTransformer(
        [('onehot', OneHotEncoder(drop='first'), ['pclass'])],
        remainder='passthrough')

ct_pclass.fit(train[['pclass']])

def encode(df, col):
    ct = ColumnTransformer(
        [('onehot', OneHotEncoder(drop='first'), [col])],
        remainder='passthrough')
    
    dfc = df.copy()
    
    for num in range(len(ct.fit_transform(dfc[[col]])[0])):
        
        dfc[col + str(num)] = [i[num] for i in ct.fit_transform(dfc[[col]])]

    return dfc.drop(columns = [col])

train = encode(train, 'embarked')
train = encode(train, 'pclass')

test = df

for num in range(len(ct_embarked.transform(test[['embarked']])[0])):
        test['embarked' + str(num)] = [i[num] for i in ct_embarked.transform(test[['embarked']])]
        
for num in range(len(ct_pclass.transform(test[['pclass']])[0])):
        test['pclass' + str(num)] = [i[num] for i in ct_pclass.transform(test[['pclass']])]
        
test.drop(columns = ['pclass', 'embarked'], inplace = True)

X_train = train[train.columns.tolist()[1:]]
y_train = train['survived']

X_test = test

xgb = XGBClassifier(random_state=101)

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

st.write(y_pred[0])





