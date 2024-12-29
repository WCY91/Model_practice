import tensorflow as  tf
from tensorflow.keras.models import Sequential,load_model
import pickle
import pandas as pd
import numpy as np

model = load_model('model.h5')

with open('onehot_encoder_geo.pkl','rb') as file:
    label_encoder_geo=pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}

geo_encoded = label_encoder_geo.transform([[input_data['Geography']]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=label_encoder_geo.get_feature_names_out(['Geography']))

input_df = pd.DataFrame([input_data])
input_df['Gender']=label_encoder_gender.transform(input_df['Gender'])
input_df=pd.concat([input_df.drop("Geography",axis=1),geo_encoded_df],axis=1)

input_scaled=scaler.transform(input_df)

prediction=model.predict(input_scaled)
prediction_proba = prediction[0][0]
prediction_proba

if prediction_proba > 0.5: # sigmoid 
    print('The customer is likely to churn.')
else:
    print('The customer is not likely to churn.')