import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pickle
import pandas as pd
data = pd.read_csv("Churn_Modelling.csv")
print(data.head())

data = data.drop(['RowNumber','CustomerId','Surname'],axis=1)
print(data)

label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])
print(data)

from sklearn.preprocessing import OneHotEncoder
onehot_encoder_geo = OneHotEncoder()
geo_encoder = onehot_encoder_geo.fit_transform(data[['Geography']]).toarray()
print(geo_encoder)

print(onehot_encoder_geo.get_feature_names_out(['Geography']))

geo_encoded_df = pd.DataFrame(geo_encoder,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
print(geo_encoded_df)

data = pd.concat([data.drop('Geography',axis=1),geo_encoded_df],axis=1)

with open('label_encoder_gender.pkl','wb') as file:
    pickle.dump(label_encoder_gender,file)

with open('onehot_encoder_geo.pkl','wb') as file:
    pickle.dump(onehot_encoder_geo,file)

print(data.head())

X = data.drop('Exited',axis=1)
y = data['Exited']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

with open('scaler.pkl','wb') as file:
    pickle.dump(scaler,file)

# use simple ann to predict
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import datetime

model = Sequential([
    Dense(64,activation='relu',input_shape=(X_train.shape[1],)),
    Dense(32,activation = 'relu'),
    Dense(1,activation='sigmoid')
])

print(model.summary())

# build a optimizer and define the super parameters
import tensorflow as tf

lr = 0.01
opt = tf.keras.optimizers.Adam(learning_rate=lr)
loss = tf.keras.losses.BinaryCrossentropy()

# compile the model
model.compile(optimizer = opt,loss = loss,metrics = ['accuracy'])

# set the early callback to prevent overfitting
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import os

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(os.path.dirname(log_dir), exist_ok=True)

# log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorflow_callback=TensorBoard(log_dir=log_dir,histogram_freq=1)

early_stopping_callback = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)

history = model.fit(
    X_train,y_train,validation_data=(X_test,y_test),epochs=100,
    callbacks=[early_stopping_callback]
)

model.save('model.h5')

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Baseline Training Accuracy', linestyle='--')
plt.plot(history.history['val_accuracy'], label='Baseline Validation Accuracy', linestyle='--')
  
plt.xlabel('Epochs')
plt.ylabel(f'Accuracy')
plt.legend()
plt.show()