import pandas as pd
df = pd.read_csv('./Churn_Modelling.csv')

df.drop(['CustomerId','Surname','RowNumber'],axis='columns',inplace=True)

def obj_unique(df):
  for col in df:
    if df[col].dtype=='object':
      print(f'{col}: {df[col].unique()}')
      
obj_unique(df)

df['Gender'].replace({'Female':1,'Male':0},inplace=True)

df1=pd.get_dummies(data=df,columns=['Geography'])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df2=df1.copy()
to_be_scaled=['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']

df2[to_be_scaled]=scaler.fit_transform(df2[to_be_scaled])

X=df2.drop('Exited',axis='columns')
y=df2['Exited']


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=5)

import tensorflow as tf
from tensorflow import keras

model=keras.Sequential([
                        keras.layers.Dense(12,input_shape=(12,),activation='relu'),
                        keras.layers.Dense(6,activation='relu'),
                        keras.layers.Dense(1,activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']

)

model.fit(X_train,y_train,epochs=100)

yp=model.predict(X_test)

y_predicted=[]

for i in yp:
  if i>=0.5:
    y_predicted.append(1)
  else:
    y_predicted.append(0)
    
model.evaluate(X_test,y_test)

from sklearn.metrics import confusion_matrix,classification_report

print(classification_report(y_predicted,y_test))
