import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


def create_model(data):
  X = data.drop(['diagnosis'], axis=1)
  Y = data['diagnosis']

 #scale data
  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  #split the data
  X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
  )

  #---------------------
  iris = datasets.load_iris()
  X = iris.data
  Y = iris.target


  models = {
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(),
    'Naive Bayes': GaussianNB()
}

  st.title('Model Seçimi ve Tahmin')

  selected_model = st.sidebar.selectbox('Please select a model:', list(models.keys()))

  model = models[selected_model]
  model.fit(X_train, Y_train)

  Y_pred = model.predict(X_test)

  accuracy = metrics.accuracy_score(Y_test, Y_pred)

# Sonuçları gösterme
  st.write('Seçilen Model:', selected_model)
  st.write('Doğruluk:', accuracy)  
  #-------------------

  #train the model
  model = LogisticRegression()
  model.fit(X_train, Y_train)


  #test model
  Y_pred = model.predict(X_test)
  print('Accuracy of our model: ',accuracy_score(Y_test, Y_pred))
  print("Classification report: \n", classification_report(Y_test, Y_pred))
  

  return model, scaler


def get_clean_data():
  data=pd.read_csv("data.csv")

  data = data.drop(['Unnamed: 32', 'id'], axis=1)
  data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
  return data

def main():

 data = get_clean_data()

 model, scaler = create_model(data)

 with open('model.pkl', 'wb') as f:
   pickle.dump(model, f)

 with open('scaler.pkl', 'wb') as f:
   pickle.dump(scaler, f)

   

if __name__=='__main__':
  main()