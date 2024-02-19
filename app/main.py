from matplotlib import cm
from sklearn.linear_model import LogisticRegression
import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def create_model(data):
  X = data.drop(['diagnosis'], axis=1)
  Y = data['diagnosis']

 #scale data
  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  #split the data
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

  models = {'KNN': KNeighborsClassifier(),'SVM': SVC(),'Naive Bayes': GaussianNB()}
  st.sidebar.title("Model Selection")
  selected_model = st.sidebar.selectbox('Please select a model:', list(models.keys()))

  if selected_model == "KNN":
        model = KNeighborsClassifier()
  elif selected_model == "SVM":
        model = SVC()
  else:
        model = GaussianNB()
        
  
  st.write('Selected Model:', selected_model)
        
  model = models[selected_model]



  model.fit(X_train, Y_train)
  Y_pred = model.predict(X_test)

  model = LogisticRegression()
  model.fit(X_train, Y_train)

  # Model evaluation
  accuracy = accuracy_score(Y_test, Y_pred)
  precision = precision_score(Y_test, Y_pred)
  recall = recall_score(Y_test, Y_pred)
  f1 = f1_score(Y_test, Y_pred)
  cm = confusion_matrix(Y_test, Y_pred)

  accuracy = metrics.accuracy_score(Y_test, Y_pred)
  st.write('Accuracy:', accuracy)

  return model, scaler, accuracy, precision, recall, f1, cm

def get_clean_data():
  data=pd.read_csv("data.csv")

  data = data.drop(['Unnamed: 32', 'id'], axis=1)
  data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
  return data

def add_sidebar():
  st.sidebar.header("Cell Nuclei Measurements")
  #dosya yükleme alanı
  uploaded_file = st.sidebar.file_uploader("Please choose a data set", type=['csv', 'xlsx'])

  if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)
    st.write("Selected data set:")
    data = get_clean_data()
    st.write(data)

    data = get_clean_data()
    slider_labels = [
      ("Radius (mean)", "radius_mean"), ("Texture (mean)", "texture_mean"),
                     ("Perimeter (mean)", "perimeter_mean"), ("Area (mean)", "area_mean"),
                     ("Smoothness (mean)",
                      "smoothness_mean"), ("Compactness (mean)", "compactness_mean"),
                     ("Concavity (mean)", "concavity_mean"), ("Concave points (mean)",
                                                              "concave points_mean"),
                     ("Symmetry (mean)", "symmetry_mean"), ("Fractal dimension (mean)",
                                                            "fractal_dimension_mean"),
                     ("Radius (se)", "radius_se"), ("Texture (se)",
                                                    "texture_se"), ("Perimeter (se)", "perimeter_se"),
                     ("Area (se)", "area_se"), ("Smoothness (se)", "smoothness_se"),
                     ("Compactness (se)",
                      "compactness_se"), ("Concavity (se)", "concavity_se"),
                     ("Concave points (se)",
                      "concave points_se"), ("Symmetry (se)", "symmetry_se"),
                     ("Fractal dimension (se)",
                      "fractal_dimension_se"), ("Radius (worst)", "radius_worst"),
                     ("Texture (worst)", "texture_worst"), ("Perimeter (worst)",
                                                            "perimeter_worst"),
                     ("Area (worst)", "area_worst"), ("Smoothness (worst)",
                                                      "smoothness_worst"),
                     ("Compactness (worst)",
                      "compactness_worst"), ("Concavity (worst)", "concavity_worst"),
                     ("Concave points (worst)",
                      "concave points_worst"), ("Symmetry (worst)", "symmetry_worst"),
                     ("Fractal dimension (worst)", "fractal_dimension_worst")]

    input_dict = {}

    for label, key in slider_labels:
      input_dict[key] = st.sidebar.slider(
        label,
        min_value = float(0),
        max_value = float(data[key].max()),
        value = float(data[key].mean())
      )

    return input_dict

def get_scaled_values(input_dict):
  data = get_clean_data()

  X = data.drop(['diagnosis'], axis=1)

  scaled_dict = {}

  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value

  return scaled_dict

def get_radar_chart(input_data):
  
 input_data = get_scaled_values(input_data)
 categories = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension']

 fig = go.Figure()

 fig.add_trace(
        go.Scatterpolar(
            r=[input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
                input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
                input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
                input_data['fractal_dimension_mean']],
            theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension'],
            fill='toself',
            name='Mean'
        )
    )

 fig.add_trace(
        go.Scatterpolar(
            r=[input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
                input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
                input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']],
            theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension'],
            fill='toself',
            name='Standard Error'
        )
    )

 fig.add_trace(
        go.Scatterpolar(
            r=[input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
                input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
                input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
                input_data['fractal_dimension_worst']],
            theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension'],
            fill='toself',
            name='Worst'
        )
    )

 fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        autosize=True
    )
 return fig

def add_predictions(input_data):
  model = pickle.load(open("model.pkl", "rb"))
  scaler = pickle.load(open("scaler.pkl", "rb"))

  input_array = np.array(list(input_data.values())).reshape(1, -1)
  input_array_scaled = scaler.transform(input_array)

  prediction = model.predict(input_array_scaled)

  st.subheader("Cell Cluster Prediction")
  st.write("The cell cluster is:")

  if prediction[0] == 0:
    st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
  else:
    st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)


  st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
  st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])
  st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")


def main():
  
  
  st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="auto"
    )
  
  with open("assets/style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)


  input_data = add_sidebar()
  
  with st.container():
    st.title("Breast Cancer Diagnosis")
    st.write("This app predicts whether a breast mass is benign or malignant using a machine learning model based on measurements from your cytosis lab. You can also update the measurements manually using the sliders in the sidebar and select a model at the bottom of the sidebar.")
  

    data = pd.read_csv("data.csv")
    data = get_clean_data()
    st.subheader("First 10 lines:")
    st.write(data.head(10))

    st.subheader("Last 10 lines:")
    st.write(data.tail(10))

    st.subheader("Columns:")
    st.write(data.columns)

    col1, col2= st.columns([4,2])


    with col1:
     radar_chart = get_radar_chart(input_data)
     st.plotly_chart(radar_chart)

    with col2:
     add_predictions(input_data)

  data = get_clean_data()
 
  model, scaler, accuracy, precision, recall, f1, cm = create_model(data)
  st.subheader("Model Evaluation Results:")
  st.write("Accuracy:", accuracy)
  st.write("Precision:", precision)
  st.write("Recall:", recall)
  st.write("F1 Score:", f1)
  st.write("Confusion Matrix:")
  st.write(cm)

if __name__=='__main__':
 main()