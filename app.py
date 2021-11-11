from flask import  Flask,request
import pandas as pd 
import numpy as np
import sys
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

port = int(os.environ.get("PORT", 5000))
iris_dataset = load_iris()
app = Flask(__name__)


pickle_out = open("./model.pkl","rb")
knn = pickle.load(pickle_out)

@app.route('/home')
def welcome():
    return "Welcome All"

    
@app.route('/predict')

def predict_iris():

    # Read all necessary request parameters
    sl = request.args.get('sl')

    sw = request.args.get('sw')

    pl = request.args.get('pl')

    pw = request.args.get('pw')

    # Use the predict method of the model to

    # get the prediction for unseen data

    x_new = np.array([[sl, sw, pl, pw]])

    prediction  = knn.predict(x_new)
    pred = iris_dataset['target_names'][prediction]

    return 'Predicted value is :'+str(pred)


if __name__ =='__main__' :
    app.run(host="0.0.0.0",port = 5000)

