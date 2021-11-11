import pickle
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

iris_dataset = load_iris()

# Load Iris model into memory
with open ("./model.pkl","rb") as pickle_out:
    knn = pickle.load(pickle_out)

 #Test Data

x_new = np.array([[1.2, 1.6, 1.8, 2.4]])

prediction  = knn.predict(x_new)
pred = iris_dataset['target_names'][prediction]
print(f"Prediction is : {x_new} , the target value is : {prediction} and target name is : {pred} ")

