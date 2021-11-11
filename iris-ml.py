from sklearn import neighbors
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
from sklearn.metrics import accuracy_score
import pickle
### Load Data from sklearn
iris_dataset = load_iris()


### Segregate features and taget Labels

x = iris_dataset.data

y = iris_dataset.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)

### K-Nearest Neighbours Classifier 

knn =neighbors.KNeighborsClassifier()
### Train model based on test
knn.fit(x_train,y_train)

predictions = knn.predict(x_test)
### Check Accuracy

print(accuracy_score(y_test,predictions))

### Create a pickle file using serialization

pickle_out = open("./model.pkl","wb")
pickle.dump(knn,pickle_out)

pickle_out.close()