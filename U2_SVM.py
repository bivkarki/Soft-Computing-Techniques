from sklearn import datasets
import numpy as np

iris=datasets.load_iris()
X=iris.data[:,[2,3]] #only 2 features
Y=iris.target

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

from sklearn.svm import SVC
svm= SVC(kernel='linear',C=1,random_state=0)
svm.fit(X_train,Y_train)
Y_pred=svm.predict(X_test)
print("Misclassified samples: %d"%(Y_test!=Y_pred).sum()) #compute


from sklearn.metrics import accuracy_score
print("Accuracy: %.2f"%accuracy_score(Y_test,Y_pred))
