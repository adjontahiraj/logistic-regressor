from sklearn import svm
import time
from sklearn import metrics
import numpy as np
def load_features(filename):
    data_x = []
    data_y = []
    features = np.load(filename)
    for entry in features:
        expected = entry[len(entry)-1]
        if(expected > 6):
            expected = 2
        elif(expected == 5 or expected == 6):
            expected = 1
        else:
            expected = 0
        data_y.append(expected)
        data_x.append(np.delete(entry,len(entry)-1))
    dx = np.array(data_x)
    dy = np.array(data_y)
    return dx,dy


t0 = time.time()
x, y = load_features('hw2_winequality-red_train.npy')
x_test, y_test = load_features('hw2_winequality-red_test.npy')
model = svm.SVC(kernel='rbf', verbose=3)
fit = model.fit(x,y)
print(fit)
y_pred = model.predict(x_test)
t1 = time.time()
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("TIME",t1-t0)
