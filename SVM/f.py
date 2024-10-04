import numpy as np
import cv2
import glob
from sklearn import svm
import time
import matplotlib.pyplot as plt
path1 = 'train/3/'
path2 = 'train/4/'

path3 = 'val/3/'
path4 = 'val/4/'


def dataloader(path1, path2):
    class1 = glob.glob(path1+'*jpg')
    class2 = glob.glob(path2+'*jpg')
    class1_final = np.zeros((len(class1), 768))
    class2_final = np.zeros((len(class2),768))
    for i in range(len(class1)):
        x = cv2.imread(class1[i])
        x = cv2.resize(x,(16,16))
        x = x/255.0
        x = np.array(x)
        x= x.flatten()
        class1_final[i]=x

        y = cv2.imread(class2[i])
        y = cv2.resize(y,(16,16))
        y = y/255.0
        y = np.array(y)
        y = y.flatten()
        class2_final[i]=y
    y1 = np.zeros(len(class1))
    y2 = np.zeros(len(class2))
    for i in range(len(class1)):
        y1[i]=1
        y2[i]=-1

    X = np.vstack((class1_final,class2_final))
    Y = np.hstack((y1,y2))
    
    Y = np.reshape(Y, ((len(Y),1)))

    return X,Y

X, Y = dataloader(path1, path2)
X_val, Y_val = dataloader(path3, path4)

print(X.shape)
print(Y.shape)
t1 = time.time()
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)
y = clf.support_vectors_
print(y.shape)
t2 = time.time()
decision_function = clf.decision_function(X)
weight = clf.coef_[0]
weight = np.array(weight)
weight_new = weight.reshape(16,16,3)
plt.imshow(weight_new)
plt.savefig('weight_vector_sklearn.jpg')


bias = clf.intercept_[0]

def prediction_accuracy(X,Y, w, b):
    y_predictions =  np.zeros((len(Y),1))

    for i in range(len(X)):
        if np.dot(X[i], w)+b>=0:
            y_predictions[i]=1
        else:
            y_predictions[i]=-1

    accuracy = 100*sum(y_predictions==Y)/len(Y)

    return accuracy

print(f"training accuracy using sklearn SVM is {prediction_accuracy(X, Y, weight, bias)}")
print(f"validation accuracy using skelearn SVM is {prediction_accuracy(X_val, Y_val, weight, bias)}")

print(f"training time using sklearn SVM is {t2-t1}")