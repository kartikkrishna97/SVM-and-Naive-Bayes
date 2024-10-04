import numpy as np
import cv2
import glob
from sklearn import svm
import time
import matplotlib.pyplot as plt
t1 = time.time()
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

model = svm.SVC(kernel='rbf')
model.fit(X,Y)
t2 = time.time()
y = model.support_vectors_
print(y.shape)



y_pred = model.predict(X_val)
Y_val = Y_val.T
x= (y_pred==Y_val)
d = sum(x[0])
print(d)
accuracy = d/400

print(f"gaussian sklearn SVM prediction accuracy is {100*accuracy}")
print(f"time taken for sklearn is {t2-t1}")
