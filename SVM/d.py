from cvxopt import matrix, solvers
import numpy as np
import cv2
import glob
from sklearn.svm import SVC
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


import numpy as np
import cv2
import glob
from sklearn.svm import SVC
import time

from tqdm import tqdm
t1 = time.time()
path1 = 'train/0/'
path2 = 'train/1/'
path3 = 'train/2/'
path4 = 'train/3/'
path5 = 'train/4/'
path6 = 'train/5/'

path1_val = 'val/0/'
path2_val = 'val/1/'
path3_val = 'val/2/'
path4_val = 'val/3/'
path5_val = 'val/4/'
path6_val = 'val/5/'
def confusion_mat(y_pred, y_true):
    classes = np.unique(y_true)

    confmat = np.zeros((len(classes), len(classes)))

    for i in range(len(y_pred)):
        true_class = y_true[i]
        
        predicted_class = y_pred[i]
        confmat[int(true_class)][int(predicted_class)] += 1

    return confmat

def dataloader_1(path0, path1, path2, path3, path4, path5):
     class0 = glob.glob(path0+'*jpg')
     print(len(class0))
     class1 = glob.glob(path1+'*jpg')
     print(len(class1))
     class2 = glob.glob(path2+'*jpg')
     print(len(class2))
     class3 = glob.glob(path3+'*jpg')
     print(len(class3))
     class4 = glob.glob(path4+'*jpg')
     print(len(class4))
     class5 = glob.glob(path5+'*jpg')
     print(len(class5))
     class0_final = np.zeros((len(class0), 768))
     class1_final = np.zeros((len(class1),768))
     class2_final = np.zeros((len(class2), 768))
     class3_final = np.zeros((len(class3),768))
     class4_final = np.zeros((len(class4), 768))
     class5_final = np.zeros((len(class5),768))

     for i in tqdm(range(len(class1))):
        x = cv2.imread(class0[i])
        x = cv2.resize(x,(16,16))
        x = x/255.0
        x = np.array(x)
        x= x.flatten()
        class0_final[i]=x

        y = cv2.imread(class1[i])
        y = cv2.resize(y,(16,16))
        y = y/255.0
        y = np.array(y)
        y = y.flatten()
        class1_final[i]=y

        t = cv2.imread(class2[i])
        t = cv2.resize(t,(16,16))
        t = t/255.0
        t = np.array(t)
        t= t.flatten()
        class2_final[i]=t

        r = cv2.imread(class3[i])
        r = cv2.resize(r,(16,16))
        r = r/255.0
        r = np.array(r)
        r = r.flatten()
        class3_final[i]=r
        
        s = cv2.imread(class4[i])
        s = cv2.resize(s,(16,16))
        s = s/255.0
        s = np.array(s)
        s= x.flatten()
        class4_final[i]=s

        v = cv2.imread(class5[i])
        v = cv2.resize(v,(16,16))
        v = y/255.0
        v = np.array(v)
        v = v.flatten()
        class5_final[i]=v

     y0 = np.zeros(len(class0))
     y1 = np.ones(len(class1))
     y2 = np.full(len(class2), 2)
     y3 = np.full(len(class3), 3)
     y4 = np.full(len(class4), 4)
     y5 = np.full(len(class5), 5)

     X = np.vstack((class0_final,class1_final,class2_final, class3_final, class4_final, class5_final))
     Y = np.hstack((y0,y1,y2,y3,y4,y5))
     Y = np.reshape(Y, ((len(Y),1)))

     return X,Y

X, Y = dataloader_1(path1, path2, path3, path4, path5, path6)

X_val, Y_val = dataloader_1(path1_val, path2_val, path3_val, path4_val, path5_val, path6_val)


def cross_validate(X,Y, C1, k):

    K = k


    num_samples = len(X)
    fold_size = num_samples // K

    best_accuracy = 0

    for i in tqdm(range(K)):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < K - 1 else num_samples
        val_X = X[start_idx:end_idx]
        val_Y = Y[start_idx:end_idx]
        
        
        train_X = np.concatenate([X[:start_idx], X[end_idx:]])
        train_y = np.concatenate([Y[:start_idx], Y[end_idx:]])
        train_y = train_y.T
        train_y = train_y[0]
        model = SVC(C=C1, kernel='rbf', gamma=0.001)  
        model.fit(train_X, train_y)
        
        s = len(val_Y)
        predictions = model.predict(val_X)
        val_Y_new = val_Y.T
        t = (predictions==val_Y_new)
        f = sum(t[0])
        accuracy = f/s


        if accuracy>best_accuracy:
            best_accuracy=accuracy

    return best_accuracy

C = [1e-5, 1e-3, 1, 5, 10]

acc = []

for c in C:
    print(c)
    t = cross_validate(X,Y,c,5)
    acc.append(t)
    print(t)


acc = np.array(acc)
C = np.array(C)

plt.plot(acc, C)
plt.xlabel('C')
plt.ylabel('accuracy')

plt.legend()
plt.savefig('CvsAccuracy.jpg')

        

