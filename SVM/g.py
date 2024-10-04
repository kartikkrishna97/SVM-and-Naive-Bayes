import numpy as np
import cv2
import glob
from sklearn.svm import SVC
import time
import matplotlib.pyplot as plt
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
l = len(Y_val)
model = SVC(kernel='rbf', decision_function_shape='ovr', gamma=0.001, C=1)
model.fit(X, Y)
t2 =time.time()
y_pred = model.predict(X_val)

Y_val_new = Y_val.T

x = sum(y_pred==Y_val_new)
print(sum(x))
accuracy = sum(x)/l

confusion_matrix = confusion_mat(y_pred, Y_val)

print(f"accuracy of sklearn multiclass SVM is {accuracy}")
print(f"confusion matrix of sklearn SVM is {confusion_matrix}")
print(t2-t1)




lst = []
lst_1 = []
for i in range(len(y_pred)):
    if y_pred[i]!=Y_val[i]:
        lst.append(i)
        lst_1.append(Y_val[i])

lst = np.array(lst)
lst = lst.astype(int)
lst_1 = np.array(lst_1)
lst_1 = lst_1.T
lst_1 = lst_1[0]
lst_1 = lst_1.astype(int)
print(lst)
print(lst_1)
X_1 = X_val[lst[0]]


X_1 = X_1.reshape(16, 16, 3)
plt.imshow(X_1)
plt.savefig(f'{lst_1[0]}_image1.jpg')

X_2 = X_val[lst[1]]

X_2 = X_2.reshape(16, 16, 3)
plt.imshow(X_2)
plt.savefig(f'{lst_1[1]}_image2.jpg')


X_3 = X_val[lst[2]]

X_3 = X_3.reshape(16, 16, 3)
plt.imshow(X_3)
plt.savefig(f'{lst_1[2]}_image3.jpg')


X_4 = X_val[lst[3]]

X_4 = X_4.reshape(16, 16, 3)
plt.imshow(X_4)
plt.savefig(f'{lst_1[3]}_image4.jpg')

X_5 = X_val[lst[4]]
print(X_5.shape)
X_5 = X_5.reshape(16, 16, 3)
plt.imshow(X_5)
plt.savefig(f'{lst_1[4]}_image5.jpg')

X_6 = X_val[lst[5]]

X_6 = X_6.reshape(16, 16, 3)
plt.imshow(X_6)
plt.savefig(f'{lst_1[5]}_image6.jpg')

X_7 = X_val[lst[6]]


X_7 = X_7.reshape(16, 16, 3)
plt.imshow(X_7)
plt.savefig(f'{lst_1[6]}_image7.jpg')

X_8 = X_val[lst[7]]

X_8 = X_8.reshape(16, 16, 3)
plt.imshow(X_8)
plt.savefig(f'{lst_1[7]}_image8.jpg')


X_9 = X_val[lst[8]]

X_9 = X_9.reshape(16, 16, 3)
plt.imshow(X_3)
plt.savefig(f'{lst_1[8]}_image9.jpg')


X_10 = X_val[lst[9]]

X_10 = X_10.reshape(16, 16, 3)
plt.imshow(X_10)
plt.savefig(f'{lst_1[9]}_image10.jpg')

X_11 = X_val[lst[10]]
print(X_10.shape)
X_11 = X_11.reshape(16, 16, 3)
plt.imshow(X_11)
plt.savefig(f'{lst_1[10]}_image11.jpg')

X_12 = X_val[lst[11]]

X_12 = X_12.reshape(16, 16, 3)
plt.imshow(X_12)
plt.savefig(f'{lst_1[11]}_image12.jpg')




    
    


