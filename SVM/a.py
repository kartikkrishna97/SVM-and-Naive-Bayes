from cvxopt import matrix, solvers
import numpy as np
import cv2
import glob
threshold = 1e-4
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

s = Y*X
y = (Y*X).T
t = np.dot(s,y)
P = matrix(t, tc = 'd')
A = matrix(Y.reshape(1, -1), tc = 'd')


b = matrix(np.array([0]), tc = 'd')
q = matrix(-1*np.ones((X.shape[0],1)), tc = 'd')
C = np.ones((X.shape[0],1))
temp1 = -1*np.eye(X.shape[0])
temp2 = np.eye(X.shape[0])
G = matrix(np.vstack((temp1,temp2)), tc = 'd')
zeros = np.zeros((X.shape[0],1))
h = matrix(np.vstack((zeros,C)), tc = 'd')


solvers.options['show_progress'] = True
cvxopt_solver = solvers.qp(P, q, G, h, A, b)

alpha = np.array(cvxopt_solver['x'])


support_vectors = []

for i in range(len(alpha)):
    if alpha[i]>threshold:
        support_vectors.append(alpha[i])


support_vectors = np.array(support_vectors)
vec = support_vectors.T
indices = np.argsort(vec)[::-1][:6]



support_vectors_indices = []
for i in range(len(alpha)):
    if alpha[i]>threshold:
        support_vectors_indices.append(i)

support_vectors_indices = np.array(support_vectors_indices)

w = []
y_new = []
x_new = []
for i in range(len(support_vectors_indices)):
    x = (Y[support_vectors_indices[i]]*alpha[support_vectors_indices[i]])
    y = X[support_vectors_indices[i]]*x
    w.append(y)
    y_new.append(Y[support_vectors_indices[i]])
    x_new.append(X[support_vectors_indices[i]])
  

w = np.array(w)

w = np.sum(w, axis=0)
w = np.reshape(w, (len(w),1))
y_new = np.array(y_new)
x_new = np.array(x_new)


b = []
for i in range(len(y_new)):
    
    s = np.dot(x_new[i],w)
    

    r = sum(y_new[i]-s)
    
    b.append((y_new[i]-s))
    

b = sum(b)/len(b)





X_val, Y_val = dataloader(path3, path4)
print(X_val.shape)
print(Y_val.shape)

def prediction_accuracy(X,Y):
    y_predictions =  np.zeros((len(Y),1))

    for i in range(len(X)):
        if np.dot(X[i], w)+b[0]>=0:
            y_predictions[i]=1
        else:
            y_predictions[i]=-1

    accuracy = 100*sum(y_predictions==Y)/len(Y)

    return accuracy[0]

t2 = time.time()
# clf = svm.SVC(kernel='linear')
# clf.fit(X, Y)
# x =  clf.support_vectors_

# print(len(x))





# print(f"the difference between sklearn and cvxopt support vectors is {abs(sum(support_vectors)-sum(x))}")
print(f"time taken with cvxopt is {t2-t1}")

print(f"train accuracy is {prediction_accuracy(X,Y)}")
print(f"validation accuracy is {prediction_accuracy(X_val,Y_val)}")

w_new = np.reshape(w, (16,16,3))
plt.imshow(w_new)
plt.savefig('weight_vector.jpg')

X_1 = X[indices[0][0]]


X_1 = X_1.reshape(16, 16, 3)
plt.imshow(X_1)
plt.savefig('top1_image.jpg')

X_2 = X[indices[0][1]]

X_2 = X_2.reshape(16, 16, 3)
plt.imshow(X_2)
plt.savefig('top2_image.jpg')


X_3 = X[indices[0][2]]

X_3 = X_3.reshape(16, 16, 3)
plt.imshow(X_3)
plt.savefig('top3_image.jpg')


X_4 = X[indices[0][3]]

X_4 = X_4.reshape(16, 16, 3)
plt.imshow(X_4)
plt.savefig('top4_image.jpg')

X_5 = X[indices[0][4]]
print(X_5.shape)
X_5 = X_5.reshape(16, 16, 3)
plt.imshow(X_5)
plt.savefig('top5_image.jpg')

X_6 = X[indices[0][5]]

X_6 = X_6.reshape(16, 16, 3)
plt.imshow(X_6)
plt.savefig('top6_image.jpg')








