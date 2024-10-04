from cvxopt import matrix, solvers
import numpy as np
import cv2
import glob
from tqdm import tqdm 

threshold = 1.5e-3

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
print(X_val.shape)
print(X.shape)
def diff_norm(x1,x2):
    return x1.T@x1 + x2.T@x2 - 2*(x1.T@x2)
def gaussian_kernel(x1, x2, gamma):
    return np.exp(-1*gamma*diff_norm(x1,x2))




def compute_kernel_matrix(images, gamma):
    n_samples = len(images)
    K = np.zeros((n_samples, n_samples))

    for i in tqdm(range(n_samples)):
        for j in range(n_samples):
            K[i, j] = gaussian_kernel(images[i], images[j], gamma)

    return K

def compute_gaussian_kernel_validation(X_val, sv_X, sigma):
    kernel_vals = np.zeros((len(sv_X), len(X_val)))
    for i in tqdm(range(len(sv_X))):
        for j in range(len(X_val)):
            kernel_vals[i, j] = gaussian_kernel(sv_X[i], X_val[j], sigma)
    return kernel_vals

K = compute_kernel_matrix(X, 0.001)


P = matrix(np.outer(Y, Y) * K, tc = 'd')  

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


support_alphas = np.array(support_vectors)



support_vectors_indices = []
for i in range(len(alpha)):
    if alpha[i]>threshold:
        support_vectors_indices.append(i)

support_vectors_indices = np.array(support_vectors_indices)


y_support = []
x_support = []
for i in range(len(support_vectors_indices)):
    y_support.append(Y[support_vectors_indices[i]])
    x_support.append(X[support_vectors_indices[i]])
  

y_new = np.array(y_support)
x_new = np.array(x_support)

index_min = -1

for i in support_vectors_indices:
    if alpha[i]<0.9:
        index_min=i
        break


bias = Y[index_min]*1.0


for i in support_vectors_indices:
    bias -= alpha[i] * Y[i] * K[i][index_min]




validation_kernel = compute_gaussian_kernel_validation(X_val, x_support, 0.001)
predicted_labels = np.dot(validation_kernel.T, (support_alphas*y_support))+bias


predicted_labels = np.sign(predicted_labels)


accuracy = 100*(sum(Y_val==predicted_labels))/len(Y_val)

print(f"validation accuracy for gaussian kernel is {accuracy[0]} ")









