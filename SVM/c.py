from numpy.linalg import norm
import numpy as np
import cv2
import glob
from tqdm import tqdm
from cvxopt import matrix, solvers
threshold = 1.5e-3


path = 'train/'

def diff_norm(x1,x2):
    return x1.T@x1 + x2.T@x2 - 2*(x1.T@x2)
def gaussian_kernel(x1, x2, gamma):
    return np.exp(-1*gamma*diff_norm(x1,x2))

def gaussian_kernel_1(x1, x2, gamma):
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

def confusion_mat(y_pred, y_true):
    classes = np.unique(y_true)

    confmat = np.zeros((len(classes), len(classes)))

    for i in range(len(y_pred)):
        true_class = y_true[i]
        
        predicted_class = y_pred[i]
        confmat[int(true_class)][int(predicted_class)] += 1

    return confmat

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

# X, Y = dataloader(path0, path1, path2, path3, path4, path5)
X_val, Y_val = dataloader_1(path1_val, path2_val, path3_val, path4_val, path5_val, path6_val)

def dataloader(path1, path2):
    print(path1)
    print(path2)
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



def train_gaussian_SVM(X,Y):
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
    

    y_support = np.array(y_support)
    x_support = np.array(x_support)

    index_min = -1

    for i in support_vectors_indices:
        if alpha[i]<0.9:
            index_min=i
            break


    bias = Y[index_min]*1.0


    for i in support_vectors_indices:
        bias -= alpha[i] * Y[i] * K[i][index_min]

    return support_alphas, x_support, y_support, bias


def predict_function(X_val, support_alphas, x_support, y_support, bias):
    validation_kernel = compute_gaussian_kernel_validation(X_val, x_support, 0.001)
    predicted_labels = np.dot(validation_kernel.T, (support_alphas*y_support))+bias

    return predicted_labels




def multclass_predict(X, Y, num_classes):
    correct = 0
    y_true = []
    y_pred = []
    cnt = [[0] * num_classes for _ in range(len(X))]  
    score = [[0] * num_classes for _ in range(len(X))]  

    for j in range(num_classes):
        for l in range(num_classes):
            if j > l:
                X_train, Y_train = dataloader(path + f"{j}/", path + f"{l}/")
                support_alphas, x_support, y_support, bias = train_gaussian_SVM(X_train, Y_train)
                pred_jl = predict_function(X, support_alphas, x_support, y_support, bias)

                
                for i in range(len(X)):
                    if pred_jl[i] >= 0:
                        cnt[i][j] += 1
                        score[i][j] = max(score[i][j], abs(pred_jl[i]))
                    else:
                        cnt[i][l] += 1
                        score[i][l] = max(score[i][l], abs(pred_jl[i]))

 
    y_true = Y[:] 

    
    for i in range(len(X)):
        max_votes = max(cnt[i])  
        candidates = [j for j in range(num_classes) if cnt[i][j] == max_votes]  
        
       
        max_score = -1
        ans = -1
        for x in candidates:
            if score[i][x] > max_score:
                max_score = score[i][x]
                ans = x
        y_pred.append(ans)

        
        if y_pred[i] == y_true[i]:
            correct += 1

    
    confusion_matrix = confusion_mat(y_pred, y_true)
    
    
    print(f"Accuracy = {100 * correct / len(X):.2f}%")
    print(f"Confusion matrix: \n{confusion_matrix}")



multclass_predict(X_val, Y_val, num_classes=6)
