import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm
from urllib.parse import urlparse
import matplotlib.pyplot as plt

path1 = 'Corona_train.csv'
path2 = 'Corona_validation.csv'
path3 = 'Twitter_train_1.csv'
path4 = 'Twitter_validation.csv'
path5 = 'Twitter_train_2.csv'
path6 = 'Twitter_train_5.csv'
path7 = 'Twitter_train_10.csv'
path8 = 'Twitter_train_25.csv'
path9 = 'Twitter_train_50.csv'
path10 = 'Twitter_train_100.csv'

def data_loader(path):
    data = pd.read_csv(path)

    x = np.array(data['CoronaTweet'])
    x = np.reshape(x, (len(x),1))
    y = np.array(data['Sentiment'])
    y = np.reshape(y,(len(y),1))

    for i in range(len(y)):
        if y[i] == 'Positive':
            y[i]=0
        elif y[i] =='Negative':
            y[i]=1
        else:
            y[i]=2

    return x, y


def data_loader_domain(path):
    data = pd.read_csv(path)

    x = np.array(data['Tweet'])
    x = np.reshape(x, (len(x),1))
    y = np.array(data['Sentiment'])
    y = np.reshape(y,(len(y),1))

    for i in range(len(y)):
        if y[i] == 'Positive':
            y[i]=0
        elif y[i] =='Negative':
            y[i]=1
        else:
            y[i]=2

    return x, y

def data_loader_domain_100(path):
    data = pd.read_csv(path)

    x = np.array(data['Tweet'])
    x = np.reshape(x, (len(x),1))
    y = np.array(data['Sentimwnt'])
    y = np.reshape(y,(len(y),1))

    for i in range(len(y)):
        if y[i] == 'Positive':
            y[i]=0
        elif y[i] =='Negative':
            y[i]=1
        else:
            y[i]=2

    return x, y


x_train, y_train = data_loader(path1)
    
x_val, y_val = data_loader(path2)

x_domain_1, y_domain_1 = data_loader_domain(path3)
x_domain_2, y_domain_2 = data_loader_domain(path5)
x_domain_5, y_domain_5 = data_loader_domain(path6)
x_domain_10, y_domain_10 = data_loader_domain(path7)
x_domain_25, y_domain_25 = data_loader_domain(path8)
x_domain_50, y_domain_50 = data_loader_domain(path9)
x_domain_100, y_domain_100 = data_loader_domain_100(path10)
x_domain_val, y_domain_val = data_loader_domain(path4)


stop_words = set(stopwords.words('english'))

def preprocess_data(x,stop_words):
    new_x = []
    porter_stemmer = PorterStemmer()
    for sent in x:
        x = sent[0].split(' ')
        
        
        lst = []
        for i in range(len(x)):
            if x[i] not in stop_words:
                lst.append(porter_stemmer.stem(x[i]))
                
        lst1 = []
        for i in range(len(lst)):
            if lst[i] not in  urlparse(lst[i]).scheme:
                lst1.append(lst[i])
  
        new_x.append(lst1)

        
     
        
    return new_x

x_train = preprocess_data(x_train, stop_words)

x_domain_1 = preprocess_data(x_domain_1, stop_words)
x_domain_2 = preprocess_data(x_domain_2, stop_words)
x_domain_5 = preprocess_data(x_domain_5, stop_words)
x_domain_10 = preprocess_data(x_domain_10, stop_words)
x_domain_25 = preprocess_data(x_domain_25, stop_words)
x_domain_50 = preprocess_data(x_domain_50, stop_words)
x_domain_100 = preprocess_data(x_domain_100, stop_words)
x_domain_val = preprocess_data(x_domain_val, stop_words)

def get_prior(y):
    phi = np.zeros(3)
    for i in range(len(y)):
        if y[i] == 0: 
            phi[0]+=1
        if y[i] == 1:
            phi[1]+=1
        if y[i] == 2:
            phi[2]+=1

    phi = phi/len(y)
    
    return phi

phi = get_prior(y_train)

def naive_bayes_dict(x, y):
    theta_dict = {}
    for i in tqdm(range(len(x))):
        for j in range((len(x[i]))):
            if x[i][j] not in theta_dict:
                theta_dict[x[i][j]]={}
                theta_dict[x[i][j]]['class 0']=1
                theta_dict[x[i][j]]['class 1']=1
                theta_dict[x[i][j]]['class 2']=1
            if y[i]==0:
                if x[i][j] in theta_dict:
                    theta_dict[x[i][j]]['class 0'] +=1
                    
            elif y[i]==1:
                if x[i][j] in theta_dict:
                    theta_dict[x[i][j]]['class 1'] +=1
            else:
                if x[i][j] in theta_dict:
                    theta_dict[x[i][j]]['class 2'] += 1

    return theta_dict



def dict_domain(x,y,theta_dict1):
    theta_dict_new = theta_dict1
    for i in tqdm(range(len(x))):
        for j in range((len(x[i]))):
            if x[i][j] not in theta_dict_new:
                theta_dict_new[x[i][j]]={}
                theta_dict_new[x[i][j]]['class 0']=1
                theta_dict_new[x[i][j]]['class 1']=1
                theta_dict_new[x[i][j]]['class 2']=1

            if y[i]==0:
                if x[i][j] in theta_dict_new:
                    theta_dict_new[x[i][j]]['class 0'] +=1

            elif y[i]==1:
                if x[i][j] in theta_dict_new:
                    theta_dict_new[x[i][j]]['class 1'] +=1

            else:
                if x[i][j] in theta_dict_new:
                    theta_dict_new[x[i][j]]['class 2'] += 1
    

    
    
                    
    return theta_dict_new

theta_dict_naive = naive_bayes_dict(x_train, y_train)
theta_dict_naive1 = naive_bayes_dict(x_train, y_train)
theta_dict_naive2 = naive_bayes_dict(x_train, y_train)
theta_dict_naive5 = naive_bayes_dict(x_train, y_train)
theta_dict_naive10 = naive_bayes_dict(x_train, y_train)
theta_dict_naive25 = naive_bayes_dict(x_train, y_train)
theta_dict_naive50 = naive_bayes_dict(x_train, y_train)
theta_dict_naive100 = naive_bayes_dict(x_train, y_train)


theta_dict_naive_target_1 = naive_bayes_dict(x_domain_1, y_domain_1)
theta_dict_naive_target_2 = naive_bayes_dict(x_domain_2, y_domain_2)
theta_dict_naive_target_5 = naive_bayes_dict(x_domain_5, y_domain_5)
theta_dict_naive_target_10 = naive_bayes_dict(x_domain_10, y_domain_10)
theta_dict_naive_target_25 = naive_bayes_dict(x_domain_25, y_domain_25)
theta_dict_naive_target_50 = naive_bayes_dict(x_domain_50, y_domain_50)
theta_dict_naive_target_100 = naive_bayes_dict(x_domain_100, y_domain_100)


theta_dict_domain_1 = dict_domain(x_domain_1,y_domain_1, theta_dict_naive1)

theta_dict_domain_2 = dict_domain(x_domain_2, y_domain_2, theta_dict_naive2)


theta_dict_domain_5 = dict_domain(x_domain_5, y_domain_5, theta_dict_naive5)

theta_dict_domain_10 = dict_domain(x_domain_10, y_domain_10, theta_dict_naive10)

theta_dict_domain_25 = dict_domain(x_domain_25, y_domain_25, theta_dict_naive25)

theta_dict_domain_50 = dict_domain(x_domain_50, y_domain_50, theta_dict_naive50)

theta_dict_domain_100 = dict_domain(x_domain_100, y_domain_100, theta_dict_naive100)






def normalise_dict(dict_n):

    count0 = 0
    count1 = 0
    count2 = 0

    for key, value in enumerate(dict_n):
        count0+=dict_n[value]['class 0']
        count1+=dict_n[value]['class 1']
        count2 += dict_n[value]['class 2']


    for key, value in enumerate(dict_n):
        dict_n[value]['class 0']= dict_n[value]['class 0']/count0
        dict_n[value]['class 1']= dict_n[value]['class 1']/count1
        dict_n[value]['class 2']= dict_n[value]['class 2']/count2
        
        
    return dict_n

theta_dict_naive_new = normalise_dict(theta_dict_naive)

theta_dict_naive_new_target_1 = normalise_dict(theta_dict_naive_target_1)
theta_dict_naive_new_target_2 = normalise_dict(theta_dict_naive_target_2)
theta_dict_naive_new_target_5 = normalise_dict(theta_dict_naive_target_5)
theta_dict_naive_new_target_10 = normalise_dict(theta_dict_naive_target_10)
theta_dict_naive_new_target_25 = normalise_dict(theta_dict_naive_target_25)
theta_dict_naive_new_target_50 = normalise_dict(theta_dict_naive_target_50)
theta_dict_naive_new_target_100 = normalise_dict(theta_dict_naive_target_100)


theta_dict_new_domain_1 = normalise_dict(theta_dict_domain_1)
theta_dict_new_domain_2 = normalise_dict(theta_dict_domain_2)
theta_dict_new_domain_5 = normalise_dict(theta_dict_domain_5)
theta_dict_new_domain_10 = normalise_dict(theta_dict_domain_10)
theta_dict_new_domain_25 = normalise_dict(theta_dict_domain_25)
theta_dict_new_domain_50  = normalise_dict(theta_dict_domain_50)
theta_dict_new_domain_100 = normalise_dict(theta_dict_domain_100)


def predict_class(sent, dict_new):
    ans = np.log(phi)
    
    for word in sent:
        if word in dict_new:
            ans += np.log(np.array([dict_new[word]['class 0'],dict_new[word]['class 1'],dict_new[word]['class 2'] ]))
        else:
            ans += np.log(np.array([1/len(dict_new),1/len(dict_new),1/len(dict_new)]))
            
        
    
    y = np.argmax(ans)

    return y

def find_accuracy(val_set,val_preds, dict_new):
    y_pred = []
    for i in range(len(val_set)):
        s = predict_class(val_set[i], dict_new)
        y_pred.append(s) 
    count = 0 

    for i in range(len(val_preds)):
        if y_pred[i] == val_preds[i]:
            count+=1
    
    
    classes = np.unique(val_preds)

    confmat = np.zeros((len(classes), len(classes)))
    for i in range(len(val_preds)):
        true_class = val_preds[i]
     
        predicted_class = y_pred[i]
      
        confmat[true_class[0]][predicted_class] += 1

    count = 0
    for i in range(len(confmat)):
        for j in range(len(confmat)):
            if i==j:
                count+=confmat[i][j]

    acc = count/sum(sum(confmat))
        

    
        
    return acc, confmat



accuracy_0, confusion_matrix_0 = find_accuracy(x_domain_val, y_domain_val, theta_dict_naive )

accuracy_1,confusion_matrix_1 = find_accuracy(x_domain_val,y_domain_val, theta_dict_new_domain_1)
accuracy_2,confusion_matrix_2 = find_accuracy(x_domain_val,y_domain_val, theta_dict_new_domain_2)
accuracy_5,confusion_matrix_5 = find_accuracy(x_domain_val,y_domain_val, theta_dict_new_domain_5)
accuracy_10,confusion_matrix_10 = find_accuracy(x_domain_val,y_domain_val, theta_dict_new_domain_10 )
accuracy_25,confusion_matrix_25 = find_accuracy(x_domain_val,y_domain_val, theta_dict_new_domain_25)
accuracy_50,confusion_matrix_50 = find_accuracy(x_domain_val,y_domain_val,theta_dict_new_domain_50 )
accuracy_100,confusion_matrix_100 = find_accuracy(x_domain_val,y_domain_val, theta_dict_new_domain_100 )

accuracy_11,confusion_matrix_11 = find_accuracy(x_domain_val,y_domain_val, theta_dict_naive_new_target_1)
accuracy_21,confusion_matrix_21 = find_accuracy(x_domain_val,y_domain_val, theta_dict_naive_new_target_2)
accuracy_51,confusion_matrix_51 = find_accuracy(x_domain_val,y_domain_val, theta_dict_naive_new_target_5)
accuracy_101,confusion_matrix_101 = find_accuracy(x_domain_val,y_domain_val, theta_dict_naive_new_target_10)
accuracy_251,confusion_matrix_251 = find_accuracy(x_domain_val,y_domain_val, theta_dict_naive_new_target_25)
accuracy_501,confusion_matrix_501 = find_accuracy(x_domain_val,y_domain_val, theta_dict_naive_new_target_50)
accuracy_1001,confusion_matrix_1001 = find_accuracy(x_domain_val,y_domain_val, theta_dict_naive_new_target_100)

print()

print(f"domain adaption on 0% of target data is {accuracy_0}")

# print(f"target train from scratch accuracy  is {accuracy_sratch}")
print(f"domain adaption on 2% of target data is {accuracy_2}")
print(f"domain adaption on 5% of target data is {accuracy_5}")
print(f"domain adaption on 10% of target data is {accuracy_10}")
print(f"domain adaption on 25% of target data is {accuracy_25}")
print(f"domain adaption on 50% of target data is {accuracy_50}")
print(f"domain adaption on 100% of target data is {accuracy_100}")

print()
print(f"target train from scratch accuracy on 1% is {accuracy_11}")
print(f"target train from scratch accuracy on 2% is {accuracy_21}")
print(f"target train from scratch accuracy on 5% is {accuracy_51}")
print(f"target train from scratch accuracy on 10% is {accuracy_101}")
print(f"target train from scratch accuracy on 25% is {accuracy_251}")
print(f"target train from scratch accuracy on 50% is {accuracy_501}")
print(f"target train from scratch accuracy on 100% is {accuracy_1001}")

accuracy_adaptation = [accuracy_1, accuracy_2, accuracy_5, accuracy_10, accuracy_25, accuracy_50, accuracy_100]
accuracy_scratch = [accuracy_11, accuracy_21, accuracy_51, accuracy_101, accuracy_251, accuracy_501, accuracy_1001]


x = [1,2,5,10,25,50, 100]
plt.plot(x, accuracy_adaptation, label='domain adaptation')
plt.plot(x, accuracy_scratch, label='scratch')
plt.xlabel('train size')
plt.ylabel('validation accuracy')
plt.title('domain adaptation vs scratch learning plot')
plt.legend()
plt.savefig('f.jpg')