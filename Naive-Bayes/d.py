import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm
from urllib.parse import urlparse
from tqdm import tqdm

path1 = 'Corona_train.csv'
path2 = 'Corona_validation.csv'

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


x_train, y_train = data_loader(path1)
    
x_val, y_val = data_loader(path2)

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


def build_vocab(x):
    dict_train = {}
    count = 0
    for lst in x:
        for i in range(len(lst)):
            if lst[i] not in dict_train:
                dict_train[lst[i]] = count
                count = count+1
                
    return dict_train

dict_new = build_vocab(x_train)

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



def bigram_dict(x_train):
    theta_dict = {}

    
    for i in tqdm(range(len(x_train))):
        for j in range((len(x_train[i]))):
            if x_train[i][j] not in theta_dict:
                theta_dict[x_train[i][j]]={}
                theta_dict[x_train[i][j]]['class 0']=1
                theta_dict[x_train[i][j]]['class 1']=1
                theta_dict[x_train[i][j]]['class 2']=1

            if y_train[i]==0:
                if x_train[i][j] in theta_dict:
                    theta_dict[x_train[i][j]]['class 0'] +=1

            elif y_train[i]==1:
                if x_train[i][j] in theta_dict:
                    theta_dict[x_train[i][j]]['class 1'] +=1

            else:
                if x_train[i][j] in theta_dict:
                    theta_dict[x_train[i][j]]['class 2'] += 1
    

    
    for i in tqdm(range(len(x_train))):
        for j in range((len(x_train[i])-1)):
            if x_train[i][j]+x_train[i][j+1] not in theta_dict:
                theta_dict[x_train[i][j]+x_train[i][j+1]]={}
                theta_dict[x_train[i][j]+x_train[i][j+1]]['class 0']=1
                theta_dict[x_train[i][j]+x_train[i][j+1]]['class 1']=1
                theta_dict[x_train[i][j]+x_train[i][j+1]]['class 2']=1

            if y_train[i]==0:
                if x_train[i][j]+x_train[i][j+1] in theta_dict:
                    theta_dict[x_train[i][j]+x_train[i][j+1]]['class 0'] +=1

            elif y_train[i]==1:
                if x_train[i][j]+x_train[i][j+1] in theta_dict:
                    theta_dict[x_train[i][j]+x_train[i][j+1]]['class 1'] +=1
            else:   
                if x_train[i][j]+x_train[i][j+1] in theta_dict:
                    theta_dict[x_train[i][j]+x_train[i][j+1]]['class 2'] +=1 
                    
    return theta_dict

def trigram_dict(x_train):
    theta_dict = {}

    
    for i in tqdm(range(len(x_train))):
        for j in range((len(x_train[i]))):
            if x_train[i][j] not in theta_dict:
                theta_dict[x_train[i][j]]={}
                theta_dict[x_train[i][j]]['class 0']=1
                theta_dict[x_train[i][j]]['class 1']=1
                theta_dict[x_train[i][j]]['class 2']=1

            if y_train[i]==0:
                if x_train[i][j] in theta_dict:
                    theta_dict[x_train[i][j]]['class 0'] +=1

            elif y_train[i]==1:
                if x_train[i][j] in theta_dict:
                    theta_dict[x_train[i][j]]['class 1'] +=1

            else:
                if x_train[i][j] in theta_dict:
                    theta_dict[x_train[i][j]]['class 2'] += 1
    

    
    for i in tqdm(range(len(x_train))):
        for j in range((len(x_train[i])-1)):
            if x_train[i][j]+x_train[i][j+1] not in theta_dict:
                theta_dict[x_train[i][j]+x_train[i][j+1]]={}
                theta_dict[x_train[i][j]+x_train[i][j+1]]['class 0']=1
                theta_dict[x_train[i][j]+x_train[i][j+1]]['class 1']=1
                theta_dict[x_train[i][j]+x_train[i][j+1]]['class 2']=1

            if y_train[i]==0:
                if x_train[i][j]+x_train[i][j+1] in theta_dict:
                    theta_dict[x_train[i][j]+x_train[i][j+1]]['class 0'] +=1

            elif y_train[i]==1:
                if x_train[i][j]+x_train[i][j+1] in theta_dict:
                    theta_dict[x_train[i][j]+x_train[i][j+1]]['class 1'] +=1
            else:   
                if x_train[i][j]+x_train[i][j+1] in theta_dict:
                    theta_dict[x_train[i][j]+x_train[i][j+1]]['class 2'] +=1 
                    
    for i in tqdm(range(len(x_train))):
        for j in range((len(x_train[i])-2)):
            if x_train[i][j]+x_train[i][j+1]+x_train[i][j+2] not in theta_dict:
                theta_dict[x_train[i][j]+x_train[i][j+1]+x_train[i][j+2]]={}
                theta_dict[x_train[i][j]+x_train[i][j+1]+x_train[i][j+2]]['class 0']=1
                theta_dict[x_train[i][j]+x_train[i][j+1]+x_train[i][j+2]]['class 1']=1
                theta_dict[x_train[i][j]+x_train[i][j+1]+x_train[i][j+2]]['class 2']=1

            if y_train[i]==0:
                if x_train[i][j]+x_train[i][j+1]+x_train[i][j+2] in theta_dict:
                    theta_dict[x_train[i][j]+x_train[i][j+1]+x_train[i][j+2]]['class 0'] +=1

            elif y_train[i]==1:
                if x_train[i][j]+x_train[i][j+1] in theta_dict:
                    theta_dict[x_train[i][j]+x_train[i][j+1]+x_train[i][j+2]]['class 1'] +=1
            else:   
                if x_train[i][j]+x_train[i][j+1] in theta_dict:
                    theta_dict[x_train[i][j]+x_train[i][j+1]+x_train[i][j+2]]['class 2'] +=1
                    
    return theta_dict

theta_dict = bigram_dict(x_train)
theta_dict_tri = trigram_dict(x_train)


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

theta_dict_new = normalise_dict(theta_dict)
theta_dict_tri_new = normalise_dict(theta_dict_tri)



count0 = 0
for key,value in enumerate(theta_dict_new):
    count0+=theta_dict_new[value]['class 2']



x_val = preprocess_data(x_val,stop_words)


def predict_class(sent, dict_new):
    ans = np.log(phi)
    
    for word in sent:
        if word in theta_dict_new:
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
    acc = count/len(val_preds)
    
    classes = np.unique(val_preds)

    confmat = np.zeros((len(classes), len(classes)))
    for i in range(len(val_preds)):
        true_class = val_preds[i]
     
        predicted_class = y_pred[i]
      
        confmat[true_class[0]][predicted_class] += 1
        

    
        
    return acc, confmat

accuracy1, confusion_matrix1 = find_accuracy(x_val,y_val, theta_dict_new)

print(f"accuracy for bigram naive bayes is {accuracy1}")
print()
print(f"confusion matrix for bigram naive bayes is {confusion_matrix1}")
accuracy2, confusion_matrix2 = find_accuracy(x_val,y_val, theta_dict_tri_new)

print(f"accuracy for trigram naive bayes is {accuracy2}")
print()
print(f"confusion matrix for trigram naive bayes is {confusion_matrix2}")