import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
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
  
        new_x.append(lst)
        
     
        
    return new_x

x_train = preprocess_data(x_train, stop_words)

def get_prior(y):
    phi = np.zeros(3)
    for i in range(len(y)):
        if y[i] == 0: 
            phi[0]+=1
        if y[i] == 1:
            phi[1]+=1
        if y[i] == 2:
            phi[2]+=1

    phi = phi/len(y_train)
    
    return phi

phi = get_prior(y_train)
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


def normalise_dict(dict_n):

    count0 = 0
    count1 = 0
    count2 = 0

    for key, value in enumerate(dict_n):
        count0+=dict_n[value]['class 0']
        count1+=dict_n[value]['class 1']
        count2 += dict_n[value]['class 2']


    for key, value in enumerate(theta_dict):
        dict_n[value]['class 0']= dict_n[value]['class 0']/count0
        dict_n[value]['class 1']= dict_n[value]['class 1']/count1
        dict_n[value]['class 2']= dict_n[value]['class 2']/count2
        
        
    return dict_n

theta_dict_new = normalise_dict(theta_dict)


    

x_val = preprocess_data(x_val,stop_words)

def predict_class(sent):
    ans = np.log(phi)
    
    for word in sent:
        if word in theta_dict_new:
            ans += np.log(np.array([theta_dict_new[word]['class 0'],theta_dict_new[word]['class 1'],theta_dict_new[word]['class 2'] ]))
        else:
            ans += np.log(np.array([1/len(theta_dict_new),1/len(theta_dict_new),1/len(theta_dict_new)]))
            
        
    
    y = np.argmax(ans)

    return y


def find_accuracy(val_set,val_preds):
    y_pred = []
    for i in range(len(val_set)):
        s = predict_class(val_set[i])
        y_pred.append(s) 

    classes = np.unique(val_preds)
    confmat = np.zeros((len(classes), len(classes)))
    for i in range(len(val_preds)):
        true_class = val_preds[i]
     
        predicted_class = y_pred[i]
      
        confmat[true_class[0]][predicted_class] += 1

    count = 0
    for i in range(len(confmat)):
        for j in range(len(confmat)):
            if i ==j:
                count+=confmat[i][j]

                
    acc = count/sum(sum(confmat))
        

    
        
    return acc, confmat

accuracy, confusion_matrix = find_accuracy(x_val,y_val)

print(f"accuracy after removing stopwords and stemming is {accuracy}")
print(f"confusion matrix after removing stopwords and stemming is {confusion_matrix}")