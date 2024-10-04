import pandas as pd
import numpy as np
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

def preprocess_data(x):
    new_x = []
    for sent in x:
        x = sent[0].split(' ')
        new_x.append(x)
        
    return new_x

x_train = preprocess_data(x_train)



def get_prior(y):
    phi = np.zeros(3)
    for i in range(len(y_train)):
        if y_train[i] == 0: 
            phi[0]+=1
        if y_train[i] == 1:
            phi[1]+=1
        if y_train[i] == 2:
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



ans = np.log(phi)


def predict_class(sent):
    ans = np.log(phi)
 
    for word in sent:
        if word in theta_dict_new:
            ans += np.log(np.array([theta_dict_new[word]['class 0'],theta_dict_new[word]['class 1'],theta_dict_new[word]['class 2'] ]))
        else:
            ans += np.log(np.array([1/len(theta_dict),1/len(theta_dict),1/len(theta_dict)]))
            
        
    
    y = np.argmax(ans)
    return y
            
    
x_val = preprocess_data(x_val)

def find_accuracy(val_set,val_preds):
    
    y_pred = []
    for i in range(len(val_set)):
        s = predict_class(val_set[i])
        y_pred.append(s) 
    count = 0 

    for i in range(len(y_val)):
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
            if i == j:
                count+=confmat[i][j]

    acc = count/sum(sum(confmat))
        

    
        
    return acc, confmat

accuracy, confusion_matrix = find_accuracy(x_val,y_val)
accuracy1, confusion_matrix1 = find_accuracy(x_train, y_train)
print(f"Basic naive bayes classifier accuracy on train set is {accuracy1}")
print()
print(f"Basic naive bayes classifier accuracy on validation set is  {accuracy} ")
print()
print(f"The confusion matrix for the basic naive bayes classifier is{confusion_matrix}")

        
        
