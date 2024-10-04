import pandas as pd
import numpy as np
import random


path2 = 'Corona_validation.csv'

def data_loader(path):
    data = pd.read_csv(path)

    x = np.array(data['CoronaTweet'])
    x = np.reshape(x, (len(x),1))
    y = np.array(data['Sentiment'])
    y = np.reshape(y,(len(y)))

    for i in range(len(y)):
        if y[i] == 'Positive':
            y[i]=0
        elif y[i] =='Negative':
            y[i]=1
        else:
            y[i]=2

    return x, y



    
x_val, y_val = data_loader(path2)

y_random_pred = np.zeros((len(y_val)))
for i in range(len(y_random_pred)):
    y_random_pred[i] = random.randint(0,2)

def find_accuracy(val_preds, val_ground):
    count = 0
    for i in range(len(val_preds)):
        if val_preds[i]==val_ground[i]:
            count+=1
            


    classes = np.unique(val_ground)

    confmat = np.zeros((len(classes), len(classes)))

    for i in range(len(val_preds)):
        true_class = val_ground[i]
        
        predicted_class = val_preds[i]
        
        
        
      
        confmat[int(true_class)][int(predicted_class)] += 1
        
        

    count = 0
    for i in range(len(confmat)):
        for j in range(len(confmat)):
            if i == j:
                count+=confmat[i][j]

    acc = count/sum(sum(confmat))
    
    return acc, confmat

accuracy, confusion_matrix = find_accuracy(y_random_pred,y_val)

print(f"accuracy for randomly predicting a class is {accuracy}")
print(f"confusion matrix for randomly predicting a class is {confusion_matrix}")

y_positive = np.zeros((len(y_val)))
print(len(y_positive))

accuracy_positive, confusion_matrix_positive = find_accuracy(y_positive,y_val)

print(f"accuracy for predicting all classes as 0 is {accuracy_positive}")
print(f"onfusion matrix for randomly predicting a class is {confusion_matrix_positive}")