from wordcloud import WordCloud
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
path1 = 'Corona_train.csv'

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
print(x_train.shape)


def preprocess_data(x):
    new_x = []
    for sent in x:
        x = sent[0].split(' ')
        new_x.append(x)
        
    return new_x

x_train = preprocess_data(x_train)

class0_dict = {}
class1_dict = {}
class2_dict = {}

for i in range(len(x_train)):
    for j in range(len(x_train[i])):
        if y_train[i]==0:
            if x_train[i][j] not in class0_dict:
                class0_dict[x_train[i][j]]=1
            else:
                class0_dict[x_train[i][j]]+=1

        if y_train[i]==1:
            if x_train[i][j] not in class1_dict:
                class1_dict[x_train[i][j]]=1
            else:
                class1_dict[x_train[i][j]]+=1

        if y_train[i]==2:
            if x_train[i][j] not in class2_dict:
                class2_dict[x_train[i][j]]=1
            else:
                class2_dict[x_train[i][j]]+=1


print(len(class0_dict))
print(len(class1_dict))
print(len(class2_dict))

class0_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(class0_dict)
class1_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(class1_dict)
class2_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(class2_dict)

# Display the word cloud using matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(class0_wordcloud, interpolation='bilinear')
plt.axis('off')  # Remove axis labels and ticks

plt.savefig('class0_wordcloud.jpg')

plt.figure(figsize=(10, 5))
plt.imshow(class1_wordcloud, interpolation='bilinear')
plt.axis('off')  # Remove axis labels and ticks

plt.savefig('class1_wordcloud.jpg')

plt.figure(figsize=(10, 5))
plt.imshow(class2_wordcloud, interpolation='bilinear')
plt.axis('off')  # Remove axis labels and ticks

plt.savefig('class2_wordcloud.jpg')




