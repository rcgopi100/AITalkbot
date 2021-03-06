#This is Talkbot neural network project
#this project copied form Neuraline code and developed on top of that
#First run this training program to train all chats which system can learn when you run
import json
import random
import pickle
import numpy as np

import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
labels = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        #print(word_list)
        words.extend(word_list)
        #print(words)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in labels:
            labels.append(intent['tag'])
            #print(labels)
#print(training_data)

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters] #edited
words = sorted(list(set(words))) #edited
#print(words)
labels = sorted(list(set(labels))) #edited

#print(labels)

pickle.dump(words, open('talkbot_words.pkl', 'wb'))
pickle.dump(labels, open('talkbot_labels.pkl', 'wb'))

training_set = []
output_empty = [0] * len(labels)
#print(output_empty)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    #print(word_patterns)
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    #print(word_patterns)
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[labels.index(doc[1])] = 1
    training_set.append([bag, output_row])

random.shuffle(training_set)
training_set = np.array(training_set)

train_x = list(training_set[:, 0])
train_y = list(training_set[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=500, batch_size=5, verbose=1)
model.save('talkbotnlp_model.h5', hist)
print("Done")




