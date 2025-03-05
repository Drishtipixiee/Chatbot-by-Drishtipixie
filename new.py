
import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer

# ensure required NLTK downloads
nltk.download('wordnet')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()

# load intents.json
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignoreletters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

      # lemmatizing words
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreletters]
words = sorted(set(words))

# sorting classes
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# training data preparation
training = []
outputEmpty = [0] * len(classes)

for document in documents:
    
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]

    for word in words:
        bag.append(1 if word in wordPatterns else 0)
    
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

# shuffle and convert to numpy array
random.shuffle(training)
training = np.array(training, dtype=np.float32)  # Ensure numeric dtype

trainX = training[:, :len(words)]
trainY = training[:, len(words):]




# build neural network Model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation="softmax"))

    # compile model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# train model
hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)

# save model
model.save('chatbot_simplilearnmodel.h5')

print("Executed")
