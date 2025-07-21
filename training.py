#Proyecto final PLN   - 19/7/2025
#Fecha de inicio del proyecto - 18/7/2025
#Segundo intento, el intento uno fue desechado debido a que no habia presupuesto para obtener un modelo de open AI y hacer las cosas bonito

#Hubo que instalar las librerias con un pip install en la terminal de: nltk, numpy, tensorflow y keras
import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read()) #Estas son las intenciones osea el jason

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

words=[]
classes = []
documents=[]
ignore_letters=['?', '!', '.', ',', '¿', '-', '_']

#El bucle que maneja las intenciones y las tokeniza
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words= [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training =[]
output_empty = [0]*len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    outpout_row = list(output_empty)
    outpout_row[classes.index(document[1])] = 1
    training.append([bag, outpout_row])
random.shuffle(training)
training = np.array(training, dtype=object)
#Esto no parece tener fin, el nltk no funciona, los imports no importaban, el try no intenta, las variables no varian, los bucles no se repiten
#Ya todo se soluciono *Pulgar arriba *


#print(training)

train_x = list(training[:,0])
train_y = list(training[:,1])

#ni me pregunten que hice aqui abajo porque ni yo se, solo entendi algo de las neuronas, o bueno aqui se crea la red neuronal es una buena definicion
model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#Aqui abajo entreno al modelo, pero no entreno lo fisico, la red neuronal a alcanzado un estado de iluminacion que se dio cuenta que, carente de cuerpo fisico decidio entrnar lo mas poderoso
#La mente
sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])   #Aqui me comi la e en "crossentropy" y estuve 10 minutos buscando el error anoten eso para el pdf
train_process = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

#Aqui guardo el modelo para usarlo en el otro archivo
model.save("chatbot_UG.keras") #Primero use .h5 pero la ejecucion me dijo muy amablemente: ADVERTENCIA: ESE TIPO DE ARCHIVO ES VIEJO UN VEGESTORIO, CAMBIATE A KERAS EL NUEVO, FUERA LO VIEJO ENTRA LO NUEVO
