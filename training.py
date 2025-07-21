# Proyecto final PLN   - 19/7/2025
# Fecha de inicio del proyecto - 18/7/2025
import random
import json
import pickle
import numpy as np
import nltk

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer

# Inicializamos un lematizador para convertir palabras a su forma base
# Ejemplo: Bailando -> Bailar
lemmatizer = WordNetLemmatizer()

# Leemos el archivo recibiendo un string que luego convertimos a un objeto json
intents = json.loads(open('intents.json').read())

# Descargamos los paquetes de nltk que usaremos
# punk_tab & punk: tokenizador de oraciones
# wordnet: Paquete relacionado a los bancos de palabras
# omw-1.4: Banco de palabras multilenguajes
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',', '¿', '-', '_']

# Por cada intencion se revisan patrones y realizamos las siguientes acciones
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenizamos el patron y lo guardamos en la lista de palabras
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)

        # Guardamos en una tupla el patron tokenizado y la tag de la intencion
        documents.append((word_list, intent["tag"]))
        # Si la tag no esta en las clases entonces se guardara
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lematizamos la oracion e ignoramos ciertos simbolos irrelevantes
words = [
    lemmatizer.lemmatize(word)
    for word in words
    if word not in ignore_letters
]

# Se pone la lista de palabras en un set para eliminar las repetidas
# para despues ordenarlas alfabeticamente
words = sorted(set(words))

# Serializamos las palabras y clases como archivos
# para despues leerlos en otro archivo
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
# Llenamos un arreglo con ceros del mismo tamaño de las clases
# Esta sera clonada en el siguiente bucle parautilizarla como plantilla de BoW
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    # Lematizamos los patrones y lo convertimos a minusculas
    word_patterns = [
        lemmatizer.lemmatize(word.lower())
        for word in word_patterns
    ]

    # Guardamos en el BoW si la palabra está presente 1, si no 0
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # Definimos la fila de salida
    output_row = list(output_empty)

    # Ponemos 1 en la posición de la clase correcta
    output_row[classes.index(document[1])] = 1
    # Guardamos en los datos de entrenamiento el BoW
    training.append([bag, output_row])

# Barajamos los datos de entrenamiento para evitar el sesgo
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Creamos la red neuronal
model = Sequential()
# Se agrega una neurona con la función ReLU para relaciones no lineales.
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
# Se agrega una neurona para evitar overfitting
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# Capa de salida de la red neuronal
# Usa softmax para obtener una probabilidad por cada clase
model.add(Dense(len(train_y[0]), activation='softmax'))

# Se aplica un algoritmo de optimizacion llamado Stochastic Gradient Descent
sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)

# Compilamos el modelo usando categorical_crossentropy que produce un one-hot
# array que contiene la coincidencia probable para cada categoría,
model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy']
)

# Entrenamos el modelo
train_process = model.fit(
    np.array(train_x),
    np.array(train_y),
    epochs=200,
    batch_size=5,
    verbose=1
)

# Guardamos el modelo para posteriormente usarlo en otra ejecucion
model.save("chatbot_UG.keras")
