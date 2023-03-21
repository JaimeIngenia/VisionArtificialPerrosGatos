import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2
import keras
import matplotlib
from tensorflow.keras.callbacks import TensorBoard

datos, metadatos = tfds.load('cats_vs_dogs' , as_supervised = True , with_info = True)

#print(metadatos)

#tfds.as_dataframe(datos['train'].take(5), metadatos)

plt.figure(figsize=(20,20))

TAMANO_IMG=100

# for i, (imagen, etiqueta) in enumerate(datos['train'].take(25)):
#   imagen = cv2.resize(imagen.numpy(),(TAMANO_IMG,TAMANO_IMG))
#   imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)


#   plt.subplot(5,5, i+1)
#   plt.xticks([])
#   plt.yticks([])
#   plt.imshow(imagen, cmap='gray')
  
datos_entrenamiento = []
for i, (imagen, etiqueta) in enumerate(datos['train']):
    imagen = cv2.resize(imagen.numpy(),(TAMANO_IMG,TAMANO_IMG))
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen = imagen.reshape(TAMANO_IMG, TAMANO_IMG, 1)

    datos:datos_entrenamiento.append([imagen, etiqueta]) 
    
#print(datos_entrenamiento[0])
print("Mi primer print")
print(len(datos_entrenamiento))

X = []  #IMAGENES DE ENTRADA (PIXELES)
Y = []  #ETIQUETAS PERRO O GATO

for imagen, etiqueta in datos_entrenamiento:
  X.append(imagen)
  Y.append(etiqueta)
  
  
  ###################### NORMALIZATION #################


X = np.array(X).astype(float) / 255
Y = np.array(Y)

print("Mi sEGUNDO print")
#print(Y)
#print(X)
print(X.shape)

###################### entrenamiento 10:02 #################
modeloDenso = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(100, 100, 1)),
    tf.keras.layers.Dense(150, activation='relu'),
    tf.keras.layers.Dense(150, activation='relu'),
    tf.keras.layers.Dense(1,   activation='sigmoid'),
])

modelo_CNN = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), activation = 'relu', input_shape=(100, 100, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation = 'relu'),
    tf.keras.layers.Dense(  1, activation = 'sigmoid')
])

modelo_CNN2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), activation = 'relu', input_shape=(100, 100, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(250, activation = 'relu'),
    tf.keras.layers.Dense(  1, activation = 'sigmoid')
])
###################### compilacion  #################
modeloDenso.compile(optimizer='adam',
                    loss="binary_crossentropy",
                    metrics=['accuracy'])
modelo_CNN.compile(optimizer='adam',
                    loss="binary_crossentropy",
                    metrics=['accuracy'])
modelo_CNN2.compile(optimizer='adam',
                    loss="binary_crossentropy",
                    metrics=['accuracy'])

###################### Entrenamiento  #################

tensorboardCNN2 = TensorBoard(log_dir='logsJaime/CNN2Jaime')
modelo_CNN2.fit(X,Y,batch_size=32,
                validation_split=0.15,
                epochs=3,
                callbacks=[tensorboardCNN2])