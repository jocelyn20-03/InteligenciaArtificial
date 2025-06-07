
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
model = Sequential([
    Flatten(input_shape=(28, 28)),    
    Dense(128, activation='relu'),    
    Dense(64, activation='relu'),     
    Dense(10, activation='softmax')   
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nPrecisión en el conjunto de prueba: {test_acc:.4f}")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión validación')
plt.title('Precisión del modelo')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida validación')
plt.title('Pérdida del modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.show()
import numpy as np
index = np.random.randint(0, x_test.shape[0])
img = x_test[index]
true_label = np.argmax(y_test[index])
prediction = model.predict(img.reshape(1, 28, 28))
predicted_label = np.argmax(prediction)
plt.imshow(img, cmap='gray')
plt.title(f"Etiqueta real: {true_label} - Predicción: {predicted_label}")
plt.axis('off')
plt.show()
