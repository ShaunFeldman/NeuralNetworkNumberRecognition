import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#dataset of 60,000 images of 0-9 digits for training, 10,000 images for testing
mnist = tf.keras.datasets.mnist

#x = pixel to train, y = digit answer
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# #normalizing to 0-1 
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

model = tf.keras.models.Sequential()

# #flatten input shape from 28x28 pixels grid to one column of 784 pixels (784x1)
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

# #rectified linear unit (0 if negative, straight up linearly, 1 if >= 1)
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))

# #output layer (0-9 digits), softmax (makes sure all outputs sum = 1)
model.add(tf.keras.layers.Dense(10, activation='softmax'))



model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# #epochs (how often the model sees the same data)
history = model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

# #model complete YAY!
model.save('handwrittenmodel.keras')

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

###^^^^ Above was ran, model was created, trained, and saved
###Below now uses the model and tests it

# model = tf.keras.models.load_model('handwrittenmodel.keras')


loss, accuracy = model.evaluate(x_test, y_test)

#loss of approx 0.0946 GOOD
print(f"The model has a loss of: {loss}")

#Accuracy of approx 97% GOOD
print(f"The model has an accuracy of {accuracy}%")

image_number = 1
print(f"The following is my own created dataset")
#succesfully read my own created digits: 0, 6 and 7 were accurately recognized
while os.path.isfile(f"my_own_test_images/digit{image_number}.jpeg"):
    try:
        img = cv2.imread(f"my_own_test_images/digit{image_number}.jpeg")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1