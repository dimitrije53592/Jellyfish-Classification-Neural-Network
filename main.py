import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import image_dataset_from_directory
import logging
tf.get_logger().setLevel(logging.ERROR)

main_path = './dataset'

img_size = (224, 224)
batch_size = 32

train_dataset = image_dataset_from_directory(main_path, 
                                        shuffle=True, 
                                        batch_size=batch_size, 
                                        image_size=img_size, 
                                        validation_split=0.2, 
                                        subset="training",
                                        seed=40)

validation_dataset = image_dataset_from_directory(main_path,
                                                  shuffle=True,
                                                  batch_size=batch_size,
                                                  image_size=img_size,
                                                  validation_split=0.2,
                                                  subset="validation",
                                                  seed=40) 

total_dataset = train_dataset.concatenate(validation_dataset)

class_names = train_dataset.class_names
num_of_classes = len(class_names)

import numpy as np
import matplotlib.pyplot as plt

all_labels = np.concatenate([y for x, y in total_dataset], axis=0)

class_counts = np.bincount(all_labels)

plt.figure(figsize=(10, 5))
plt.bar(class_names, class_counts)
plt.xticks(rotation=90)
plt.title('Number of images per class in the dataset')
plt.show()

import matplotlib.pyplot as plt

class_images = {}

for images, labels in total_dataset:
    for image, label in zip(images, labels):
        if label.numpy() not in class_images:
            class_images[label.numpy()] = image.numpy()
        if len(class_images) == num_of_classes:
            break
    if len(class_images) == num_of_classes:
        break

plt.figure(figsize=(10, num_of_classes))
for i, (class_index, image) in enumerate(class_images.items()):
    plt.subplot(1, num_of_classes, i+1)
    plt.imshow(image.astype("uint8"))
    plt.title(class_names[class_index])
    plt.axis("off")
plt.show()


from keras import Sequential
from keras import layers
from keras.optimizers.legacy import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping

def make_model(hp):

    dropout_rate = hp.Choice('dropout_rate', [0.2, 0.3, 0.4])

    model = Sequential([
        layers.Rescaling(1./255, input_shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(dropout_rate),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(dropout_rate),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_of_classes, activation='softmax')
        ])

    model.summary()

    model.compile(optimizer=Adam(learning_rate=0.0001),
                    loss=SparseCategoricalCrossentropy(),
                    metrics='accuracy')
    
    return model

import keras_tuner as kt

tuner = kt.RandomSearch(make_model,
                        objective='val_accuracy',
                        overwrite=True,
                        max_trials=10)

model_checkpoint = ModelCheckpoint(filepath='model.h5', monitor='val_accuracy', verbose=0, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=0)

tuner.search(train_dataset,
             epochs=30,
             batch_size=32,
             validation_data=(validation_dataset),
             callbacks=[early_stopping],
             verbose=1)

model = tuner.get_best_models()
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

model = tuner.hypermodel.build(best_hyperparameters)           

history = model.fit(train_dataset, validation_data=validation_dataset, epochs=150, verbose=1, callbacks=[model_checkpoint, early_stopping])

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Loss')
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

train_labels = np.array([])
train_predictions = np.array([])
for images, lab in train_dataset:
    train_labels = np.append(train_labels, lab)
    train_predictions = np.append(train_predictions, np.argmax(model.predict(images, verbose=0), axis=1))

train_cm = confusion_matrix(train_labels, train_predictions, normalize='true')    
train_cmDisplay = ConfusionMatrixDisplay(train_cm, display_labels=class_names)
train_cmDisplay.plot()
plt.show()

test_labels = np.array([])
test_predictions = np.array([])
for images, lab in validation_dataset:
    test_labels = np.append(test_labels, lab)
    test_predictions = np.append(test_predictions, np.argmax(model.predict(images, verbose=0), axis=1))

test_cm = confusion_matrix(test_labels, test_predictions, normalize='true')    
test_cmDisplay = ConfusionMatrixDisplay(test_cm, display_labels=class_names)
test_cmDisplay.plot()
plt.show()

from sklearn.metrics import accuracy_score
print('Accuracy: ', str(100*accuracy_score(test_labels, test_predictions)) + '%')
    