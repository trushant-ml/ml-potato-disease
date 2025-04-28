import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np
#from tensorflow import keras
from tensorflow.keras import layers
#from tensorflow.keras.layers.experimental import preprocessing

IMAGE_SIZE = 256
BATCH_SIZE = 32   #sTD BATCH SIZE
EPOCHS = 50
CHANNELS = 3
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    shuffle=True,
    image_size = (IMAGE_SIZE,IMAGE_SIZE),
    batch_size = BATCH_SIZE
)

print("TensorFlow version:", tf.__version__)

#Above code output will show "Found 2152 files belonging to 3 classes"
#3 clases as there are three folders and 2152 files/images

class_names = dataset.class_names
print(class_names)  #3 folders

print(len(dataset))  #68 as batch size is 32 68*32

for image_batch, label_batch in dataset.take(1):
    print(image_batch.shape)    #(32, 256, 256, 3) batch,size,3-channel-rgb
    print(label_batch.numpy())  #convert tensor to numpy

#visulizze image

for image_batch, label_batch in dataset.take(1):
    plt.imshow(image_batch[0].numpy().astype("uint8"))  #will show iage in collab or huypter
    plt.title(class_names[label_batch[0]])
    plt.axis("off")

##80% TRAINING DS | 20 (10% VALIDATE & 10% TEST)
train_size = 0.8
print(len(dataset)*train_size)  #54

train_ds = dataset.take(54)

test_ds = dataset.skip(54)

val_size=0.1
print(len(dataset)*val_size)   #6.8

val_ds = test_ds.take(6)
test_ds = test_ds.skip(6)

#or create a function for the same task
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):

    ds_size = len(ds)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)

    val_ds = ds.skip(train_size).take(val_size)
    test_ds =  ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

#cache image so that next time it fetch from memory
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

#resize so that ir can work with any other DS
'''
resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])

#data argumentation layer so that it can recognize rotated pic
data_argumentation = tf.keras.Sequential([
    layers.experimental.preprosessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprosessing.RandomRotation(0,2),
])
'''

resize_and_rescale = keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1.0/255)
])

# Data augmentation layer to recognize rotated pictures
data_argumentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),  # Note: RandomRotation expects a float (like 0.2 = 20%), not (0,2)
])

#Creating a Model #64 & kernal size based on try and error
input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3
model = models.Sequential([
        resize_and_rescale,
        data_argumentation,
        layers.Conv2D(32, (3,3), activation='relu', input_shape = input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
])

model.build(input_shape=input_shape)
print(model.summary())

model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds
)

#print(history)

scores = model.evaluate(test_ds)
print("Scores ", scores)

print(history.params)
print(history.params.keys())
print(history.history['accuracy'])

#Plot accuract chart
#dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figure=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

for images_batch, labels_batch in test_ds.take(1):

    first_image = image_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()

    print("First image to predict")
    plt.imshow(first_image)
    print("First image actual label: ", class_names[first_label])

    batch_prediction = model.predict(images_batch)   #model will predict for 32 images
    print("predicted label:",batch_prediction[0])   #prediction for 1st image in batch
    print("predicted label:", class_names[np.argmax(batch_prediction[0])]) # its array and we meed max probablility

#Writing a fucntion below which will take model & image as input and will do the prrediction
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)  #create batch

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

plt.figure(figsize=(15, 15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))

        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]]

        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}. \nConfidence: {confidence}%")
        plt.axis("off")

"""
#Save Model
model.save('model.h5')
"""
