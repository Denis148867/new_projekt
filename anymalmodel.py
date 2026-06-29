from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D


datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  #20%
)

train_data=datagen.flow_from_directory(
    "C:\\Users\\hmeli\\PycharmProjects\\PythonProject\\dataset",
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data=datagen.flow_from_directory(
    "C:\\Users\\hmeli\\PycharmProjects\\PythonProject\\dataset",
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

model = Sequential([
    Conv2D(32, (3,3),activation="relu", input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(256,activation="relu"),
    Dense(3, activation="softmax")
])

model.compile(optimizer="adam", loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_data,validation_data=val_data,epochs=20)
model.save("anymalmodel.h5")