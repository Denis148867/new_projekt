import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

model = load_model('anymalmodel.h5')

img = image.load_img("wild-hamster-by-Julian-Rad-May-2023-RRJR.jpg",target_size=(150,150))
img_array = image.img_to_array(img)/255.0
img_array =np.expand_dims(img_array,axis=0)

pred = model.predict(img_array)
classes = os.listdir("C:\\Users\\hmeli\\PycharmProjects\\PythonProject\\dataset")
print(f"модель распознала: {classes[np.argmax(pred)]}")
