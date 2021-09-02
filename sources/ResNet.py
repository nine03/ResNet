import numpy as np
from tensorflow.keras.applications.resnet50 import  ResNet50, preprocess_input , decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = ResNet50(weights='imagenet')
model.summary()

img_path ='4.jpg'
img = load_img(img_path, target_size=(224,224))

x = img_to_array(img)
x = np.expand_dims(x,axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:',decode_predictions(preds,top=3)[0])
