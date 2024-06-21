from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image


model = load_model("Tomato_model_1_ep_15_batch_64.h5")

def preprossingImagefn(data):
   image = BytesIO(data)

    # Load and preprocess the image
   img = tf.keras.preprocessing.image.load_img(image, target_size=(224, 224))  # Resize to the input size of the model
   img_array = tf.keras.preprocessing.image.img_to_array(img)  # Convert image to numpy array
   img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
   img_array /= 255.0  # Normalize the image if the model was trained on normalized images
   return img_array
 
 
 
 
def TomatoLeaf_D_pred(data):
  predictions = model.predict(preprossingImagefn(data=data))

# Make the prediction
  pred_class_index = np.argmax(predictions)
  outputname =[
 'Bacterial_spot',
 'Early_blight',
 'Late_blight',
 'Leaf_Mold',
 'Septoria_leaf_spot',
 'Spider_mites Two-spotted_spider_mite',
 'Target_Spot',
 'Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato_mosaic_virus',
 'healthy']
# Assuming a classification model
  confident =predictions[0][[pred_class_index]]*100  
  return outputname[pred_class_index],confident[0]
