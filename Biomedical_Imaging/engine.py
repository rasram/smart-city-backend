import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from fastapi import UploadFile
import io

mri_model = tf.keras.models.load_model("Biomedical_Imaging/MRI/MRI_DenseNet121_Optimized.h5")
xray_model = tf.keras.models.load_model("Biomedical_Imaging/XRay/DenseNet121_XRay.h5")

async def run_model(image_type, file: UploadFile):
  if image_type not in ["MRI", "XRay"]:
    raise ValueError("Invalid image type. Must be 'MRI' or 'XRay'.")
  
  img_bytes = await file.read()
  img = keras_image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
  img_array = keras_image.img_to_array(img)
  img_array = img_array / 255.0
  img_array = tf.expand_dims(img_array, 0)

  if image_type == "MRI":
    model = mri_model
    tumors_types = {0: 'Glioma', 1: 'Meningioma', 2: 'No Tumor', 3: 'Pituitary'}
  else:
    model = xray_model
    tumors_types = {0: 'Normal', 1: 'Pneumonia'}

  predictions = model.predict(img_array)
  predicted_class = tf.argmax(predictions[0]).numpy()

  return tumors_types[predicted_class]