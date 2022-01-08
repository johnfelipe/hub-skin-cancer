from skimage import io
import base64
from tensorflow.keras.models import load_model
import numpy as np
import gradio
from src import moleimages

model = load_model("./models/mymodel-2.h5")

def predict(input):
    mimg = moleimages.MoleImages()
    X = mimg.load_image(input)
    y_pred = model.predict(X)
    return {"Benigno": float(y_pred[0][0]), "Cancerígeno": float(1-y_pred[0][0])}

examples=[["benign.png"], ["cancerous.png"]]

io = gradio.Interface(fn=predict, inputs='image', outputs='label', capture_session=True, examples=examples,thumbnail="https://raw.githubusercontent.com/gradio-app/hub-skin-cancer/master/thumbnail.png", analytics_enabled=False,title="IDENTIFICADOR DE CANCER DE PIEL", description="Predice si una imagen de la piel es cancerosa o no. Este modelo es EXPERIMENTAL y sólo debe utilizarse con fines de investigación. Por favor, consulte a un médico por cualquier motivo de diagnóstico.")
io.launch(share=True)
