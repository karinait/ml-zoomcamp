import tensorflow.lite as tflite
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def prepare_input(x):
    rescale_factor = 1./255
    return x*rescale_factor 
    

interpreter = tflite.Interpreter(model_path='curly_vs_straight.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def predict(url):

	img = download_image(url)
	img = prepare_image(img, target_size=(200,200))

	x = np.array(img)
	X = np.array([x], dtype=np.float32)
	X = prepare_input(X)

	interpreter.set_tensor(input_index, X)
	interpreter.invoke()
	preds = interpreter.get_tensor(output_index)
	return float(preds[0,0])
	
def lambda_handler(event, context):
	url = event['url']
	pred = predict(url)
	result = {
		'prediction': pred
		}
	return result






