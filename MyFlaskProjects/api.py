import os
from flask import Flask, flash, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import keras
from keras.models import load_model
from keras import backend as K


UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		# if user does not select file, browser also
		# submit an empty part without filename
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			image = cv2.imread(os.path.dirname(os.path.realpath(__file__))+"/uploads/"+filename)
			color_result = getDominantColor(image)
			dogOrCat = catOrDog(image)
			#return redirect(url_for('upload_file',filename=filename)), jsonify({"key":
			return jsonify({"MainColor": color_result, "catOrDog": dogOrCat} )
	return '''
	<!doctype html>
	<title>API</title>
	<h1>API Running Successfully</h1>'''

def catOrDog(image):
	'''Determines if the image contains a cat or dog'''
	classifier = load_model('./models/cats_vs_dogs_V1.h5')
	image = cv2.resize(image, (150,150), interpolation = cv2.INTER_AREA)
	image = image.reshape(1,150,150,3) 
	res = str(classifier.predict_classes(image, 1, verbose = 0)[0][0])
	print(res)
	print(type(res))
	if res == "0":
		res = "Cat"
	else:
		res = "Dog"
	K.clear_session()
	return res

def getDominantColor(image):
	'''returns the dominate color among Blue, Green and Reds in the image '''
	B, G, R = cv2.split(image)
	B, G, R = np.sum(B), np.sum(G), np.sum(R)
	color_sums = [B,G,R]
	color_values = {"0": "Blue", "1":"Green", "2": "Red"}
	return color_values[str(np.argmax(color_sums))]


if __name__ == "__main__":
	app.run(host= '0.0.0.0', port=80)