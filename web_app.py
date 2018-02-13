import os
import random # for randomly picking images and styles
import re
from flask import Flask, render_template, request


# initialize flask app
app = Flask(__name__)

IMAGELIST=["2.png", "faca.jpg", "van_gough.jpg"]
STYLELIST=["2.png", "faca.jpg", "van_gough.jpg"]

# main page
@app.route('/')
def index():
	return render_template('index.html')


# user clicked stylize
@app.route('/stylize/', methods=['POST'])
def style():
	data = request.form.to_dict() 

	# here we have image data in base64
	#img = re.search(r'base64,(.*)',data['img']).group(1)
	#style = re.search(r'base64,(.*)',data['style']).group(1)
	
	return data['img']


# randomly pick an image
@app.route('/randomimage/', methods=['GET'])
def randomimage():
	return "static/" + random.choice(IMAGELIST)


# randomly pick style
@app.route('/randomstyle/', methods=['GET'])
def randomstyle():
	return "static/" + random.choice(STYLELIST)


if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port)
	#app.run(debug=True) # optional