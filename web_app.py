import os
import random # for randomly picking images and styles
import json # for decoding ajax
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
	data = request.get_data().decode("utf-8")
	data = json.loads(data)
	#print(data['i']) # image in base64
	#print(data['s']) # style in base64
	return "static/faca.jpg"


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