import os
from flask import Flask, render_template


# initialize flask app
app = Flask(__name__, template_folder='webapp_templates')

@app.route('/')
def index():
	return render_template('index.html')



if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port)
	#app.run(debug=True) # optional