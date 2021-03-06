import os
import random  # for randomly picking images and styles
import re
from flask import Flask, render_template, request
import logic.StyleTransfer as lg
# from flask_socketio import SocketIO, emit
import json
from flask_mail import Mail, Message
import configparser
from google.cloud import pubsub_v1
import psq
# from queue import do_the_work
from logic.StyleTransfer import do_the_work
from tasks import adder

from google.appengine.api import taskqueue




google_cloud_project = 'aidoodle-art'

publisher = pubsub_v1.PublisherClient()
subscriber = pubsub_v1.SubscriberClient()

q = psq.Queue(publisher, subscriber, google_cloud_project, async=False)



# initialize flask app
app = Flask(__name__)
application = app  # our hosting requires application in passenger_wsgi
app.config['SECRET_KEY'] = 'secret!'


mail = Mail(app)

config = configparser.RawConfigParser()
config.read('secret.properties')

app.config['MAIL_SERVER'] = config.get('DEFAULT', 'MAIL_SERVER')
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = config.get('DEFAULT', 'MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = config.get('DEFAULT', 'MAIL_PASSWORD')
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

# ping_timeout should be high enough
# socketio = SocketIO(app, ping_timeout=1200, ssl_context='adhoc')

IMAGELIST = ["2.png", "faca.jpg", "van_gough.jpg"]
STYLELIST = ["2.png", "faca.jpg", "van_gough.jpg"]

# regex
VALIDMAIL = re.compile(
    r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$")


# main page
@app.route('/')
def index():
    return render_template('index.html')




# user clicked sendToMail button
@app.route('/sendToMail/', methods=['POST'])
def send_to_mail():
    data = request.form.to_dict()
    #
    # img = re.search(r'base64,(.*)', data['img']).group(1)
    # style = re.search(r'base64,(.*)', data['style']).group(1)

    # check validity of email address (basic)
    if not VALIDMAIL.match(data['mail']):
        print("Invalid mail address")
        return
    else:
        print("ok")
        task = taskqueue.add(
            url='/stylize',
            target='v1.stylize-modul',
            params={'data': data})



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
    app.run(port=port)#host='0.0.0.0',
    # socketio.run(app)
# app.run(debug=True) # optional
