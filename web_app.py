import os
import random  # for randomly picking images and styles
import re
from flask import Flask, render_template, request
import logic.StyleTransfer as lg
from flask_socketio import SocketIO, emit
import json
from flask_mail import Mail, Message
import configparser

# initialize flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

mail=Mail(app)

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
socketio = SocketIO(app, ping_timeout=1200)

IMAGELIST = ["2.png", "faca.jpg", "van_gough.jpg"]
STYLELIST = ["2.png", "faca.jpg", "van_gough.jpg"]

# regex
VALIDMAIL = re.compile(r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$")

# main page
@app.route('/')
def index():
    return render_template('index.html')


# user clicked sendToMail button
@socketio.on('sendToMail')
def stylizeEvent(json_string):
    data = json.loads(json_string['content'])

    img = re.search(r'base64,(.*)', data['img']).group(1)
    style = re.search(r'base64,(.*)', data['style']).group(1)

    # check validity of email address (basic)
    if not VALIDMAIL.match(data['mail']):
        print("Invalid mail address")
        return
    else:
        emit('willSendMail', data['mail'])
        msg = Message('AI Doodle Result', sender = config.get('DEFAULT', 'MAIL_USERNAME'), recipients = [data['mail']])
        msg.body = "We are sending you your stylized image."

        style_transfer = lg.StyleTransfer(width=200, height=200, content_image_base64=img,
                                          style_image_base64=style, iterations=10, max_fun=20)
        result = style_transfer.transfer()#.decode("utf-8")
        msg.attach("result.jpg", 'image/jpg', result)#'application/octect-stream' "image/jpg"
        mail.send(msg)


    style_transfer = lg.StyleTransfer(width=200, height=200, content_image_base64=img,
                                      style_image_base64=style, iterations=1, web_socket_channel='updateresult',
                                      max_fun=20)
    # result = style_transfer.transfer().decode("utf-8")


# emit('updateresult', result)

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
    socketio.run(app)
# app.run(debug=True) # optional
