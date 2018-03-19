from flask import Flask, render_template, request
import re
import configparser
from flask_mail import Mail, Message
import logic.StyleTransfer as lg

config = configparser.RawConfigParser()
config.read('secret.properties')

app = Flask(__name__)
app.config['MAIL_SERVER'] = config.get('DEFAULT', 'MAIL_SERVER')
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = config.get('DEFAULT', 'MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = config.get('DEFAULT', 'MAIL_PASSWORD')
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)



@app.route('/stylize/', methods=['POST'])
def stylize():
    data = request.get('data')

    img = re.search(r'base64,(.*)', data['img']).group(1)
    style = re.search(r'base64,(.*)', data['style']).group(1)

    print('Start to do the work')
    msg = Message('AI Doodle Result', sender=config.get('DEFAULT', 'MAIL_USERNAME'), recipients=[data['mail']])
    msg.body = "We are sending you your stylized image."

    style_transfer = lg.StyleTransfer(width=500, height=500, content_image_base64=img,
                                      style_image_base64=style, iterations=10, max_fun=20)
    result = style_transfer.transfer()  # .decode("utf-8")
    msg.attach("result.jpg", 'image/jpg', result)  # 'application/octect-stream' "image/jpg"
    mail.send(msg)