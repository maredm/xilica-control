from flask import Flask, render_template
from flask_sse import sse
import json
import flask
import xilica
import pprint
import os
from dotenv import load_dotenv

load_dotenv() 

app = Flask(__name__)
app.config["REDIS_URL"] = os.environ.get("REDIS_URL")
app.register_blueprint(sse, url_prefix='/sse')

client = [None]

def sse_reset():
    sse.publish({"message": f"Not connected"}, type='connection')
    sse.publish({"message": f"Disconnected"}, type='status')
    sse.publish({"message": f"--"}, type='device_info')
    sse.publish({"message": f"--"}, type='requests')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/do')
def inde():
  sse.publish({'name': 'Pizza'}, type='customer')
  return render_template("index.html")

@app.route('/connect', methods=['POST', 'GET'])
def publish_hello():
    if client[0] is None:
      client[0] = xilica.Client('10.0.2.2', 1)
    else:
      client[0]._socket.close()
    try:
      client[0].connect()
      sse.publish({"message": f"Connected to {client[0].endpoint}:{xilica.PORT}"}, type='connection')
      sse.publish({"message": f"Syncing current setting from unit"}, type='status')
    except Exception as e:
      sse_reset()
      sse.publish({"message": f"Not connected - {str(e)}"}, type='connection')
    finally:
      replies = client[0].send(client[0].upsync().read())
      sse.publish({"message": f"Sync successful!"}, type='status')
      sse.publish({"message": json.dumps(client[0].dsp.metadata)}, type='device_info')
      sse.publish({"message": '<br>'.join([str(reply) for reply in replies])}, type='requests')
      client[0]._socket.close()
    return "Message sent!"

@app.route('/disconnect', methods=['POST', 'GET'])
def publish():
    try:
      client[0]._socket.close()
    except:
      pass
    sse_reset()
    return "Message sent!"
    