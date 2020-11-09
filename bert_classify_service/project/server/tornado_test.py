from flask import Flask,request
import json
import sys
import tornado.wsgi
import tornado.httpserver
sys.path.append("/home/project/bert_classify/")
import classify  
from sys import argv
from flask_cors import CORS
import tensorflow as tf
app=Flask(__name__)
CORS(app, supports_credentials=True)

cls = classify.BertClassify()
cls.set_mode(tf.estimator.ModeKeys.PREDICT)
@app.route('/classify/status',methods=['GET'])
def status():
    return 'OK'

@app.route('/classify',methods=['POST'])
def weibo_classify():
    info_weibo = request.form.get("info","")
    cls.input_queue.put([info_weibo])
    predict = cls.output_queue.get()[0]
    return json.dumps(str(predict.tolist().index(max(predict))))

def start_tornado(app,port =9999):
    http_server = tornado.httpserver.HTTPServer(tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    tornado.ioloop.IOLoop.instance().start()

def start_all(app):
    start_tornado(app,9999)
if __name__=='__main__':
    start_all(app)
