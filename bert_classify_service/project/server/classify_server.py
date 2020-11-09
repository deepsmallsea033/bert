from flask import Flask,request
import json
import sys
sys.path.append("/home/project/bert_classify/")
import classify  
from sys import argv
from flask_cors import CORS
import tensorflow as tf
app=Flask(__name__)
CORS(app, supports_credentials=True)

@app.route('/classify/status',methods=['GET'])
def status():
    return 'OK'

@app.route('/classify',methods=['POST'])
def weibo_classify():
    info_weibo = request.form.get("info","")
    cls.input_queue.put([info_weibo])
    predict = cls.output_queue.get()[0]
    return json.dumps(str(predict.tolist().index(max(predict))))

if __name__=='__main__':
    cls = classify.BertClassify()
    cls.set_mode(tf.estimator.ModeKeys.PREDICT)
    app.run(debug=False,port=int(argv[1]),host='0.0.0.0')
