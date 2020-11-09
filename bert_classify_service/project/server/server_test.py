
#_*_coding:utf-8 _*_
  
import requests
import sys

def classify_test(weibo_info):
    url="http://127.0.0.1:9999/classify"
    data={"info":weibo_info}
    res=requests.post(url=url,data=data)
    try:
        results=res.json()
        print (results)
    except:
        print ('Error: chat_test:',input_info)

if __name__== '__main__':
    while True:
        input_info=input("我：")
        classify_test(input_info)

