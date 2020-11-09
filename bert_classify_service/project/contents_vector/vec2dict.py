#!/bin/usr/python
#coding:utf-8
import pickle
import create_vectors as vec
bert_file_name = '/home/project/contents_vector/renren_sensi_vec2dict.pkl'
input_file = '/home/project/contents_vector/renren_sensit_words.csv'
import re

class vec2dict:
    def __init__(self):
        self.dic = {}
        self._read_dic()

    # 批量插入数据
    def add_batch_data(self, keys, values):
        for key, value in zip(keys, values):
            self.dic[key] = value

    # 插入单条数据
    def add_data(self, key, value):
        self.dic[key] = value

    # 根据key删除数据
    def delete_data(self, key):
        if self.dic and self.dic.get(key, ''):
            self.dic.pop(key)

    # 根据key获取数据
    def get_data(self, key):
        return self.dic.get(key, '')

    # 获取全部数据
    def get_all_data(self):
        return self.dic

    # 提交
    def commit(self):
        self._save_dic()

    def _save_dic(self):
        try:
            with open(bert_file_name, 'wb')as file:
                pickle.dump(self.dic, file)
                print ('bert data saved successfully')
        except:
            print ('save bert data failed')

    def _read_dic(self):
        try:
            with open(bert_file_name, 'rb')as file:
                self.dic = pickle.load(file)
        except FileNotFoundError:
            print ('local bert data is none')


if __name__ == '__main__':
    bd = vec2dict()
    bert = vec.BertVector()
    data = []
    vec = []
    f = open(input_file,'r')
    index =0
    cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")

    for lines in f:
        line = lines.strip()
        line = cop.sub('',line)    
        if len(line)==0:
            continue
        else:
            data.append(line)
            vec.append(bert.encode([line])[0])

    bd.add_batch_data(data, vec)
    bd.commit()
