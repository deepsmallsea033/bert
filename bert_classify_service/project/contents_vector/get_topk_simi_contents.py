#!/bin/usr/python
#coding:utf-8
from annoy import AnnoyIndex
import pickle
import create_vectors as cv
vf = 768
vt = AnnoyIndex(vf, metric='angular') #采用余弦距离计算
vec2dict = {}
id2content_dict = {}
import re

def create_annoy_tree():
    with open('/home/project/contents_vector/renren_sensi_vec2dict.pkl','rb') as file:
        vec2dict = pickle.load(file)
    for index,content in enumerate(vec2dict):
        vt.add_item(index,vec2dict[content])
        id2content_dict.setdefault(index,content)
    vt.build(40)
    vt.save('/home/project/contents_vector/simi_vector_index')

def get_topk_simi(vec):
    u = AnnoyIndex(vf, metric='angular')
    u.load('/home/project/contents_vector/simi_vector_index')
    result = u.get_nns_by_vector(vec,2,include_distances=True)
    for ids,simi in zip(result[0],result[1]):
        print (id2content_dict[ids],0.5*(abs(1-simi))+0.5)
if __name__ == '__main__':
    create_annoy_tree()
    bert = cv.BertVector()
    cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")
    while True:
        line = input()
        line=cop.sub('',line)
        vec = bert.encode([line])[0]
        get_topk_simi(vec)

