#encoding=utf-8
'''
Created on Jan 9, 2015

@author: ckt
'''
import os
import codecs

import jieba

class JiebaSeg(object):
    '''
    用jieba分词来对语料进行切词，作为自己实现的分词模型的训练语料
    jieba分词参考：https://github.com/fxsjy/jieba
    '''
    def __init__(self, data_path):
        data_path = '%s/%s' % (os.path.abspath('./'), data_path)
        self._load_data(data_path)
        
    def _load_data(self, data_path):
        with open(data_path, 'r') as fin:
            lines = [line.strip() for line in fin]
        self.data = lines
    
    def _do_tag(self, seg_list):
        tags = []
        for word in seg_list:
            if len(word) == 1:
                tags.append('S')
            if len(word) >= 2:
                tags.append('B')
                for w in word[1:-1]:
                    tags.append('I')
                tags.append('E')
        return tags
    
    def cut_data_and_write(self, out_file):
        fout = codecs.open(out_file, 'w', 'utf8')
        for line in self.data:
            try:
                seg_list = jieba.cut(line)
                tags = self._do_tag(seg_list)
                tagged = ['%s/%s' % (word, tag) for (word, tag) in zip(line.decode('utf8'), tags)]
                print >>fout, '\t'.join(tagged)
            except Exception as e:
                print str(e), line
        fout.close()