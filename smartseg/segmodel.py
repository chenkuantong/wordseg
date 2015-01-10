#encoding=utf-8
'''
Created on Jan 9, 2015

@author: ckt
'''
import math
import pickle

class SmartSegModel(object):
    '''
    classdocs
    '''
    def __init__(self):
        '''
        Constructor
        '''
        pass
    
    def feature_extract(self):
        self.feas = {} 
        N = 0
        for terms in self.data:
            words = [w for (w, tag) in terms]
            tags = [tag for (w, tag) in terms]
            L = len(tags)
            N += L
            for k in range(L):
                w_0 = words[k]
                tag = tags[k]
                #f1: (word, 0)
                v = self.feas.get(('f1',w_0), [])
                v.append(tag)
                self.feas[('f1',w_0)] = v
                
                if k > 0:
                    tag_n1 = tags[k-1]
                    word_n1 = words[k-1]
                else:
                    tag_n1 = 'S'
                    word_n1 = '-'
                    
                #f2: (tag, -1)
                v = self.feas.get(('f2', tag_n1), [])
                v.append(tag)
                self.feas[('f2', tag_n1)] = v
                
                #f3: (word, -1)
                v = self.feas.get(('f3', word_n1), [])
                v.append(tag)
                self.feas[('f3', word_n1)] = v
                
                if k+1 == L:
                    word_p1 = '-'
                else:
                    word_p1 = words[k+1]
                    
                #f4: (word, 1)
                v = self.feas.get(('f4', word_p1), [])
                v.append(tag)
                self.feas[('f4', word_p1)] = v
                
                #f5: (word, -1, word, 0)
                k = ('f5', w_0, word_n1)
                v = self.feas.get(k, [])
                v.append(tag)
                self.feas[k] = v
        
        removed_key_list = []
        
        for k,v in self.feas.iteritems():
            if len(v) <= 2:
                removed_key_list.append(k)
        
        for k in removed_key_list:
            self.feas.pop(k)
            
        print '语料中总共%i个标注结果, 解析出 %i个特征' % (N, len(self.feas))
        fea_idx = 0
        self.feas_vec = []
        for k in self.feas:
            self.feas[k] = fea_idx
            fea_idx += 4
            
        for terms in self.data:
            words = [w for (w, tag) in terms]
            tags = [tag for (w, tag) in terms]
            L = len(tags)
            for k in range(L):
                fea = []
                w_0 = words[k]
                tag = tags[k]
                #f1: (word, 0)
                if ('f1', w_0) in self.feas:
                    fea.append(self.feas[('f1', w_0)])
                
                if k > 0:
                    tag_n1 = tags[k-1]
                    word_n1 = words[k-1]
                else:
                    tag_n1 = 'S'
                    word_n1 = '-'
                    
                #f2: (tag, -1)
                if ('f2', tag_n1) in self.feas:
                    fea.append(self.feas[('f2', tag_n1)])
                
                #f3: (word, -1)
                if ('f3', word_n1) in self.feas:
                    fea.append(self.feas[('f3', word_n1)])

                if k+1 == L:
                    word_p1 = '-'
                else:
                    word_p1 = words[k+1]
                    
                #f4: (word, 1)
                if ('f4', word_p1) in self.feas:
                    fea.append(self.feas[('f4', word_p1)])
                
                #f5: (word, -1, word, 0)
                k = ('f5', w_0, word_n1)
                if k in self.feas:
                    fea.append(self.feas[k])
                    
                self.feas_vec.append((tag, fea))
                
    def load_train_data(self, data_path, line_limit=5000):
        with open(data_path, 'r') as fin:
            lines = [line for line in fin]
            self.data = []
            for line in lines:
                if line_limit < 0:
                    break
                else:
                    line_limit -= 1
                if not line.strip():
                    continue
                terms = [t.rsplit('/', 1) for t in line.strip().split('\t')]
                self.data.append(terms)
            print '从语料中总共读取 %i行结果' % (len(self.data))
    
    def train(self, step=5, max_iter=2000, gamma=0.01):
        self.fea_num = len(self.feas)
        self.w = [0.0] * (4 * self.fea_num)  #每个特征对应的权值
        self.data_num = len(self.feas_vec)
        tag_idx = {'S':0, 'B':1, 'I':2, 'E':3}
        for it in range(max_iter):
            self.dw = [0.0] * (4 * self.fea_num)
            cost = 0.0
            for vec in self.feas_vec:
                (tag, fea) = vec
                #S:0, B:1, I:2, E:3
                Phi_s = math.exp(sum([self.w[f] for f in fea]))
                Phi_b = math.exp(sum([self.w[f+1] for f in fea]))
                Phi_i = math.exp(sum([self.w[f+2] for f in fea]))
                Phi_e = math.exp(sum([self.w[f+3] for f in fea]))
                Phi_sum = Phi_s + Phi_b + Phi_i + Phi_e
                p_s = Phi_s / Phi_sum
                p_b = Phi_b / Phi_sum
                p_i = Phi_i / Phi_sum
                p_e = Phi_e / Phi_sum
                tag_p = [0.0] * 4
                tag_p[tag_idx[tag]] = 1
                Phi = [p_s, p_b, p_i, p_e]
                cost += math.log(Phi[tag_idx[tag]])
                for f in fea:
                    self.dw[f] += -p_s + tag_p[0] - gamma * self.w[f]
                    self.dw[f + 1] += -p_b + tag_p[1] - gamma * self.w[f + 1]
                    self.dw[f + 2] += -p_i + tag_p[2] - gamma * self.w[f + 2]
                    self.dw[f + 3] += -p_e + tag_p[3] - gamma * self.w[f + 3]
            for k in range(4 * self.fea_num):
                self.w[k] = self.w[k] + step * self.dw[k]/float(self.data_num)
            
            if it % (max_iter / 100) == 0:
                print it, cost
    
    def save_model(self, line_limit):
        print '开始保存特征文件和模型文件...'
        feas_file = './data/feas_idx.pkl.%i' % line_limit
        with open(feas_file, 'w') as f:                     # open file with write-mode
            pickle.dump(self.feas, f)
            
        feas_wei = './data/feas_wei.pkl.%i' % line_limit
        with open(feas_wei, 'w') as f:                     # open file with write-mode
            pickle.dump(self.w, f)
        print '保存特征文件和模型文件结束'
        
    def load_model(self, line_limit):
        print '开始加载特征文件和模型文件...'
        feas_file = './data/feas_idx.pkl.%i' % line_limit
        with open(feas_file, 'r') as f:                     # open file with write-mode
            self.feas = pickle.load(f)
            
        feas_wei = './data/feas_wei.pkl.%i' % line_limit
        with open(feas_wei, 'r') as f:                     # open file with write-mode
            self.w = pickle.load(f)
        print '结束加载特征文件和模型文件'
        
    def cut_demo(self):
        with open('./data/test.txt', 'r') as fin:
            for line in fin:
                words = line.strip().decode('utf8')
                tags = []
                self.feas_vec = []
                predict_tag = 'S'
                L = len(words)
                for k in range(L):
                    fea = []
                    w_0 = words[k].encode('utf8')

                    #f1: (word, 0)
                    if ('f1', w_0) in self.feas:
                        fea.append(self.feas[('f1', w_0)])
                    
                    if k > 0:
                        tag_n1 = predict_tag
                        word_n1 = words[k-1].encode('utf8')
                    else:
                        tag_n1 = 'S'
                        word_n1 = '-'
                        
                    #f2: (tag, -1)
                    if ('f2', tag_n1) in self.feas:
                        fea.append(self.feas[('f2', tag_n1)])
                    
                    #f3: (word, -1)
                    if ('f3', word_n1) in self.feas:
                        fea.append(self.feas[('f3', word_n1)])
    
                    if k+1 == L:
                        word_p1 = '-'
                    else:
                        word_p1 = words[k+1].encode('utf8')
                        
                    #f4: (word, 1)
                    if ('f4', word_p1) in self.feas:
                        fea.append(self.feas[('f4', word_p1)])
                    
                    #f5: (word, -1, word, 0)
                    k = ('f5', w_0, word_n1)
                    if k in self.feas:
                        fea.append(self.feas[k])
                        
                    predict_tag = self.predict(fea)
                    tags.append(predict_tag)
                    
                print self.parse_tag(words, tags)
    
    def predict(self, fea):
        Phi_s = math.exp(sum([self.w[f] for f in fea]))
        Phi_b = math.exp(sum([self.w[f+1] for f in fea]))
        Phi_i = math.exp(sum([self.w[f+2] for f in fea]))
        Phi_e = math.exp(sum([self.w[f+3] for f in fea]))
        Phi_max = max([Phi_s, Phi_b, Phi_i, Phi_e])
        if Phi_max == Phi_s:
            return 'S'
        elif Phi_max == Phi_b:
            return 'B'
        elif Phi_max == Phi_i:
            return 'I'
        elif Phi_max == Phi_e:
            return 'E'
        else:
            return 'S'
        
    def parse_tag(self, words, tags):
        buff = ''
        for (word, tag) in zip(words, tags):
            if tag == 'S' or tag == 'E':
                buff += '%s/ ' % word
            else:
                buff += word
        return buff.encode('utf8')