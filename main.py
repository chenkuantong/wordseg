#encoding=utf-8
'''
Created on Jan 9, 2015

@author: ckt
'''
import getopt 
import sys


from smartseg import segmodel

if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], 'hpt:s:l:')
    
    help_info = """usage:
        -h: 帮助
        -p: 用开源的分词工具，分割语料作为模型训练数据,生成train.txt
        -t: N, 训练分词模型,　N为选取前train.txt中N行数据进行训练
        -s: N, 在测试集上进行分词测试  ./data/test.txt, N为采用前train.txt中N行数据进行训练得到的模型
    """
    
    if not opts:
        print help_info
        
    for option, value in opts:
        if option in ['-h', '--help']:
            print help_info
        
        if option in ['-p']:
            from pretrain import jiebaseg
            cutter = jiebaseg.JiebaSeg('./data/亵渎.txt') #读取语料　./data/亵渎.txt
            cutter.cut_data_and_write('./data/train.txt')
        
        if option in ['-t']:
            cutter = segmodel.SmartSegModel()
            N = int(value)
            cutter.load_train_data('./data/train.txt', line_limit=N)
            cutter.feature_extract()
            cutter.train()
            cutter.save_model(N)
        
        if option in ['-s']:
            N = int(value)
            cutter = segmodel.SmartSegModel()
            cutter.load_model(N)
            cutter.cut_demo()
        