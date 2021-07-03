# Notes for budgetq analysis
## conda installation
For topic modeling only:
```
conda create -n budgetq_env python=3.7.10 &&
conda activate budgetq_env &&
pip install psycopg2-binary &&
conda install -c conda-forge jieba3k -y &&
pip install --upgrade gensim
pip install gensim==3.4.0
pip install smart_open==1.9.0
conda install -c conda-forge openpyxl
conda install -c conda-forge pandas
```

```
conda install -c conda-forge jupyter -y
ipython kernel install --user --name budgetq --display-name budgetq
pip install black
```

Hyper-parameter tuning:
num_of_topics: 22,23,24,25,26,27,28,29,30
25 is the best

random_state: 10,100,1000,10000,100000
1000 is the best

passes: 1,2,...
2 is the best

bigram/trigram/none
none is the best; trigram is the worst

alpha: 50/[50,100,200,300,400,500,600,1000]
alpha = 50/200 is the best

filter_no_above: [0.1,0.05,0.02,0.01,0.009,0.008,0.007,0.006,0.005,0.004,0.003,
0.002,0.001,0.0009,0.0008,0.0007,0.0006,0.0005]
Coherence score keeps increasing from 0.435 to 0.717 as filter_no_above decreases, except that 0.05 and 0.02 are the worst

num_of_topics 20 (-29.8, 0.460) better than 40 (0.405). 20-25 is okay.
alpha, eta auto (-28.5, 0.463) better than np.empty(20).fill(0.025) (-29.8, 0.460).
no bigram (-27.9, 0.489) better than has bigram (-28.5, 0.463).

num_of_topics 25 is best
alpha 0.25 is best since perplexity -48 is better than -59

alpha: 0.125
Perplexity: -59.09508700332216
Coherence Score: 0.5617456594020727
save_filename: topics_t17.xlsx

alpha: 0.25
Perplexity: -48.88279510049568
Coherence Score: 0.564484125681871
save_filename: topics_t15.xlsx

## POS
I'll use POS Tagging to remove unnecessary words

import jieba.posseg as pseg
def keep_words(s, to_keep=['ns', 'v', 'n', 'a']):
    s_ = []
    pos_list = pseg.lcut(s)
    for item in pos_list:
        if item.flag in to_keep:
            s_.append(item.word)
    return s_
    keep_words('这苹果，快吃完了居然没有一个好吃的，并且都小的很，和图片严重不同！对京东太失望了，下次还是去...')
