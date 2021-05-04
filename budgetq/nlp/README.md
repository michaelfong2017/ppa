# Notes for budgetq analysis
## conda installation
For topic modeling only:
```
conda create -n budgetq_env python=3.7.10 &&
conda activate budgetq_env &&
conda install -c bioconda mysqlclient -y &&
conda install -c conda-forge jieba3k -y &&
pip install --upgrade gensim
pip install gensim==3.4.0
pip install smart_open==1.9.0
conda install -c conda-forge openpyxl
```

Hyper-parameter tuning:
num_of_topics: 22,23,24,25,26,27,28,29,30
25 best

random_state: 10,100,1000,10000,100000
1000 best

passes: 1,2,...
2 best

bigram/trigram/none
none best trigram worst

alpha: 50/[50,100,200,300,400,500,600,1000]

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

