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

num_of_topics 20 (-29.8, 0.460) better than 40 (0.405). 20-25 is okay.
alpha, eta auto (-28.5, 0.463) better than np.empty(20).fill(0.025) (-29.8, 0.460).
no bigram (-27.9, 0.489) better than has bigram (-28.5, 0.463).

