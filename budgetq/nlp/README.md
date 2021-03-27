# Notes for budgetq analysis
## conda installation
For topic modeling only:
```
conda create -n budgetq_env python=3.7.10 &&
conda activate budgetq_env &&
conda install -c bioconda mysqlclient -y &&
conda install -c conda-forge jieba3k -y &&
pip install --upgrade gensim
```