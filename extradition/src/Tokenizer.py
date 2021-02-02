#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 23:01:35 2020

@author: michael
"""

#%%
import pandas as pd
#%%
from spyder_kernels.utils.iofuncs import load_dictionary
splitted_msgs = load_dictionary('data/splitted_msgs.spydata')[0]['splitted_msgs']

#%%
import pycantonese as pc

#%%
N = 100000
tokens = []
token_lengths = []
import os.path
if not os.path.isfile('pycantonese_10w.txt'):
    with open('pycantonese_10w.txt', 'w') as f: 
        for i in range(N):
            f.write("[tokenized_msg " + str(i+1) + "]\n\n")
            token = []
            for j in range(len(splitted_msgs[i])):
                t = pc.segment(splitted_msgs[i][j])
                token_lengths.append(len(t))
                f.write("\"")
                f.write('|'.join((filter(lambda x: x not in ['\n', ' ', '\t'], t))))
                f.write("\"\n")
        
                token.append(t)

            f.write("\n\n")
            tokens.append(token)
else:
    for i in range(N):
        for j in range(len(splitted_msgs[i])):
            t = pc.segment(splitted_msgs[i][j])
            token_lengths.append(len(t))
        
#%%
from spyder_kernels.utils.iofuncs import load_dictionary
tokens = load_dictionary('data/tokens.spydata')[0]['tokens']
token_lengths = load_dictionary('data/token_lengths.spydata')[0]['token_lengths']
    
#%%
for token in tokens:
    print(token)