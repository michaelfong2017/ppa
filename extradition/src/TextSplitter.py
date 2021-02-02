# -*- coding: utf-8 -*-

#%%
import pandas as pd
#%%
df2 = pd.read_excel('data/df_10w_filtered.xlsx', sheet_name='Sheet1', na_filter=False)

print("Column headings:")
print(df2.columns)

filtered_msgs = df2['filtered_msg']

#%%
from spyder_kernels.utils.iofuncs import load_dictionary
filtered_msgs = load_dictionary('data/filtered_msgs.spydata')[0]['filtered_msgs']

#%%
def split_text(text):
    return list(filter(lambda c : c != "" and c != " ", ""
                       .join((char if char.isalpha() or char.isdigit() or char==" " else "\n") for char in text)
                       .split("\n")))

#%%
N=100000
splitted_msgs = []
max=0
for i in range(N):
    splitted_msgs.append(split_text(filtered_msgs[i]))
    if len(splitted_msgs[i]) > max:
        max = len(splitted_msgs[i])
        
#%%
df_splitted_msgs = pd.DataFrame(splitted_msgs)
    
#%%
import csv

with open("data/df_10w_splitted.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(splitted_msgs)
