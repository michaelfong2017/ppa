#!/Users/michael/opt/anaconda3/envs/extradition_env/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 14:12:39 2020

@author: michael
"""

#%%
import pandas as pd
#%%
df = pd.read_excel('data/dataset_10w_jieba_msg_title.xlsx', sheet_name='Sheet1')

print("Column headings:")
print(df.columns)

#%%
pd.set_option('display.max_colwidth', None)
import re

#%%
N=100000
### Store new messages
all_new_msgs = []
### Store imgs
all_imgs = []
### Store links
all_links = []
### Count </blockquote>
blockquote_counts = []
### Count </span>, </strong>, <div style=...>, </ins>, </em>, </pre>, </code>, </del>
style_counts = []
for i in range(N):
    blockquote_count = 0
    style_count = 0
    
    
    msg = df["item_data_msg"][i]
    
    ### Delete <br />
    new_msg = re.sub(r'<br />', r'', msg)
    
    ### Extract imgs
    imgs = re.findall(r'<img\s.*?src=\"(.*?)[\"\s\n]', new_msg)
    all_img = '\n'.join(imgs)
    all_imgs.append(all_img)
    
    ### Delete img
    new_msg = re.sub(r'<img\s(.*?)/>', r'', new_msg)
    
    
    ### Extract links
    links = re.findall(r'(?<=<a\shref=\")(.*?)[\"\s\n]', new_msg)
    ### Delete links
    new_msg = re.sub(r'<a\s(.|\n)*?</a>', r'', new_msg)
    
    ### Extract links 2 (further extract after deleting)
    links2 = re.findall(r'(?:http|https)[a-zA-Z0-9\.\/\?\:@\-_\=#%\&]*', new_msg)
    ### Delete links2
    
    new_msg = re.sub(r'(?:http|https)[a-zA-Z0-9\.\/\?\:@\-_\=#%\&]*', r'', new_msg)
    ### Extract links 3 (further extract after deleting)
    links3 = re.findall(r'(?:t.me\/)[a-zA-Z0-9\.\/\?\:@\-_\=#%\&]*', new_msg)
    all_link = '\n'.join(links + links2 + links3)
    all_links.append(all_link)
    ### Delete links3
    new_msg = re.sub(r'(?:t.me\/)[a-zA-Z0-9\.\/\?\:@\-_\=#%\&]*', r'', new_msg)
    
        
    
    ### Delete <blockquote>, </blockquote>, <span ...>, </span>, <strong>, </strong>,
    ### <div style=...>, <ins>, </ins>, <em>, </em>, <pre>, </pre>,
    ### <code...>, </code>, <del>, </del> 
    ### Count </span>, </strong>, <div style=...>, </ins>, </em>, </pre>, </code>, </del>
    subn1 = re.subn(r'<blockquote>', r'', new_msg)
    new_msg = subn1[0]
    subn2 = re.subn(r'</blockquote>', r'', new_msg)
    new_msg = subn2[0]
    blockquote_count = blockquote_count + subn2[1]
    
    
    subn3 = re.subn(r'<span\sstyle=(.*?)>', r'', new_msg)
    new_msg = subn3[0]
    subn4 = re.subn(r'</span>', r'', new_msg)
    new_msg = subn4[0]
    style_count = style_count + subn4[1]
    
    subn5 = re.subn(r'<strong>', r'', new_msg)
    new_msg = subn5[0]
    subn6 = re.subn(r'</strong>', r'', new_msg)
    new_msg = subn6[0]
    style_count = style_count + subn6[1]
    
    subn7 = re.subn(r'<div\sstyle=(.*?)>', r'', new_msg)
    new_msg = subn7[0]
    subn8 = re.subn(r'</div>', r'', new_msg)
    new_msg = subn8[0]
    style_count = style_count + subn8[1]
    
    subn9 = re.subn(r'<ins>', r'', new_msg)
    new_msg = subn9[0]
    subn10 = re.subn(r'</ins>', r'', new_msg)
    new_msg = subn10[0]
    style_count = style_count + subn10[1]
    
    subn11 = re.subn(r'<em>', r'', new_msg)
    new_msg = subn11[0]
    subn12 = re.subn(r'</em>', r'', new_msg)
    new_msg = subn12[0]
    style_count = style_count + subn12[1]
    
    subn13 = re.subn(r'<pre>', r'', new_msg)
    new_msg = subn13[0]
    subn14 = re.subn(r'</pre>', r'', new_msg)
    new_msg = subn14[0]
    style_count = style_count + subn14[1]
    
    subn15 = re.subn(r'<code(.*?)>', r'', new_msg)
    new_msg = subn15[0]
    subn16 = re.subn(r'</code>', r'', new_msg)
    new_msg = subn16[0]
    style_count = style_count + subn16[1]
    
    subn17 = re.subn(r'<del>', r'', new_msg)
    new_msg = subn17[0]
    subn18 = re.subn(r'</del>', r'', new_msg)
    new_msg = subn18[0]
    style_count = style_count + subn18[1]
    
    
    
    style_counts.append(style_count)
    blockquote_counts.append(blockquote_count)
    
    # print(new_msg)
    # print('\n')
    all_new_msgs.append(new_msg)
    
# for i in range(N):
    # print("Row index "+str(i)+" (starts from 0):")
    # print(all_imgs[i])
    # print(all_links[i])
    # print('\n')


#%%
df2 = df.copy()
df2['filtered_msg'] = all_new_msgs
df2['msg_img'] = all_imgs
df2['msg_link'] = all_links
df2['blockquote_count'] = blockquote_counts
df2['style_count'] = style_counts

options = {}
options['strings_to_formulas'] = False
options['strings_to_urls'] = False
with pd.ExcelWriter('data/df_10w_filtered.xlsx', engine='xlsxwriter',options=options) as writer:  
    df2.to_excel(writer)
    
    
    
    
    