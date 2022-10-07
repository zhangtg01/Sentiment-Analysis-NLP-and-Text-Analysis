#!/usr/bin/env python
# coding: utf-8

# ### BAIT 508 Individual Project: SEC Filings Text Analytics

# Objectives: Use NLP and Python skills (`pandas`, `BeautifulSoup`, `nltk`, `wordcloud`, user-defined functions, ...) to analyze text data of SEC

# ### Import the appropriate libraries you need to solve the questions.

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import wordcloud as wc
import requests
from bs4 import BeautifulSoup
import spacy
import warnings
warnings.filterwarnings('ignore')
from textblob import TextBlob


# ### Please assign the variables `first_name`, `last_name`, `student_id`, and `email` with your first name, last name, student ID, and email address.

# In[3]:


first_name = str("Taige")
last_name = str("Zhang")

# ## Download and preprocess the data
# 
# 
# 
# - Download `corpus_10k_2015-2019.csv` file into the same directory, where `hw2_starter.ipynb` is located. (If not, there will be an extra deduction on your grade)
# - First, create user-defined `isYear` function with two parameters (`target_year`, `text`) which check the `year`column value is the same as `target_year` in the `text`.
# - Second, open `corpus_10k_2015-2019.csv` file with `open` function and filter the data which the `year` is `2019` using `isYear` function you defined.
# - Save the filtered data as a `txt` file called `corpus.subset.txt`.
# - Read the txt file you made as a pandas dataframe `df`.
# - Drop <b>the columns</b> where <b>all elements are NaN</b> (in other words, this column contains no useful information)
#  using `dropna` method from `pandas`.
# 
# [Conditions]
# - drop `Nan` value : Yes
# - fill missing value with empty string : Yes
# 
# In[45]:


# ans1 
def isYear(target_year,text):
    if text[4] == str(target_year):
        return True
    else:
        return False


# In[13]:



with open('corpus_10k_2015-2019.csv','r') as f:
    line = f.readlines()        #read from line to line, line is a list containing lines with str format
    with open('corpus.subset.txt', 'w') as sub:
        for l in line:          #l is each line
            words=l.split(',')
            if words[4] == 'year' or isYear(2019,words):
                sub.writelines(l)


# In[4]:


#read the csv file
df_org = pd.read_csv('corpus.subset.txt')
df_org.shape


# In[5]:


#dropna from df
df = df_org.dropna(axis=1,how='all')
df.shape


# In[6]:


# fillna with ''
df = df.fillna('')
ans1 = df.shape
print(ans1)


# ##  Scrape SIC code and names on the web using BeautifulSoup 
# 
# ### Question 2: How many `header cell`, `row`, `cell`, and `caption` are in the SIC code table from <b>"List"</b> section of the Wikipedia page ? Please sum up the count of `header cell`, `row`, `cell`, and `caption`, then assign it to `ans2` variable.
# 
# - Collect the industry names for sic codes from the <b>"List"</b> section of the Wikipedia page ("https://en.wikipedia.org/wiki/Standard_Industrial_Classification").
# - If the above link is not directly linked with the Wikipedia page, please copy and paste the URL on the new tab.
# - Create `code_to_industry_name` dictionary where the `key` is the sic code and the `value` is the industry name.
# - Then, replace the SIC code "0100 (01111...)" from the table with 0100.
# 
# 
# [Hint: HTML Table Tags]
# - `<th>`: defines a header cell in a table
# - `<tr>`: defines a row in a table
# - `<td>`: defines a cell in a table
# - `<caption>`: defines a table caption

# In[7]:


# ans2
# https://blog.csdn.net/u010916338/article/details/105493393
# lecture notes

# Specify url: url
url = 'https://en.wikipedia.org/wiki/Standard_Industrial_Classification'

# Package the request, send the request and catch the response: r
r = requests.get(url)

# Extracts the response as html: html_doc
html_doc = r.text

# Create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc, "lxml")

#find the second table in soup
table_list = soup.find_all('table')[1]
table_list


# In[8]:


trs = table_list.find_all('tr')
num_tr = len(trs)
# num_tr
ths = table_list.find_all('th')
num_th = len(ths)
# num_th
caps = table_list.find_all('caption')
num_cap = len(caps)
num_cap


# In[9]:


code_to_industry_name = dict()
num_td = 0
for tr in trs:                             #每一行
    try:
        tds = tr.find_all('td')            #每一个cell
        k = tds[0].get_text().strip()      #编号
        v = tds[1].get_text().strip()      #名称
        code_to_industry_name[k] = v
        num_td += 2                        #每个循环里有两个td
    #print()
    except:
        continue

code_to_industry_name['0100'] = code_to_industry_name.pop('0100 (01111...)')
code_to_industry_name


# In[10]:


num_td


# In[11]:


ans2 = num_cap+num_td+num_tr+num_th
ans2


# - Add a new column `industry_name` to `df` using `lambda` function.
# - Values in `industry_name` must correspond to the `sic` in the `df`.
# - For example, if a row has a SIC code of `1000`, then value of its industry name will be `Forestry`.
# 
# [Hint]
# - `lamda` : https://www.w3schools.com/python/python_lambda.asp

# In[12]:


# checked the len of sic and there are some 3 digits
df_len_sic = df['sic'].apply(lambda x:len(str(x)))
3 in df_len_sic


# In[13]:


# https://stackoverflow.com/questions/339007/how-to-pad-zeroes-to-a-string
#fill df['sic'] to 4digits
df['industry_name'] = df['sic'].apply(lambda x : code_to_industry_name[str(x).zfill(4)])


# ## Now, you get the preprocess the dataframe `df` to analyze.
# 
# ## Industry analysis (Q3-Q5) : use the dataframe `df`
# 
# ### Question3. What are the 5 most common industries? Get them from `industry_name`, not from `sic` code
# - Store a `list` of 5 most common industry names in the `ans3`.
# - Sort the `ans3` in the descending order.

# In[14]:


# ans3
top_5_ind = df['industry_name'].value_counts()[:5]
# type(top_5_ind)
top_5_ind = top_5_ind.sort_values()
ans3 = list(top_5_ind.sort_index(ascending=False).index)
# ans3 = list(top_5_ind.index)
ans3


# ### Question4. Count the number of rows, which `industry_name` value starts with `Wholesale` and ends with `Supplies` . Please assign the number to the `ans4` variable. 
# 
# - Here, first character (`W` and `S`) is uppercase.
# - `ans4` data type is `int`.
# 
# Hint:
# - string.startswith: https://www.w3schools.com/python/ref_string_startswith.asp
# - string.endswith: https://www.w3schools.com/python/ref_string_endswith.asp

# In[15]:


# ans4
df_WS = df[df['industry_name'].apply(lambda x:x.startswith('Wholesale') & x.endswith('Supplies'))]
ans4 = len(df_WS)
ans4


# ### Question5. What is the `name` of the company `id` with `1353611-2019`?
# - Store the company name as a `string` in the `ans5`.

# In[16]:


# ans5
comp5_name = df[df['id']=='1353611-2019'].name    #a series
ans5 = comp5_name[2]
ans5


# ## Keyword analysis (Q6 and Q7)
# 
# ## (Q6-Q7) : use the dataframe `df`
# ### For Q6 and Q7 you will filter out stopwords and non-alphanumeric English characters. 
# - You can use `nltk.corpus.stopwords` for our definition of stopwords. 
# - Alphanumeric English characters are letters in the alphabet (a-z, A-Z) and numbers (0-9).
# - For example, <b>"Python is awesome👍 great7777 !!! "</b> would be filtered to <b>"Python awesome great"</b> after removing stopwords (in this case "is"), exclamation mark, numbers, and the emoji (non-alphanumeric).
# - Please use `nltk.tokenize` package to solve Q6 and Q7. 
# 
# 
# [Hint]
# - `nltk.corpus` for stopwords : https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
# - `nltk.tokenize` package : https://www.nltk.org/api/nltk.tokenize.html

# In[17]:


import nltk
from nltk.corpus import stopwords
from collections import Counter


# ### Question6. What are the 5 most common words from the `Item_5` column?
# - Store a `list` of the 5 most common words in `ans6`.
# - Sort the `ans6` in the ascending order.
# 
# [Conditions]
# - Need lowercase : `Yes`
# - Filter stopwords : `No`
# - Filter non-Alphanumeric English character : `Yes`
# - Filter single length character : `No`

# In[18]:


# columns = [col for col in df.columns if 'item' in col]


# In[19]:


# ans6
stopwords = nltk.corpus.stopwords.words('english')


# In[20]:


# https://stackoverflow.com/questions/1653425/a-za-z-a-za-z0-9-regular-expression/1653559
import re
new_word=[]

for item in df['item_5']:
    word = re.sub(r"[^a-zA-Z0-9\s]","",item.lower())
    words = nltk.word_tokenize(word)
    for w in words:
        new_word.append(w)

c = Counter(new_word)
mc = c.most_common(5)
#generate the list of common words
ls_mc = sorted(list(dict(mc).keys()),reverse=False)
ans6 = ls_mc
ans6


# ### Question7. What are the 5 most common words from the `Item_5` column without stopwords?
# - Store a `list` of the 5 most common words in `ans7`.
# - Sort the `ans7` in the descending order.
# 
# [Conditions]
# - Need lowercase : `Yes`
# - Filter stopwords : `Yes`
# - Filter non-Alphanumeric English character : `Yes`
# - Filter single length character : `No`

# In[21]:


# ans7
#make a list to store all of the eligible words
new_word2=[]

for item in df['item_5']:
    word = re.sub(r"[^a-zA-Z0-9\s]","",item.lower())
    words = nltk.word_tokenize(word)
    for w in words:
        if w not in stopwords:
            new_word2.append(w)

c = Counter(new_word2)
mc = c.most_common(5)
#generate the list of common words
ls_mc = sorted(list(dict(mc).keys()),reverse=True)
ans7 = ls_mc
ans7


# ## Named Entity Recognition (Q8-Q11)
# ## (Q8-Q11) : use the dataframe `df_50`
# 
# - To reduce the processing time, please select first 50 rows from the dataframe `df` using `head(50)` option and save the dataframe as `df_50`. 
# - `df_50` dataframe will be used for Q8-Q11.
# - If any of the entities are spaces, exclude them when considering the most common.
#     - `(" ")` is not a valid entity. To remove the entities that are spaces, please use `split`, `strip` and `join` method.
# - Suppose that you have the following sentence, which is <b>'&nbsp;&nbsp;I&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;love&nbsp;&nbsp;&nbsp;&nbsp;python&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'</b>.
# - In this case, you need to make it as <b>'I love python'</b>.

# In[22]:


# https://www.py.cn/jishu/jichu/23561.html
df_50 = df.head(50)

def strip_space(x):
    a = str(x).strip()
    a = a.split()
    str_out=(' ').join(a)
    return str_out

df_50 = df_50.applymap(strip_space)
df_50


# ### Question8. What are the 5 most common `PERSON` named entities overall from the `item_1` column?
# - Store a `list` of the 5 most common `PERSON` named entities in `ans8`.
# - Sort the `ans8` in the descending order.
# 
# 
# [Conditions]
# - Need lowercase : `No`
# - Filter stopwords : `No`
# - Filter non-Alphanumeric English character : `No`
# - Filter single length character : `No`

# In[23]:


# !python3 -m spacy download en
# !python3 -m spacy download en_core_web_sm


# In[24]:


nlp = spacy.load('en_core_web_sm', exclude=["tagger","parser","matcher"])
print(nlp)


# In[25]:


# create an empty defaultdict
word_ls8=[]

# Print all of the found entities and their labels
for i in df_50['item_1']:
    doc = nlp(i)
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            word_ls8.append(ent.text)
            
c = Counter(word_ls8)
mc = c.most_common(5)
#     ner_categories[ent.label_] += 1
    #print(ent.label_, ent.text)


# In[26]:


#generate the list of common words
ls_mc = sorted(list(dict(mc).keys()),reverse=True)
ans8 = ls_mc
ans8


# ### Question9. How many `ORG` named entities only have 1 occurrence case in `item_8` column?
# - Store the number of cases as a `integer` in `ans9`.
# - `ORG` includes companies, agencies, and institutions.
# 
# [Conditions]
# - Need lowercase : `No`
# - Filter stopwords : `No`
# - Filter non-Alphanumeric English character : `No`
# - Filter single length character : `No`

# In[27]:


# ans9
# create an empty defaultdict
word_ls9=[]

# Print all of the found entities and their labels
for i in df_50['item_8']:
    doc = nlp(i)
    for ent in doc.ents:
        if ent.label_ == 'ORG':
            word_ls9.append(ent.text)
            
list_9 = Counter(word_ls9)
dic9 = dict(list_9)


# In[28]:


b = Counter(dic9.values())
ans9 = b[1]
ans9


# ### Question10. What are the 4 most common named entities overall from the `item_9` column?
# - Store a `list` of the 4 most common named entities in `ans10`.
# - Sort the `ans10` in the descending order.
# 
# [Conditions]
# - Need lowercase : `No`
# - Filter stopwords : `No`
# - Filter non-Alphanumeric English character : `No`
# - Filter single length character : `No`

# In[29]:


# ans10
word_ls10=[]

# Print all of the found entities and their labels
for i in df_50['item_9']:
    doc = nlp(i)
    for ent in doc.ents:
        word_ls10.append(ent.text)
            
c = Counter(word_ls10)
b = c.most_common(4)

ans10 = list(dict(b).keys())
ans10 = sorted(ans10,reverse=True)
ans10


# ### Question11. Count the number of companies whose `item_1` contains the most common named entity in all rows of `item_1`.
# - Filter out companies whose `item_1` contain the result you get above.
# - Count the numbers of distinct companies and store your answer in `ans11`.
# - You will use `df_50` for all evaluation
# 
# [Conditions]
# - Need lowercase : `No`
# - Filter stopwords : `No`
# - Filter non-Alphanumeric English character : `No`
# - Filter single length character : `No`

# In[30]:


# ans11
word_ls11=[]

# Print all of the found entities and their labels
for i in df_50['item_1']:
    doc = nlp(i)
    for ent in doc.ents:
        word_ls11.append(ent.text)
            
c = Counter(word_ls11)
most_common = c.most_common(1)
most_common
# ans10 = list(dict(b).keys())
# ans10 = sorted(ans10,reverse=True)
# ans10


# In[31]:


count = 0
for n in df_50['item_1']:
    if most_common[0][0] in str(n):
        count+=1
ans11 = count
print(ans11)


# ## NER for specific firm (Q12-Q13)
# ## (Q12-Q13) : use the dataframe `df`
# - You want to find the information on the company with id `1653710-2019`.
# - Given list comprehension, you want to find out common entities in the dataframe `df`.
# - If any of the entities are spaces, exclude them when considering the most common.
# - `(" ")` is not a valid entity. To remove the entities that are spaces, please use `split` and `join` method.
# - Suppose that you have the following sentence, which is <b>'&nbsp;&nbsp;I&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;love&nbsp;&nbsp;&nbsp;&nbsp;python&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'</b>.
# - In this case, you need to make it as <b>'I love python'</b>.

# ### Question12. what are the 4 most common `PERSON` named entities mentioned by the company with id `1653710-2019` across all rows with fixed prefix `item_`?
# - Store a `list` of the 4 most common `PERSON` named entities in `ans12`.
# - Sort the `ans12` in the descending order.
# 
# [Conditions]
# - Need lowercase : `No`
# - Filter stopwords : `No`
# - Filter non-Alphanumeric English character : `No`
# - Filter single length character : `No`

# In[35]:


# ans12

df_12 = df_50[df_50['id']=='1653710-2019']
df_12


# In[72]:


word_ls12=[]
for i in df_12.values:
    doc = nlp(str(i))
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            word_ls12.append(ent.text)
c = Counter(word_ls12)


# In[73]:


mc = dict(c.most_common(4)).keys()
ls_mc4 = sorted(list(mc),reverse=True)
ans12 = ls_mc4
ans12


# ### Question13. What are the 2 most common `GPE` named entities mentioned by the company with id `1653710-2019` across all with fixed prefix `item_`?
# - Store a `list` of the 2 most common `GPE` named entities in `ans13`.
# - `GPE` includes geopolitical entities such as countries, cities, and states.!
# - Sort the `ans13` in the ascending order.
# 
# [Conditions]
# - Need lowercase : `No`
# - Filter stopwords : `No`
# - Filter non-Alphanumeric English character : `No`
# - Filter single length character : `No`

# In[74]:


# ans13
word_ls13=[]
for i in df_12.values:
    doc = nlp(str(i))
    for ent in doc.ents:
        if ent.label_ == 'GPE':
            word_ls13.append(ent.text)
c = Counter(word_ls13)


# In[75]:


mc = dict(c.most_common(2)).keys()
ls_mc2 = sorted(list(mc))
ans13 = ls_mc2


# ## Twitter analysis (Q14-Q15)
# ### `tweets.json` collected  50,000 tweets containing below keywords: 
# - Keyword : `analytics`, `technology`, `big data`, `machine learning`, `artificial intelligence`
# - The way used to collect the Twitter streaming data is using `tweepy` and `twython` module.
# - `tweepy` for Twitter streaming : http://docs.tweepy.org/en/latest/streaming_how_to.html
# 
# ### Save and read the `tweets.json` file as `tweets` 
# - Download `tweets.json` file into the same directory, where `hw2_starter.ipynb` is located. (If not, there will be an extra deduction on your grade)
# - Open `tweets.json` file as `tweets` with `open` function.
# - please select first `10,000` tweets from `tweets`.
# 
# [Hint]
# - `open` function : https://www.w3schools.com/python/ref_func_open.asp

# In[76]:


import json

with open('tweets.json','r') as f:
    data = json.load(f)
    tw = data[:10000]


# In[77]:


len(tw)


# In[64]:


tw[0]['text']


# ### Question14. Find the firm that has the most common words between `item_1` and the 10,000 tweets. 
# ### Q14 : use the dataframe `df_50`
# 
# - First, find the 100 most common words of each firm's `item_1` column in `df_50` by using `nltk.tokenize` package.
# - Then, use the top the top 100 most common words of the 10,000 tweets after removing stop words by using `nltk.tokenize` package.
# - Next find the most common words between `item_1` and the 10,000 tweets. 
# - Disregard the word count, we are only interested in the number of unqiue words that appear in intersection of both common words.
# - Store the answer as a string in `ans14`.
# - Don't need to filter the words with length 1.
# - Need to filter stopwords and non-alphanumeric english character in each firm's `item_1` column.
# - When you filter the stopwords, you need to make words lowercase.
# - Store the firm name as a `string` in `ans14`.
# 
# [Conditions]
# - Need lowercase : `Yes`
# - Filter stopwords : `Yes`
# - Filter non-Alphanumeric English character : `Yes`
# - Filter single length character : `No`
# 
# [Hint]
# - `nltk.corpus` for stopwords : https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
# - `isalnum()`: https://www.w3schools.com/python/ref_string_isalnum.asp

# In[81]:



tw_text=[]
for word in tw:
    tw_text.append(word['text'])     #word['text']找到字典中键是text下的每一条推文
word_list14=[]
for tweet in tw_text:
    words = nltk.word_tokenize(tweet.lower())
    for word in words:
        if str(word) not in stopwords:
            word_list14.append(word)
word_list14 = list(filter(str.isalnum, word_list14))
c = Counter(word_list14)
top100_com_tw = c.most_common(100)
top100_com_tw


# In[79]:


list_100_tw = list(dict(top100_com_tw).keys())
list_100_tw


# In[35]:


stopwords = nltk.corpus.stopwords.words('english')


# In[36]:


# ans14
words14=[]
for w in df_50['item_1']:
    words_temp = []
    words = nltk.word_tokenize(w.lower())
    for word in words:
        if str(word) not in stopwords:
            words_temp.append(word)
    words_temp = list(filter(str.isalnum,words_temp))
    c = Counter(words_temp)
    most_com_100 = c.most_common(100)
    words14.append(list(dict(most_com_100).keys()))
words14


# In[38]:


# https://www.geeksforgeeks.org/python-intersection-two-lists/
# https://stackoverflow.com/questions/30902558/finding-length-of-the-longest-list-in-an-irregular-list-of-lists
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3
# def find_max_list(list):
#     list_len = [len(i) for i in list]
#     a = max(list_len).index()
#     return list[a]
    
    
long_list = [] 
for i in words14:
    long_list.append(intersection(i,list_100_tw))
index_of_max = long_list.index(max(long_list, key=len))
ans14 = df_50.iloc[index_of_max]['name']
ans14


# ### Question15. In the selected 10,000 tweets, what are the 5 most common named entities mentioned?
# - You need to use the NER for this question.
# - If any of the entities are spaces, exclude them when considering the most common.
#     - `(" ")` is not a valid entity. To remove the entities that are spaces, please use `split`, `strip` and `join` method.
# - Suppose that you have the following sentence, which is <b>'&nbsp;&nbsp;I&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;love&nbsp;&nbsp;&nbsp;&nbsp;python&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'</b>.
# - In this case, you need to make it as <b>'I love python'</b>.
# - Store a list of the 5 most common named entities in `ans15`.
# - Sort the `ans15` in the ascending order.
# 
# [Conditions]
# - Need lowercase : `No`
# - Filter stopwords : `No`
# - Filter non-Alphanumeric English character : `No`
# - Filter single length character : `No`

# In[82]:


# ans15
def strip_space(x):
    a = str(x).strip()
    a = a.split()
    str_out=(' ').join(a)
    return str_out

# defined strip_space function in the previous question

# word_list15 = strip_space(word_list14)

#code from nlp_part4_named entity recognition updated-inclass
list_15 = []

for item in tw_text:
    doc = nlp(strip_space(item))
    for ent in doc.ents:
        list_15.append(ent.text)
            
c = Counter(list_15)

mc = dict(c.most_common(5)).keys()
ls_mc5 = sorted(list(mc),reverse=False)
ans15 = ls_mc5
ans15


# ## Word cloud and sentiment analysis (Q16-Q19)
# 
# ## (Q16-Q19) : use the dataframe `df`
# 
# - Use `wordcloud` library and `WordCloud` function in it.
# - Define user-defined `generate_wordcloud` function with one parameter `values` to generate word cloud for one input value.
# - You don't need `axis` in the wordcloud and use `bilinear` interpolation. 
# 
# [Hint]
# - `bilinear` for `imshow()` : https://matplotlib.org/3.3.1/gallery/images_contours_and_fields/interpolation_methods.html

# In[36]:


# import necessary modules
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk

get_ipython().run_line_magic('matplotlib', 'inline')

def generate_wordcloud(values):
    wordcloud = WordCloud(width=800, height=400).generate(str(values))
    plt.figure(figsize=(20,10)) # set up figure size
    plt.imshow(wordcloud,interpolation='bilinear') # word cloud image show
    plt.axis("off") # turn off axis
#     plt.savefig('my_word_cloud.png') # save as PNG file
#     plt.savefig('my_word_cloud.pdf') # save as PDF file


# ## For the following analyses, find the top two most common industries names
# - Assign the most common industry name as `top_1` and the second most common industry name as `top_2`.

# In[33]:


c = Counter(list(df['industry_name']))

top_1 = c.most_common()[0][0]
top_2 = c.most_common()[1][0]
print(top_1,top_2)


# ### Question16. Make two separate wordclouds for `item_1` column.
# 
# - One for the most common industry and another one for the second most common industry.
# - Save the graph named "`hw2_ans16a_{student_id}.png`" and "`hw2_ans16b_{student_id}.png`".<br/>
#   (e.g.) <b>hw2_ans16a_37510930.png</b>, <b>hw2_ans16b_37510930.png</b>, respectively.

# In[37]:


# hw2_ans16a_37510930.png

generate_wordcloud(list(df[df['industry_name']==top_1]['item_1']))
plt.savefig('hw2_ans16a_26805440.png')


# In[71]:


# hw2_ans16b_37510930.png
generate_wordcloud(list(df[df['industry_name']==top_2]['item_1']))
plt.savefig('hw2_ans16b_26805440.png')


# ### Question17. Make two separate wordclouds for `item_1a`column.
# - One for the most common industry and another one for the second most common industry.
# - Save the graph named "`hw2_ans17a_{student_id}.png`" and "`hw2_ans17b_{student_id}.png`".<br/>
#   (e.g.) <b>hw2_ans17a_37510930.png</b>, <b>hw2_ans17b_37510930.png</b>, respectively.

# In[83]:


# hw2_ans17a_37510930.png
generate_wordcloud(list(df[df['industry_name']==top_1]['item_1a']))
plt.savefig('hw2_ans17a_26805440.png')


# In[265]:


# hw2_ans17b_37510930.png
generate_wordcloud(list(df[df['industry_name']==top_2]['item_1a']))
plt.savefig('hw2_ans17b_26805440.png')


# ### Question18. Make two separate wordclouds for `item_7` column
# - One for the most common industry and another one for the second most common industry.
# - Save the graph named "`hw2_ans18a_{student_id}.png`" and "`hw2_ans18b_{student_id}.png`".<br/>
#   (e.g.) <b>hw2_ans18a_37510930.png</b>, <b>hw2_ans18b_37510930.png</b>, respectively.

# In[266]:


generate_wordcloud(list(df[df['industry_name']==top_1]['item_7']))
plt.savefig('hw2_ans18a_26805440.png')


# In[267]:


generate_wordcloud(list(df[df['industry_name']==top_2]['item_7']))
plt.savefig('hw2_ans18b_26805440.png')

# ### Question19. Make two histograms of the polarity for `item_1a` column. 
# - One for the most common industry and another one for the second most common industry.
# - Save the graph named "`hw2_ans19a_{student_id}.png`" and "`hw2_ans19b_{student_id}.png`".<br/>
#   (e.g.) <b>hw2_ans19a_37510930.png</b>, <b>hw2_ans19b_37510930.png</b>, respectively.

# In[33]:
df_19_1 = df[df['industry_name']==top_1]['item_1a']
pol_list1 = []
for s in df_19_1:
    tb = TextBlob(str(s))
    pol_list1.append(tb.sentiment.polarity)

plt.clf()
# In[38]:


plt.hist(pol_list1, bins=10) #, normed=1, alpha=0.75)

plt.xlabel('polarity score for top1')
plt.ylabel('sentence count')
# plt.grid(True)
plt.savefig('hw2_ans19a_26805440.png')

# ### Question 20: Make outfile name format as `hw2_answers_{student_id}.txt` and save it to `txt` file                
# - When you write the answer, please keep format(please refer to word doc example).
# - File name should be like this : <b>hw2_answers_37510930.txt</b>

    

df_19_2 = df[df['industry_name']==top_2]['item_1a']
pol_list2 = []
for s in df_19_2:
    tb = TextBlob(str(s))
    pol_list2.append(tb.sentiment.polarity)


# In[39]:

plt.clf()
plt.hist(pol_list2, bins=10)#, normed=1, alpha=0.75)

plt.xlabel('polarity score for top2')
plt.ylabel('sentence count')

# plt.grid(True)
plt.savefig('hw2_ans19b_26805440.png')

