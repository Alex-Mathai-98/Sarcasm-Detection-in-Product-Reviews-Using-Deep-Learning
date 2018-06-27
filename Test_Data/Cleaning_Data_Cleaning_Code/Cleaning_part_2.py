#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 19:10:53 2018

@author: Alex Mathai
"""

import pickle
import numpy as np
from nltk import sent_tokenize,word_tokenize
import pandas as pd
import os

def read_test_data():

    path = os.path.join(os.getcwd(),"SOP/Test_Data/Cleaning_Data_Cleaning_Code/test_data_part_1.pkl")

    with open(path,"rb") as f:
        df = pickle.load(f,encoding = 'latin1')
    
    df_stars = df.iloc[:,0]
    df_title = df.iloc[:,1]
    df_review = df.iloc[:,5]
    df_classi = df.iloc[:,6]
    
    df = pd.concat([df_stars,df_title,df_review,df_classi],axis = 1)
    
    return df

def one_hot_stars(df,index):
    i = (int)((float)(df.iloc[index,0]))
    vec = np.zeros( shape = (5,1) )
    vec[i-1][0] = 1    
    return vec

def split_truth(df):
    
    y = df.iloc[:,6]
    
    y = y.as_array()
    length = len(y)
    
    y_train = y[0:0.8*length,0]
    y_test = y[0.8*length:,0]
    
    return y_train,y_test

def create_compartment_list(word_list,compartments,max_len):
    
    ans = []
    
    for k in range(compartments-1):
        ans.append(word_list[k*max_len:(k+1)*max_len])

    ans.append( word_list[(k+1)*max_len:] )

    return ans

def get_complete_review_and_length(df,index,n_H0,breaker_length):
    
    title = df.iloc[index,1]
    review = df.iloc[index,2]
    
    complete = title + " " + review
    
    list_of_sents = sent_tokenize(complete)
    length = len(list_of_sents)
        
    for batch_of_words in list_of_sents:
        
        # Finding the word embeddings of each word in a sentence 
        if len(batch_of_words) <= n_H0:
            # No need to break the sentence
            continue
        else:
            # Break the sentence
            if len(batch_of_words)%breaker_length == 0:
                compartments = len(batch_of_words)//breaker_length
                
                length += compartments - 1
                
            else:
                compartments = len(batch_of_words)//breaker_length + 1
                
                length += compartments - 1
    
    return complete,length

def jumble(df,num_cols):
        
    g = np.arange( len(df) )
    np.random.shuffle(g)
    
    cols = list(df.columns.values)
    
    for h in range( num_cols ):    
        
        if h == 0 :
            temp = np.asarray(df.iloc[:,h])
            temp = np.reshape( temp,newshape = (len(df),1) )
            temp = temp[g]
            temp = np.reshape( temp,newshape = (len(df)) )
            pm = pd.Series(temp,index = range(len(df)))
            pm = pm.to_frame()
            pm.columns = [cols[h]]
            new_df = pm 
        else:
            temp = np.asarray(df.iloc[:,h])
            temp = np.reshape( temp,newshape = (len(df),1) )
            temp = temp[g]
            temp = np.reshape( temp,newshape = (len(df)) )
            pm = pd.Series(temp,index = range(len(df)))
            pm = pm.to_frame()
            pm.columns = [cols[h]]
            new_df = pd.concat( [new_df,pm], axis = 1 )    
    
    return new_df

def dump_test_data(df):
    
    path = os.path.join(os.getcwd(),"SOP/Test_Data/Cleaning_Data_Cleaning_Code/test_data_part_2.pkl")
    with open(path,"wb") as f:
        pickle.dump(df,f)
        
if __name__ == '__main__':
    
    # Read the dataframe
    df = read_test_data()
    
    # Getting the lengths and concatenating the title and the review
    new_df_stars = df.iloc[:,0]
    new_df_reviews = df.iloc[:,1]
    new_df_lengths = pd.DataFrame( columns = ['lengths'], index = range(1254))
    for k in range(len(df)):
        print(k)
        new_df_stars.iloc[k] = one_hot_stars(df,k)
        new_df_reviews.iloc[k],new_df_lengths.iloc[k,0] = get_complete_review_and_length(df,k,64,32)
    new_df_classi = df.iloc[:,3]
    new_df = pd.concat([new_df_stars,new_df_reviews,new_df_lengths,new_df_classi], axis = 1)
    
    #Jumbling the inputs
    comp_new_df = jumble(new_df,4)
    dump_test_data(comp_new_df)
    
    
        
    
    
    
    
    
