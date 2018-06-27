#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 16:24:51 2018

@author: user
"""

import sentiment
import pandas as pd
import pickle
import numpy as np

# depickling the dataset
def dump_sentiment(file):
    with open('/home/user/Documents/Alex2/SOP/Steps/Test_Data_Outputs/sentiment.pickle','wb') as f:
        pickle.dump(file,f)
        
def get_df():
    with open('/home/user/Documents/Alex2/SOP/Steps/Test Data/Max Length/final_test_data.pickle','rb') as f:
        df = pickle.load(f,encoding = 'latin1')
        
    return df

def convert_to_forms(df):    
    X_train = df.iloc[0:1008,1]
    X_train = X_train.to_frame()
    
    X_test = df.iloc[1008:,1]
    X_test = X_test.to_frame()
    
    len_train = df.iloc[0:1008,2]    
    len_train = np.asarray(len_train,dtype = np.int32)
    len_train = np.reshape(len_train,newshape = (len_train.shape[0],1))
    
    len_test = df.iloc[1008:,2]
    len_test = np.asarray(len_test,dtype = np.int32)
    len_test = np.reshape(len_test,newshape = (len_test.shape[0],1))
    
    Y_test = df.iloc[1008:,3]
    Y_test = np.asarray(Y_test,dtype = np.int32)
    Y_test = np.reshape(Y_test,newshape = (Y_test.shape[0],1))
    
    Y_train = df.iloc[0:1008,3]
    Y_train = np.asarray(Y_train,dtype = np.int32)
    Y_train = np.reshape(Y_train,newshape = (Y_train.shape[0],1))
    
    X_total = pd.concat([X_train,X_test],axis = 0)
    Y_total = np.concatenate([Y_train,Y_test],axis = 0)
    len_total = np.concatenate( [len_train,len_test],axis = 0 )
    
    return X_total,Y_total,len_total

if __name__ == '__main__':
    
    df = get_df()
    
    df_sentiment = pd.DataFrame(columns = ['Sentiment Values'],index = range(0,1254))
    
    for i in range(0,1254):
        
        review = df.iloc[i,1]
        temp = []
        for sent in review:
            string = ""
            fr = 0
            for word in sent:
                
                if fr == 0:
                    string = word
                    fr += 1
                else:
                    string = string + " " + word
                    fr += 1
                    
            senti_vec = sentiment.main(string)
            
            if len(senti_vec[0]) > 1:  
                print( senti_vec[0] )
                for gh in range( len(senti_vec[0]) ):
                    temp.append( senti_vec[0][gh] )    
            else:
                temp.append(senti_vec[0][0])
        
        df_sentiment.iloc[i,0] = temp
    
    
    dump_sentiment(df_sentiment)
        

        
        
        
        
        