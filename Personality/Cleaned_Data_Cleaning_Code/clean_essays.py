#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 11:38:37 2018

@author: Alex Mathai
"""
import numpy as np
import re
import pandas as pd
from nltk import sent_tokenize,word_tokenize
import pickle
import os


def main(n_H0,breaker_length) :
	
    pattern = re.compile(r'(\d+\_\d+)\.txt,"(.+)",(\w),(\w),(\w),(\w),(\w)') # recognizes all matches all patterns of a certain type 
    dictionary = {'y':1,'n':0}

    df = pd.DataFrame( columns = ["Text","Words_of_sents","Ext","Neu","Agr","CON","OPN","Number_of_Sentences"],index = range(2467) )
    entry_num = 0

    path = os.path.join(os.getcwd(),'SOP/Personality/Cleaned_Data_Cleaning_Code/essays.txt')

    with open(path,'r') as file:
        
        for line in file:
            matches = pattern.finditer(line) # returns all matches of the pattern
            
            for match in matches:
               
                g = match.groups()     
                
                g[1].replace("�","\'")
                g[1].replace("\\\'","\'")
                g[1].replace("/","\'")
                
                # Storing the text
                df.iloc[entry_num,0] = g[1]
                
                sents = sent_tokenize(g[1])
                number_of_sentences = len(sents)
                words_of_sents = []
                
                for sent in sents:
                    words_of_sents.append( word_tokenize(sent) )
                    
                for batch_of_words in words_of_sents:
                    
                    if len(batch_of_words) <= n_H0:
                        continue
                        
                    else:
                        
                        # Break the sentence
                        if len(batch_of_words)%breaker_length == 0:
                            compartments = len(batch_of_words)//breaker_length
                            number_of_sentences += compartments - 1                        
                            
                        else:
                            compartments = len(batch_of_words)//breaker_length + 1
                            number_of_sentences += compartments - 1

                                
                df.iloc[entry_num,1] = words_of_sents            
                df.iloc[entry_num,2] = dictionary[g[2]]
                df.iloc[entry_num,3] = dictionary[g[3]]
                df.iloc[entry_num,4] = dictionary[g[4]]
                df.iloc[entry_num,5] = dictionary[g[5]]
                df.iloc[entry_num,6] = dictionary[g[6]]
                df.iloc[entry_num,7] = number_of_sentences
                
                entry_num += 1

    path = os.path.join(os.getcwd(),'SOP/Personality/Cleaned_Data_Cleaning_Code/essays_'  + str(n_H0) + '_' + str(breaker_length) + '.pkl')
    with open(path,'wb') as f:
        pickle.dump(df,f)


if __name__ == '__main__' :

    n_H0 = 64
    breaker_length = 32

    main(n_H0,breaker_length)