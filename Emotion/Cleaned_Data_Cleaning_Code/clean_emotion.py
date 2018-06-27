# Importing Modules
from nltk import sent_tokenize,word_tokenize
import pandas as pd
import pickle
import os
import re

def store_gold_std(gold_std,n_H0,breaker_length):
    """
    Description:
    A function to store the gold_std dataframe.

    Parameters:
    gold_std -- the pandas dataframe
    """

    path = os.getcwd()
    path = os.path.join(path,'SOP/Emotion/Cleaned_Data_Cleaning_Code/gold_std_' + str(n_H0) + '_' + str(breaker_length) + '.pickle')
    with open(path,'wb') as f5:
        pickle.dump(gold_std,f5)

def get_actual_length(n_H0,breaker_length,sents) :
    
    length = len(sents)

    for sent in sents:

        words = word_tokenize(sent)
                
        actual_words_of_sents = []

        if len(words) <= n_H0:
            
            # No need to break the sentence
            actual_words_of_sents.append(words)
            
        else:            

            print(length)
            print(len(words))
            
            # Break the sentence
            if len(words)%breaker_length == 0:
                compartments = len(words)//breaker_length
                length += compartments - 1
                
            else:
                compartments = len(words)//breaker_length + 1
                length += compartments - 1
            
            print(length)
    
    return length


def populate_goldstd(n_H0,breaker_length):
    
    '''
    Description: 
    A function to store the text dataset into a pandas dataframe

    '''

    # hp - happy, sd - sadness, ag - anger, dg - disgust, sp - surprise, fr - fear, ne - no emotion
    Category_dictionary = {"hp": 0,"sd":1,"ag":2,"dg":3,"sp":4,"fr":5,"ne":6}

    # dataframe
    gold_std = pd.DataFrame(columns = ['Category','Word_list','Num_sents'],index = range(0,4090))
    path = os.path.join(os.getcwd(),'SOP/Emotion/Cleaned_Data_Cleaning_Code/category_gold_std.txt')    

    i =  0
    with open(path,'r') as f3:
        
        for line in f3:
                        
            pattern = re.compile(r'(\w\w) (\d+. )(.+)')
            matches = pattern.finditer(line)

            for match in matches:
                g = match.groups()

                # Breaking up the senteces
                sents = sent_tokenize(g[2])
                #print(sents)

                # Number of sentences
                num_sents = get_actual_length(n_H0,breaker_length,sents)
                #print(num_sents)

                word_list = []
                for sent in sents:
                    word_list.append(word_tokenize(sent))     
                #print(word_list)
                
                # Storing the emotion
                gold_std.iloc[i,0] = Category_dictionary[g[0]]
                gold_std.iloc[i,1] = word_list
                gold_std.iloc[i,2] = num_sents

            i += 1   
    
    return gold_std

    
if __name__ == '__main__':
    
    n_H0 = 32
    breaker_length = 16
    
    gold_std = populate_goldstd(n_H0,breaker_length)  
    store_gold_std(gold_std,n_H0,breaker_length)




















