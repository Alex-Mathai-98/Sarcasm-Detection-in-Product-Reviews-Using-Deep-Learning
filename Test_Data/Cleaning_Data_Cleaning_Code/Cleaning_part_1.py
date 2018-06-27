### EXTRACTING THE LIST OF TEXT FILES ###

import os 
import pandas as pd
import pickle
import re

dataset = []
list_of_paths = [] 

### CREATING THE DATASET LIST AND CREATING THE LIST_OF_PATHS LIST ###
home = os.getcwd()

path1 = os.path.join(home,"SOP/Test_Data/Cleaning_Data_Cleaning_Code/SarcasmAmazonReviewsCorpus/Ironic")
os.chdir(path1) # changing the directory to the ironic folder
for dirpath,dirnames,filenames in os.walk(os.getcwd()): # Walking through the ironic folder
    for filename in filenames :
        if ".txt" == (os.path.splitext(filename))[1]: # checking if the file has ".txt" extension
            dataset.append([filename,1]) # adding the final classification
            list_of_paths.append(os.path.join(path1, filename))
              
path2 = os.path.join(home,"SOP/Test_Data/Cleaning_Data_Cleaning_Code/SarcasmAmazonReviewsCorpus/Regular")
os.chdir(path2) # changing the directory to the regular folder
for dirpath,dirnames,filenames in os.walk(os.getcwd()): # Walking through the regular folder
    for filename in filenames :
        if ".txt" == (os.path.splitext(filename))[1]: # checking if the file has ".txt" extension
            dataset.append([filename,0]) # adding the final classification
            list_of_paths.append(os.path.join(path2, filename))
         
### FINDING ALL THE TAGS AND CREATING A PANDAS DATA FRAME ###

df = pd.df = pd.DataFrame(columns=['STARS','TITLE','DATE','AUTHOR','PRODUCT','REVIEW','CLASSIFICATION'],index=range(0,1254))

file_no = 0
for path in list_of_paths:
    with open(path,encoding='latin1',mode = 'r') as f:
        contents = f.read()
        pattern = re.compile(r'<(\w+)>(.+)</(\w+)>') # recognizes all matches all patterns of the type (<Label>, ''' ,<\Label>)
        matches = pattern.finditer(contents) # returns all matches all patterns of the type (<Label>, ''' ,<\Label>)
        for match in matches:
                g = match.groups() # returns one at a time (<Label>, ''' ,<Label>)
                if(g[0] == "STARS"): # checking if the tuple is about TITLE 
                    df.iloc[file_no,0] = g[1] # Storing that information
                elif(g[0] ==  "TITLE"):
                    df.iloc[file_no,1] = g[1]
                elif(g[0] == "DATE"):
                    df.iloc[file_no,2] = g[1]
                elif(g[0] == "AUTHOR"):
                    df.iloc[file_no,3] = g[1]
                elif(g[0] == "PRODUCT"):
                    df.iloc[file_no,4] = g[1]
                else:
                    continue
        review = re.search(re.compile(r"<REVIEW>(.*)</REVIEW>",re.DOTALL),contents).group(1) # Inorder to be able to recognize everything between the 2 labels including the new lines
        df.iloc[file_no,5] = review # Adding the review
        df.iloc[file_no,6] = dataset[file_no][1] # Adding the classification
        file_no += 1                    

pickle_out = open(os.path.join(home,"SOP/Test_Data/Cleaning_Data_Cleaning_Code/test_data.pkl"),"wb")
pickle.dump(df,pickle_out)
pickle_out.close()    




    
    
    
      


    

    
    
    
    
    
    
    