import numpy as np
import pickle

def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open("/home/user/Documents/Alex2/glove.840B.300d.txt",'r')
    model = {}
    line_no = 1
    
    
    for line in f:
        if line_no == 52344: 
            line_no += 1
        else:
            splitLine = line.split(' ')
            word = splitLine[0]
            print(line_no)
            embedding = np.zeros( (1,300) )
            
            index = 0
            for val in splitLine[1:]:
                embedding[0,index] = val
                index += 1
                
            model[word] = embedding
            line_no += 1
    
    print ("Done.",len(model)," words loaded!")
    return model

if __name__ == '__main__':    
    model = loadGloveModel("glove.840B.300d.txt")
    with open('/home/user/Documents/Alex2/SOP/Steps/Glove_Model/glove_model.pickle', 'wb') as f:
        pickle.dump(model, f)    