import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from class_neural_nets import classifier,utilities
from class_sentiment import StanfordCoreNLP,sentiment_classifier


class sarcasm_detector :

    def __init__ (self,n_H0,breaker_length) :

        self.dataframe = self.load_dataframe()
        self.n_H0 = n_H0
        self.breaker_length = breaker_length

        sample_text = "Hi I'm Alex. The creator of this Deep Learning Project. Nice to meet you !!"

        self.uti = utilities()
        self.emo_class = classifier("EMO",sample_text,self.uti.glove_model)
        self.agr_class = classifier("AGR",sample_text,self.uti.glove_model)
        self.con_class = classifier("CON",sample_text,self.uti.glove_model)
        self.ext_class = classifier("EXT",sample_text,self.uti.glove_model)
        self.neu_class = classifier("NEU",sample_text,self.uti.glove_model)
        self.opn_class = classifier("OPN",sample_text,self.uti.glove_model)
        self.senti_class = sentiment_classifier(sample_text)

    
    def load_dataframe(self):
        
        path = os.path.join(os.getcwd(),"Test_Data/Cleaning_Data_Cleaning_Code/test_data_part_2.pkl")
        with open(path,"rb") as f:
            df = pickle.load(f,encoding = 'latin1')

        return df

    def convert_sentiment_to_array(self,arr):
	        
	    y = np.zeros( shape = (1,70) )
	    length = len(arr)
	    
	    counter = 0
	    index = 0
	    while ( counter < length ) :
	        if index >= 70:
	            index = 0
	        c1  = (arr)[counter]
	        y[0,c1+index] += 1
	        index += 5
	        counter += 1
	        
	    return y

    def train_svm_and_classify(self):

        x_input = np.zeros( shape = (len(self.dataframe),615) )
        y_output = np.zeros( shape = (len(self.dataframe),1) )

        for i in range( len(self.dataframe) ) :

            text = self.dataframe.iloc[i,1]

            self.emo_class.input_string = text
            self.agr_class.input_string = text
            self.con_class.input_string = text
            self.ext_class.input_string = text
            self.neu_class.input_string = text
            self.opn_class.input_string = text
            self.senti_class.input_string = text

            con_vector = self.con_class.predict(self.n_H0,self.breaker_length,generate_sentence_vector = True,Predict = False)#90
            emo_vector = self.emo_class.predict(self.n_H0,self.breaker_length,generate_sentence_vector = True,Predict = False)#90
            agr_vector = self.agr_class.predict(self.n_H0,self.breaker_length,generate_sentence_vector = True,Predict = False)#90
            ext_vector = self.ext_class.predict(self.n_H0,self.breaker_length,generate_sentence_vector = True,Predict = False)#90
            neu_vector = self.neu_class.predict(self.n_H0,self.breaker_length,generate_sentence_vector = True,Predict = False)#90
            opn_vector = self.opn_class.predict(self.n_H0,self.breaker_length,generate_sentence_vector = True,Predict = False)#90
            senti_vector = self.convert_sentiment_to_array(self.senti_class.paragraph_sentiment())#70
            star_vector = np.transpose(self.dataframe.iloc[i,0],axes = (1,0))


            final_vector = np.concatenate([emo_vector,agr_vector,con_vector,ext_vector,neu_vector,opn_vector,star_vector,senti_vector],axis = 1)
            truth_vector = self.dataframe.iloc[i,3]

            x_input[i,:] = final_vector
            y_output[i,:] = truth_vector

            print("{} is done !".format(i))

        clf = SVC(C=0.5,gamma=0.005)
        clf = clf.fit(x_input[:999,:],y_output[:999])
        
        with open("./svm_clss.pkl","wb") as f :
            pickle.dump(clf,f)
    
        predicted_test1 = clf.predict(x_input[999:,:])
        result1 = metrics.accuracy_score( predicted_test1,y_output[999:] )
        f1score = metrics.f1_score(y_output[999:],predicted_test1)
        precision_score = metrics.precision_score(y_output[999:],predicted_test1)
        recall_score = metrics.recall_score(y_output[999:],predicted_test1)
        print( result1 )
        print(f1score)
        print(precision_score)
        print(recall_score)
    


if __name__ == '__main__':

	try_out = sarcasm_detector(64,32)    
	try_out.train_svm_and_classify()