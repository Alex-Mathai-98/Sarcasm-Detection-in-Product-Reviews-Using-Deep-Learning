from class_neural_nets import classifier,utilities
from class_sentiment import StanfordCoreNLP,sentiment_classifier
import os
import pickle
import numpy as np

class sarcasm_detector :

	def __init__ (self,text,rating,n_H0,breaker_length) :

		self.text = text
		self.n_H0 = n_H0
		self.breaker_length = breaker_length
		self.rating = rating

		self.uti = utilities()
		self.emo_class = classifier("EMO",self.text,self.uti.glove_model)
		self.agr_class = classifier("AGR",self.text,self.uti.glove_model)
		self.con_class = classifier("CON",self.text,self.uti.glove_model)
		self.ext_class = classifier("EXT",self.text,self.uti.glove_model)
		self.neu_class = classifier("NEU",self.text,self.uti.glove_model)
		self.opn_class = classifier("OPN",self.text,self.uti.glove_model)
		self.senti_class = sentiment_classifier(self.text)
		self.svm = self.get_svm_classifier()

	def get_svm_classifier(self):

		path = os.path.join(os.getcwd(),"Final_Model/svm_clss.pkl")
		with open(path,"rb") as f:
			svm = pickle.load(f)

		return svm

	def convert_stars_to_array(self):

	    ans = np.zeros(shape = (1,5),dtype = np.float )
	    ans[0,self.rating-1] = 1 #rating >= 1
	        
	    return ans

	def convert_sentiment_to_array(self,vector):
	        
		y = np.zeros(shape = (1,70))
		length = len(vector)
        
		counter = 0
		index = 0
		while ( counter < length ) :
		    if index >= 70:
		        index = 0
		    c1  = vector[counter]
		    y[0,c1+index] += 1
		    index += 5
		    counter += 1
		    
		return y

	def predict_Emotion(self) :
		self.emo_class.predict(self.n_H0,self.breaker_length,generate_sentence_vector = False,Predict = True)		

	def predict_AGR(self) :
		self.agr_class.predict(self.n_H0,self.breaker_length,generate_sentence_vector = False,Predict = True)		

	def predict_CON(self) :
		self.con_class.predict(self.n_H0,self.breaker_length,generate_sentence_vector = False,Predict = True)		

	def predict_EXT(self) :
		self.con_class.predict(self.n_H0,self.breaker_length,generate_sentence_vector = False,Predict = True)		

	def predict_NEU(self) :
		self.neu_class.predict(self.n_H0,self.breaker_length,generate_sentence_vector = False,Predict = True)		

	def predict_OPN(self) :
		self.opn_class.predict(self.n_H0,self.breaker_length,generate_sentence_vector = False,Predict = True)				

	def predict_sarcasm(self):

		emo_vector = self.emo_class.predict(self.n_H0,self.breaker_length,generate_sentence_vector = True,Predict = False)#90
		agr_vector = self.agr_class.predict(self.n_H0,self.breaker_length,generate_sentence_vector = True,Predict = False)#90
		con_vector = self.con_class.predict(self.n_H0,self.breaker_length,generate_sentence_vector = True,Predict = False)#90
		ext_vector = self.ext_class.predict(self.n_H0,self.breaker_length,generate_sentence_vector = True,Predict = False)#90
		neu_vector = self.neu_class.predict(self.n_H0,self.breaker_length,generate_sentence_vector = True,Predict = False)#90
		opn_vector = self.opn_class.predict(self.n_H0,self.breaker_length,generate_sentence_vector = True,Predict = False)#90
		star_vector = self.convert_stars_to_array()#05
		senti_vector = self.convert_sentiment_to_array(self.senti_class.paragraph_sentiment())#70

		final_vector = np.concatenate([emo_vector,agr_vector,con_vector,ext_vector,neu_vector,opn_vector,star_vector,senti_vector],axis = 1)

		prediction = self.svm.predict(final_vector)

		if prediction == 1:
			print("This review was sarcastic.")
		else:
			print("This review seems clean of sarcasm.")

		return

		
if __name__ == '__main__':

	print("Please paste your product review in the terminal")
	string = input()
	sc = sarcasm_detector(string,5,64,32)
	sc.predict_sarcasm()
