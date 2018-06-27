#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 20:56:49 2018

@author: alex
"""
import tensorflow as tf
import numpy as np
from nltk import sent_tokenize,word_tokenize
import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class classifier:
    
    ############################## UTILITY FUNCTIONS ##############################

    def __init__ (self,code_string,input_string,glove_vectors):
        
        if code_string not in ["OPN","NEU","AGR","CON","EXT","EMO"] :
            raise ValueError("Code_string is not valid. Must be one of the following -- [OPN,NEU,AGR,CON,EXT,EMO]")

        self.code_string  = code_string  # One of OPN,NEU,AGR,CON,EXT,EMO
        self.input_string = input_string        
        self.glove_model  = glove_vectors

        # Setting the number of classes
        if self.code_string == "EMO":
            self.n_y = 7

        elif self.code_string == "AGR" or self.code_string == "CON" or self.code_string == "NEU" or self.code_string == "EXT" or self.code_string == "OPN" :
            self.n_y = 2

        # Setting the classes dictionary
        if self.code_string == "EMO" :

            self.classes = {
                                "0" : "Happiness",
                                "1" : "Sadness",
                                "2" : "Anger",
                                "3" : "Disgust",
                                "4" : "Surprise",
                                "5" : "Fear",
                                "6" : "No emotion"
            }

        elif self.code_string == "AGR" :

            self.classes = {

                            "0" : "No trace of AGR.",
                            "1" : "Trace of AGR."
            }

        elif self.code_string == "CON" :

            self.classes = {

                            "0" : "No trace of CON.",
                            "1" : "Trace of CON."
            }

        elif self.code_string == "EXT" :

            self.classes = {
                            "0" : "No trace of EXT",
                            "1" : "Trace of EXT."
            }

        elif self.code_string == "OPN" :

            self.classes = {
                            "0" : "No trace of OPN",
                            "1" : "Trace of OPN."           
            }
        elif self.code_string == "NEU" :

            self.classes = {
                            "0" : "No trace of NEU",
                            "1" : "Trace of NEU."
            }


    def get_input_length_and_words(self,n_H0,breaker_length):

        string = self.input_string

        sents = sent_tokenize(string)
        number_of_sentences = len(sents)

        words_of_sents = []

        for sent in sents :

            words_of_sents.append( word_tokenize(sent) )

        for batch_of_words in words_of_sents :

            if len(batch_of_words) <= n_H0 :
                continue

            else :

                if len(batch_of_words)%breaker_length == 0:

                    compartments = len(batch_of_words)//breaker_length
                    number_of_sentences += compartments - 1

                else:

                    compartments = len(batch_of_words)//breaker_length + 1
                    number_of_sentences += compartments - 1

        self.input_length = (int)(number_of_sentences)
        self.words_of_sents  = words_of_sents

    def create_compartment_list(self,word_list,compartments,max_len):

            ans = []
            
            for k in range(compartments-1):
                ans.append(word_list[k*max_len:(k+1)*max_len])
        
            ans.append( word_list[(k+1)*max_len:] )
        
            return ans
           
    def get_broken_sentences(self,n_H0,breaker_length,words_of_sents):

        actual_words_of_sents = []
        
        # for sentence in essay
        for words_of_a_sent in words_of_sents:
    
            if len(words_of_a_sent) <= n_H0:
                
                # No need to break the sentence
                actual_words_of_sents.append(words_of_a_sent)
                
            else:
                
                # Break the sentence
                if len(words_of_a_sent)%breaker_length == 0:
                    compartments = len(words_of_a_sent)//breaker_length
                    
                    collection = self.create_compartment_list(words_of_a_sent,compartments,breaker_length)  
                    
                    for values in collection:
                        actual_words_of_sents.append(values)
                    
                else:
                    compartments = len(words_of_a_sent)//breaker_length + 1
    
                    collection = self.create_compartment_list(words_of_a_sent,compartments,breaker_length)   
            
                    for values in collection:
                        actual_words_of_sents.append(values)
    
        return actual_words_of_sents   
            
    
    def get_embedding_equivalent(self,max_len,word_list):

        word_array = np.zeros( (max_len,1,300) )
        index = 0
        for g in range(0, len(word_list) ):    
            if word_list[g] in self.glove_model:
                word_array[index,:,:] = self.glove_model[ word_list[g] ]   
                index += 1
            elif word_list[g].lower() in self.glove_model:
                word_array[index,:,:] = self.glove_model[ word_list[g].lower() ]   
                index += 1
            elif word_list[g].upper() in self.glove_model:
                word_array[index,:,:] = self.glove_model[ word_list[g].upper() ]   
                index += 1
            else:
                index += 1
                
        return word_array         

    def get_final_input(self,n_H0,breaker_length):

        self.get_input_length_and_words(n_H0,breaker_length)

        input_length = self.input_length
        words_of_sents = self.words_of_sents

        actual_words_of_sents = self.get_broken_sentences(n_H0,breaker_length,words_of_sents)

        assert( input_length == len(actual_words_of_sents) )

        x_input = np.zeros( shape = (len(actual_words_of_sents),n_H0,1,300) )

        for i in range(input_length) :

            x_input[i,:,:,:] = self.get_embedding_equivalent(n_H0,actual_words_of_sents[i])

        self.x_input = x_input


    def get_correct_path(self,n_H0,breaker_length,code_string) :

        if code_string == "EMO" :
            path = os.path.join(os.getcwd(),"Emotion/Model/parameters_" + str(n_H0) + "_" + str(breaker_length) )

        elif code_string == "AGR" :
            path = os.path.join(os.getcwd(),"Personality/Model/AGR/parameters_" + str(n_H0) + "_" + str(breaker_length) )

        elif code_string == "EXT" :
            path = os.path.join(os.getcwd(),"Personality/Model/EXT/parameters_" + str(n_H0) + "_" + str(breaker_length) )            

        elif code_string == "NEU" :
            path = os.path.join(os.getcwd(),"Personality/Model/NEU/parameters_" + str(n_H0) + "_" + str(breaker_length) )            

        elif code_string == "OPN" :
            path = os.path.join(os.getcwd(),"Personality/Model/OPN/parameters_" + str(n_H0) + "_" + str(breaker_length) )     
        
        elif code_string == "CON" : 
            path = os.path.join(os.getcwd(),"Personality/Model/CON/parameters_" + str(n_H0) + "_" + str(breaker_length) )     
                        
        return path

    ############################## UTILITY FUNCTIONS ##############################


    ############################## NETWORK BULIDING ##############################    

    def create_placeholders(self,n_H0,n_W0,n_C0,code_string):

        self.X = tf.placeholder(dtype = tf.float32, shape = (None,n_H0,n_W0,n_C0), name = code_string + "_X" ) 
        self.alex_prob = tf.placeholder( dtype = tf.float32, shape = (), name = code_string + "_alex_prob"  )

    def initialize_parameters(self,code_string):
        
        layer1_a =  tf.get_variable(name = code_string + "_layer1_a",shape = (3,1,300,20),dtype = tf.float32)
        bias1_a =  tf.get_variable(name = code_string + "_bias1_a",shape = (20),dtype = tf.float32)
    
        layer1_b =  tf.get_variable(name = code_string + "_layer1_b",shape = (4,1,300,20),dtype = tf.float32)
        bias1_b =  tf.get_variable(name = code_string + "_bias1_b",shape = (20),dtype = tf.float32) 
    
        layer1_c =  tf.get_variable(name = code_string + "_layer1_c",shape = (5,1,300,20),dtype = tf.float32)
        bias1_c =  tf.get_variable(name = code_string + "_bias1_c",shape = (20),dtype = tf.float32) 
    
        layer2 = tf.get_variable(name = code_string + "_layer2",shape = (2,1,1,60),dtype = tf.float32)
        bias2 = tf.get_variable(name = code_string + "_bias2",shape = (60),dtype = tf.float32)
    
        fc_bias = tf.get_variable(name = code_string + "_fc_bias",shape = (self.n_y),dtype = tf.float32)
    
        weight_parameters = {}
    
        weight_parameters["W1_a"] = layer1_a
        weight_parameters["bias1_a"] = bias1_a
    
        weight_parameters["W1_b"] = layer1_b
        weight_parameters["bias1_b"] = bias1_b
    
        weight_parameters["W1_c"] = layer1_c
        weight_parameters["bias1_c"] = bias1_c
    
        weight_parameters["W2"] = layer2
        weight_parameters["bias2"] = bias2
    
        weight_parameters["fc_bias"] = fc_bias
    
        self.weight_parameters = weight_parameters
        
    
    def compute_cost(self,Z3, Y):

        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y) )

        return cost
    
    def average_sentences(self,plato,input_length) :

        start = 0
        #end = tf.cast(input_length,dtype = tf.int32)
        end = input_length

        output = tf.reduce_mean(tf.gather(plato,tf.range(start,end,1), axis = 0 )  ,axis = 0,keepdims = True)

        return output

    
    def forward_propagation(self,X,parameters,input_length,n_y):  
    
        # Forward propagation: Build the forward propagation in the tensorflow graph till before the neural net
        W1_a = parameters["W1_a"]
        bias1_a = parameters["bias1_a"]

        W1_b = parameters["W1_b"]
        bias1_b = parameters["bias1_b"]

        W1_c = parameters["W1_c"]
        bias1_c = parameters["bias1_c"]

        W2 = parameters["W2"] 
        bias2 = parameters["bias2"]
        
        X = tf.cast(X,dtype = tf.float32)
        
        # CONV2D: filters W1_a,W1_b,W1_c
        Z1_a = tf.nn.conv2d(X,W1_a,strides = [1,1,1,1],padding='VALID')
        Z1_bias_a = tf.nn.bias_add(Z1_a,bias1_a)

        Z1_b = tf.nn.conv2d(X,W1_b,strides = [1,1,1,1],padding='VALID')
        Z1_bias_b = tf.nn.bias_add(Z1_b,bias1_b)

        Z1_c = tf.nn.conv2d(X,W1_c,strides = [1,1,1,1],padding='VALID')
        Z1_bias_c = tf.nn.bias_add(Z1_c,bias1_c)


        #Remaining sentence length for 3-kernel,4-kernel,5-kernel
        rem1_a = Z1_bias_a.shape[1]
        rem1_b = Z1_bias_b.shape[1]
        rem1_c = Z1_bias_c.shape[1]

        #Assert Dimensions
        assert( Z1_bias_a.shape[1:] == (rem1_a,1,20) )
        assert( Z1_bias_b.shape[1:] == (rem1_b,1,20) )
        assert( Z1_bias_c.shape[1:] == (rem1_c,1,20) )

        #Relu
        A1_bias_a = tf.nn.relu(Z1_bias_a)
        A1_bias_b = tf.nn.relu(Z1_bias_b)
        A1_bias_c = tf.nn.relu(Z1_bias_c)

        # MAXPOOL TO GET MAXIMUM VALUE OF 1 FILTER:
        P1_a = tf.nn.max_pool(A1_bias_a,ksize = [1,rem1_a,1,1],strides = [1,1,1,1],padding='VALID') # the third 1 and second 1 is redundant
        P1_b = tf.nn.max_pool(A1_bias_b,ksize = [1,rem1_b,1,1],strides = [1,1,1,1],padding='VALID') # the third 1 and second 1 is redundant
        P1_c = tf.nn.max_pool(A1_bias_c,ksize = [1,rem1_c,1,1],strides = [1,1,1,1],padding='VALID') # the third 1 and second 1 is redundant

       
        #Concat all three
        L1_ver1 = tf.concat([P1_a,P1_b,P1_c],axis = 3)
        assert( L1_ver1.shape[1:] == (1,1,60) )
        

        # MAXPOOL TO GET MAX VALUE OF 2 FILTERS
        # IMPLEMENTATION CAVEAT --> TO RESHAPE NUMPY ARRAYS WITH A UNKNOWN INDEX, USE THE UNKNOWN INDEX AS TF.SHAPE(x)[INDEX]
        L1_ver2 = tf.reshape(L1_ver1,shape = (tf.shape(L1_ver1)[0],1,60,1))    
        L1_ver3 = tf.nn.max_pool(L1_ver2,ksize = [1,1,2,1],strides = [1,1,2,1],padding='VALID') # the second one is redundant
        assert( L1_ver3.shape[1:] == (1,30,1) )

        L1_final = tf.reshape(L1_ver3,shape = (tf.shape(L1_ver3)[0],30,1,1) ) # reshaping to the same type of dimensions of your input
        
        # Breaking the filter size into three parts   
        L1_final_a = tf.slice(L1_final,[0,0,0,0],[tf.shape(L1_final)[0],10,1,1])
        L1_final_b = tf.slice(L1_final,[0,10,0,0],[tf.shape(L1_final)[0],10,1,1])
        L1_final_c = tf.slice(L1_final,[0,20,0,0],[tf.shape(L1_final)[0],10,1,1])

        # CONV2D: filters W2
        Z2_a = tf.nn.conv2d(L1_final_a,W2,strides=[1,1,1,1],padding='VALID') # the third one is redundant
        Z2_bias_a = tf.nn.bias_add(Z2_a,bias2)

        Z2_b = tf.nn.conv2d(L1_final_b,W2,strides=[1,1,1,1],padding='VALID') # the third one is redundant
        Z2_bias_b = tf.nn.bias_add(Z2_b,bias2)

        Z2_c = tf.nn.conv2d(L1_final_c,W2,strides=[1,1,1,1],padding='VALID') # the third one is redundant 
        Z2_bias_c = tf.nn.bias_add(Z2_c,bias2)

        #Relu
        A2_bias_a = tf.nn.relu(Z2_bias_a)
        A2_bias_b = tf.nn.relu(Z2_bias_b)
        A2_bias_c = tf.nn.relu(Z2_bias_c)

        # MAXPOOL TO GET MAXIMUM VALUE OF 1 FILTERS
        P2_a = tf.nn.max_pool(A2_bias_a, ksize=[1,9,1,1], strides=[1,1,1,1],padding='VALID') # the third one and second 1 is redundant
        P2_b = tf.nn.max_pool(A2_bias_b, ksize=[1,9,1,1], strides=[1,1,1,1],padding='VALID') # the third one and second 1 is redundant
        P2_c = tf.nn.max_pool(A2_bias_c, ksize=[1,9,1,1], strides=[1,1,1,1],padding='VALID') # the third one and second 1 is redundant

        P2_ver1 = tf.concat( [P2_a,P2_b,P2_c] , axis = 3 ) #concatenation of all three outputs

        # MAXPOOL TO GET MAXIMUM VALUE OF 2 FILTERS
        P2_ver2 = tf.reshape(P2_ver1,shape = (tf.shape(P2_ver1)[0],1,180,1))
        P3 = tf.nn.max_pool(P2_ver2,ksize = [1,1,2,1],strides = [1,1,2,1],padding='VALID') # the second 1 is redundant
        assert( P3.shape[1:] == (1,90,1) )
        
        # FLATTEN
        plato = tf.reshape( P3,shape = ( tf.shape(P3)[0],90) )
        assert( plato.shape[1:] == (90) )
        
        # Output
        output = self.average_sentences(plato,input_length)    
        
        return output 
    
    def fully_connected(self,output,alex_prob,parameters):

        fc_bias = parameters["fc_bias"]

        normalizer_fn = tf.contrib.layers.dropout
        normalizer_params = {'keep_prob': alex_prob}

        # FULLY-CONNECTED without non-linear activation function (Do not call softmax here).
        # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 
        Z3 = tf.contrib.layers.fully_connected(output,num_outputs = self.n_y,activation_fn = None,normalizer_fn=normalizer_fn,normalizer_params=normalizer_params)
        Z4 = tf.nn.bias_add(Z3,fc_bias)

        return Z4 

    def get_dimensions(self,n_H0,breaker_length):

        self.get_final_input(n_H0,breaker_length)

        return self.x_input.shape[2],self.x_input.shape[3]
        
    def predict(self,n_H0,breaker_length,generate_sentence_vector = True,Predict = False):
        
        
        with tf.Graph().as_default() as net1_graph:
        
            n_W0,n_C0 = self.get_dimensions(n_H0,breaker_length)
            n_y = self.n_y
            
            # Placeholders
            self.create_placeholders(n_H0,n_W0, n_C0,self.code_string)
    
            # Initialize parameters
            self.initialize_parameters(self.code_string)
            
            # Forward propagation: Build the forward propagation in the tensorflow graph
            output = self.forward_propagation(self.X,self.weight_parameters,self.input_length,n_y)
            
            # Fully connected Layer
            Z3 = self.fully_connected(output,self.alex_prob,self.weight_parameters)
    
            # Calculate the predictions
            predict_op = tf.argmax(Z3, 1) # max of columns (axis = 1) for any row is the prediction
    
            # Creating the saving object 
            saver = tf.train.Saver()
                
        # Start the session to compute the tensorflow graph
        with tf.Session(graph = net1_graph) as sess:
            
            # tf.Graph() gives us a brand new new graph
            # Now the variables that will be stored will be on a new graph all together
        
            #, latest_filename = 'emotion-model.ckpt-199'
            
            path = self.get_correct_path(n_H0,breaker_length,self.code_string)
            ckpt = tf.train.get_checkpoint_state(path)
            
            if ckpt and ckpt.model_checkpoint_path:
                print("Restoration of parameters of model with code_string " + self.code_string + " has been successfull")
                saver.restore(sess,ckpt.model_checkpoint_path)
            
            x_input = self.x_input

            [sent_vector,prediction] = sess.run([output,predict_op], feed_dict = { self.X : x_input, self.alex_prob : 1.0 })

            if Predict == True and generate_sentence_vector == False: 
                output = self.classes[str(prediction[0])]
                print("The sentence has traces of " + str(output) + ".")
            elif Predict == False and generate_sentence_vector == True:
                print( self.code_string + " Model Vector has been returned !")
                return sent_vector
            elif Predict == True and generate_sentence_vector == True: 
                output = self.classes[str(prediction[0])]
                print("The sentence has traces of " + str(output) + ".")
                return sent_vector         
            else :
                print(" \"generate_sentence_vector\" and \"Predict cannot both simultaneously be False.\" ")


class utilities : 

    def __init__ (self):

        self.get_glove_model()

    def get_glove_model(self):
        '''
        Description:
        A function that returns a fast dictionary for word_embedding look-ups.

        Returns:
        glove_model : The dictionary that maps words to it's numpy equivalents
        '''
        #print("Current path : {}, Path to glove : {}".format(os.getcwd(),os.path.join(os.getcwd(),'Glove_Model/glove_model.pickle')))


        path = os.path.join(os.getcwd(),'Glove_Model/glove_model.pickle')
        with open(path,'rb') as f:
            glove_model = pickle.load(f, encoding='latin1')
        
        self.glove_model =  glove_model

        print("Glove Vectors have been loaded successfully.")
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    