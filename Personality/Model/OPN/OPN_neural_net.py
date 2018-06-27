# Importing Modules :
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# Golbal Variables
global_counter = 0


############################################ UTILITY FUNCTIONS ############################################
def get_latest_essays(n_H0,breaker_length):
    
    path = os.path.join(os.getcwd(),'SOP/Personality/Cleaned_Data_Cleaning_Code/essays_' + str(n_H0) + '_' + str(breaker_length) + '.pkl')
    with open(path,'rb') as f:
        df = pickle.load(f, encoding = 'latin1')
        
    return df

def convert_to_forms(df): 
    
    '''
    Description:
    Shuffles the dataframe and splits the dataframe into training and testing sets
    
    Returns:
    X_train -- Training set of input features
    X_test -- Testing set of input features
    
    Y_train -- Ground truth of training set
    Y_test -- Ground truth of testing set
    
    len_train -- Length of training set
    len_test -- Length of testing set
        
    '''
    
    g = np.arange( len(df) )
    np.random.shuffle(g)

    X = np.asarray((df.iloc[:,1].values)[g])
    X = np.reshape(X,newshape = (X.shape[0],1))

    lengths = np.asarray(df.iloc[:,7],dtype = np.int32)
    lengths = (np.reshape(lengths,newshape = (lengths.shape[0],1)))[g]   

    Y = np.asarray(df.iloc[:,4],dtype = np.int32)
    Y = (np.reshape(Y,newshape = (Y.shape[0],1)))[g]

    X_train = np.reshape(X[0:1984,0],newshape = (1984,1))
    X_test = np.reshape(X[1984:,0], newshape = (483,1))

    Y_train = np.reshape(Y[0:1984,0], newshape = (1984,1))
    Y_test = np.reshape(Y[1984:,0], newshape = (483,1))
    
    len_train = lengths[0:1984,0]
    len_test = lengths[1984:,0]
    
    return X_train,Y_train,len_train,X_test,Y_test,len_test

def create_minibatches(X,Y,batch_size):
    
    '''
    Creates mini_batches of size batch_size
    
    Arguments:
    X : Input
    Y : Ground Truth
    
    Returns:
    temp_list : A list of tuples of input and correspnding ground truths
    
    Note:
    If in the last mini_batch the number of records < batch_size, we then discard it.
    
    '''
    
    num_of_batches = len(X)//batch_size
    temp_list = []
    
    r = 0;
    h = 0;
    for k in range(0,num_of_batches):

        temp_x = X[batch_size*(h):batch_size*(h+1),0]
        temp_x = np.reshape(temp_x,newshape = ((temp_x.shape)[0],1))
        
        temp_y = Y[batch_size*(h):batch_size*(h+1),:]
        temp_y = np.reshape(temp_y, newshape = ( (temp_y.shape)[0],2))
          
        temp_list.append( (temp_x,temp_y) ) # appending the tuple of train batch and the test batch
        
        h += 1
        r += batch_size
    
    return temp_list

def convert_to_one_hot( y_train, num_outputs ):
    
    length = len(y_train)    
    bhim = np.zeros( shape =(num_outputs,length) )    
    for j in range(0,length):
        (bhim[:,j])[ y_train[j,] ] = 1
    
    return bhim

def create_compartment_list(word_list,compartments,max_len):
    
    '''
    Description :
    Breaks the long list of words ("word_list") into a group of smaller lists of words ("ans") of length "max_len" each

    Parameters:
    word_list (int) -- the long list of words as input to be broken
    compartments (int) -- how many parts of the sentence you need
    max_len (int) -- how long must each compartment be

    Returns:
    ans (list) -- A list of a smaller collections of words.

    '''

    ans = []
    
    for k in range(compartments-1):
        ans.append(word_list[k*max_len:(k+1)*max_len])

    ans.append( word_list[(k+1)*max_len:] )

    return ans

def get_broken_sentences(n_H0,breaker_length,minibatch_X,index):

    '''
    Description :
    Breaks the long list of words ("word_list")  in "minibatch_X[index]" into a group of smaller lists of words ("ans")

    Parameters:
    minibatch_X (array) -- A small batch sized collection of inputs
    index (int) -- The index of the input sentence

    Returns:
    actual_words_of_sents (list) -- A list of a smaller collections of words.

    '''
    
    actual_words_of_sents = []

    
    # for sentence in essay
    for words_of_sents in minibatch_X[index,0]:
        
        #print(words_of_sents)
        if len(words_of_sents) <= n_H0:
            
            # No need to break the sentence
            actual_words_of_sents.append(words_of_sents)
            
        else:
            
            # Break the sentence
            if len(words_of_sents)%breaker_length == 0:
                compartments = len(words_of_sents)//breaker_length
                
                collection = create_compartment_list(words_of_sents,compartments,breaker_length)  
                
                for values in collection:
                    actual_words_of_sents.append(values)
                
            else:
                compartments = len(words_of_sents)//breaker_length + 1

                collection = create_compartment_list(words_of_sents,compartments,breaker_length)   
        
                for values in collection:
                    actual_words_of_sents.append(values)

    return actual_words_of_sents

def get_glove_model():
    
    '''
    Description:
    A function that returns a fast dictionary for word_embedding look-ups.

    Returns:
    glove_model : The dictionary that maps words to it's numpy equivalents
    '''

    path = os.path.join(os.getcwd(),'SOP/Glove_Model/glove_model.pickle')
    with open(path,'rb') as f:
        glove_model = pickle.load(f, encoding='latin1')
    
    return glove_model

def get_embedding_equivalent(max_len,word_list,glove_model):
    
    '''
    Description:
    Gets the word-embedding matrix equivalent of the list of words ("word_list")
    Skips a word is it does not exist in the dictionary

    Parameters:
    max_len -- The maximum length of a sentence
    word_list -- The list of words in a sentence
    glove_model -- The python dictionary for the word embeddings of english words

    Returns:
    word_array -- The embedding matrix of the sentence
    '''

    word_array = np.zeros( (max_len,1,300) )
    index = 0
    for g in range(0, len(word_list) ):    
        if word_list[g] in glove_model:
            word_array[index,:,:] = glove_model[ word_list[g] ]   
            index += 1
        elif word_list[g].lower() in glove_model:
            word_array[index,:,:] = glove_model[ word_list[g].lower() ]   
            index += 1
        elif word_list[g].upper() in glove_model:
            word_array[index,:,:] = glove_model[ word_list[g].upper() ]   
            index += 1
        else:
            index += 1
            
    return word_array

def get_lengths():
    return 1984,483

############################################ Neural Network Architecture ############################################

# This function can be re-used for any project
def create_placeholders(n_H0,n_W0, n_C0, n_y):
    """
    Creates the placeholders for the input size and for the number of output classes.
    
    Arguments:
    n_W0 -- scalar, width of an input matrix
    n_C0 -- scalar, number of channels of the input
    n_y  -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """
    
    # Keep the number of examples as a variable (None) and the height of the matrix as variables (None)
    X = tf.placeholder(dtype = tf.float32, shape = (None,n_H0,n_W0,n_C0), name = "X") 
    Y = tf.placeholder(dtype = tf.float32, shape = (None,n_y), name = "Y")
    
    
    return X,Y

def initialize_parameters():
    '''
        Initialize weight parameters for the weight matrix.
        Returns : A weight matrix with all the weights of the neural network
    '''

    #The 3-Gram Kernel
    layer1_a =  tf.get_variable(name = "OPN_layer1_a",shape = (3,1,300,20),dtype = tf.float32,initializer = tf.contrib.layers.xavier_initializer(),trainable = True)
    bias1_a =  tf.get_variable(name = "OPN_bias1_a",shape = (20),dtype = tf.float32,initializer = tf.initializers.zeros(),trainable = True)

    #The 4-Gram Kernel
    layer1_b =  tf.get_variable(name = "OPN_layer1_b",shape = (4,1,300,20),dtype = tf.float32,initializer = tf.contrib.layers.xavier_initializer(),trainable = True)
    bias1_b =  tf.get_variable(name = "OPN_bias1_b",shape = (20),dtype = tf.float32,initializer = tf.initializers.zeros(),trainable = True) 

    #The 5-Gram Kernel
    layer1_c =  tf.get_variable(name = "OPN_layer1_c",shape = (5,1,300,20),dtype = tf.float32,initializer = tf.contrib.layers.xavier_initializer(),trainable = True)
    bias1_c =  tf.get_variable(name = "OPN_bias1_c",shape = (20),dtype = tf.float32,initializer = tf.initializers.zeros(),trainable = True) 

    # The second layer
    layer2 = tf.get_variable(name = "OPN_layer2",shape = (2,1,1,60),dtype = tf.float32,initializer = tf.contrib.layers.xavier_initializer(),trainable = True)
    bias2 = tf.get_variable(name = "OPN_bias2",shape = (60),dtype = tf.float32,initializer = tf.initializers.zeros(),trainable = True)

    # The bias of the last layer
    fc_bias = tf.get_variable(name = "OPN_fc_bias",shape = (2),dtype = tf.float32,initializer = tf.initializers.zeros(),trainable = True)

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

    return weight_parameters

def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- Transpose of one hot encoding of the final classes.
    
    Returns:
    cost - Tensor of the cost function
    """

    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y) )

    return cost

def average_sentences(plato,input_tuple):

    '''
    Description : 
    Takes the average sentence vector output of each essay.
    
    Arguments:
    plato -- The expanded form(sentence wise) of each essay.
    input_tuple -- Tuple that contains the length of each essay.
    
    Returns:
    output_tuple -- A set of 16 vector outputs, each vector being the average of all the sentences in the essay.
    
    '''
    
    # Getting the length of each essay separately
    in1 = input_tuple[0]
    in2 = input_tuple[1]
    in3 = input_tuple[2]
    in4 = input_tuple[3]
    in5 = input_tuple[4]
    in6 = input_tuple[5]
    in7 = input_tuple[6]
    in8 = input_tuple[7]
    in9 = input_tuple[8]
    in10 = input_tuple[9]
    in11 = input_tuple[10]
    in12 = input_tuple[11]
    in13 = input_tuple[12]
    in14 = input_tuple[13]
    in15 = input_tuple[14]
    in16 = input_tuple[15]

    # Taking Average of all the Senteces of an Essay
    # Essay 1
    start = 0
    end = in1
    output1 = tf.reduce_mean(tf.gather(plato,tf.range(start,end,1), axis = 0 )  ,axis = 0,keepdims = True)
    
    # Essay 2
    start = end
    end = end + in2
    output2 = tf.reduce_mean(tf.gather(plato,tf.range(start,end,1), axis = 0 ) ,axis = 0,keepdims = True)
    
    # Essay 3
    start = end
    end = end + in3
    output3 = tf.reduce_mean(tf.gather(plato,tf.range(start,end,1), axis = 0 ) ,axis = 0,keepdims = True)    
    
    # Essay 4
    start = end
    end = end + in4
    output4 = tf.reduce_mean(tf.gather(plato,tf.range(start,end,1), axis = 0 ) ,axis = 0,keepdims = True) 
    
    # Essay 5
    start = end
    end = end + in5
    output5 = tf.reduce_mean(tf.gather(plato,tf.range(start,end,1), axis = 0 ) ,axis = 0,keepdims = True)
    
    # Essay 6
    start = end
    end = end + in6
    output6 = tf.reduce_mean(tf.gather(plato,tf.range(start,end,1), axis = 0 ) ,axis = 0,keepdims = True)
    
    # Essay 7
    start = end
    end = end + in7
    output7 = tf.reduce_mean(tf.gather(plato,tf.range(start,end,1), axis = 0 ) ,axis = 0,keepdims = True)
    
    # Essay 8
    start = end
    end = end + in8
    output8 = tf.reduce_mean(tf.gather(plato,tf.range(start,end,1), axis = 0 ) ,axis = 0,keepdims = True)
    
    # Essay 9
    start = end
    end = end + in9
    output9 = tf.reduce_mean(tf.gather(plato,tf.range(start,end,1), axis = 0 ) ,axis = 0,keepdims = True)
    
    # Essay 10
    start = end
    end = end + in10
    output10 = tf.reduce_mean(tf.gather(plato,tf.range(start,end,1), axis = 0 ) ,axis = 0,keepdims = True)
    
    # Essay 11
    start = end
    end = end + in11
    output11 = tf.reduce_mean(tf.gather(plato,tf.range(start,end,1), axis = 0 ) ,axis = 0,keepdims = True)
    
    # Essay 12
    start = end
    end = end + in12
    output12 = tf.reduce_mean(tf.gather(plato,tf.range(start,end,1), axis = 0 ) ,axis = 0,keepdims = True)
    
    # Essay 13
    start = end
    end = end + in13
    output13 = tf.reduce_mean(tf.gather(plato,tf.range(start,end,1), axis = 0 ) ,axis = 0,keepdims = True) 
    
    # Essay 14
    start = end
    end = end + in14
    output14 = tf.reduce_mean(tf.gather(plato,tf.range(start,end,1), axis = 0 ) ,axis = 0,keepdims = True) 
    
    # Essay 15
    start = end
    end = end + in15
    output15 = tf.reduce_mean(tf.gather(plato,tf.range(start,end,1), axis = 0 ) ,axis = 0,keepdims = True) 
    
    # Essay 16
    start = end
    end = end + in16
    output16 = tf.reduce_mean(tf.gather(plato,tf.range(start,end,1), axis = 0 ) ,axis = 0,keepdims = True) 


    output_tuple = (output1,output2,output3,output4,output5,output6,output7,output8,output9,output10,output11,output12,output13,output14,output15,output16)

    return output_tuple

def forward_propagation(X,parameters,alex_prob,input_tuple):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> Avergae -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters
    alex_prob -- probability for drop_out
    input_tuple -- The set of lengths of each essay

    Returns:
    Z4 -- the output of the last LINEAR unit
    """
    len_train,len_test =  get_lengths()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph till before the neural net
    W1_a = parameters["W1_a"]
    bias1_a = parameters["bias1_a"]

    W1_b = parameters["W1_b"]
    bias1_b = parameters["bias1_b"]

    W1_c = parameters["W1_c"]
    bias1_c = parameters["bias1_c"]

    W2 = parameters["W2"] 
    bias2 = parameters["bias2"]
    
    fc_bias = parameters["fc_bias"]

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
    
    output_tuple = average_sentences(plato,input_tuple)
    
    (output1,output2,output3,output4,output5,output6,output7,output8,output9,output10,output11,output12,output13,output14,output15,output16) = output_tuple
    
    # Concatenating the 16 average-essay vectors
    P4 = tf.concat([output1,output2,output3,output4,output5,output6,output7,output8,output9,output10,output11,output12,output13,output14,output15,output16],axis = 0)
    
    normalizer_fn = tf.contrib.layers.dropout
    normalizer_params = {'keep_prob': alex_prob}
    
    # FULLY-CONNECTED without non-linear activation function (Do not call softmax here).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 
    Z3 = tf.contrib.layers.fully_connected(P4,num_outputs = 2,activation_fn = None,normalizer_fn=normalizer_fn,normalizer_params=normalizer_params)
    Z4 = tf.nn.bias_add(Z3,fc_bias)
    
    return Z4

def model(n_H0,breaker_length,X_train,Y_train,X_test,Y_test,len_train,len_test,learning_rate,num_epochs, minibatch_size, print_cost=True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X_train -- training set, of shape (None,160, 1, 300)
    Y_train -- test set, of shape (None, n_y = 2)
    X_test -- training set, of shape (None,160,1,300)
    Y_test -- test set, of shape (None, n_y = 2)
    len_train -- Contains the lengths of each essay in the training set 
    len_test -- Contains the lengths of each essay in the testing set
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    n_y = Y_train.shape[1]
    n_W0 = 1
    
    costs = []    
                                  
    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0,n_W0,300,n_y)
    alex_prob = tf.placeholder( dtype = tf.float32, shape = (), name = "alex_prob" )
    
    # Minibatch of 16 variable sized essays
    in1 = tf.placeholder( dtype = tf.int32,shape = (),name = "in1" )
    in2 = tf.placeholder( dtype = tf.int32,shape = (), name = "in2" )
    in3 = tf.placeholder( dtype = tf.int32,shape = (), name = "in3" )
    in4 = tf.placeholder( dtype = tf.int32,shape = (), name = "in4" )
    in5 = tf.placeholder( dtype = tf.int32,shape = (), name = "in5" )
    in6 = tf.placeholder( dtype = tf.int32,shape = (), name = "in6" )
    in7 = tf.placeholder( dtype = tf.int32,shape = (), name = "in7" )
    in8 = tf.placeholder( dtype = tf.int32,shape = (), name = "in8" )
    in9 = tf.placeholder( dtype = tf.int32,shape = (), name = "in9" )
    in10 = tf.placeholder( dtype = tf.int32,shape = (), name = "in10" )
    in11 = tf.placeholder( dtype = tf.int32,shape = (), name = "in11" )
    in12 = tf.placeholder( dtype = tf.int32,shape = (), name = "in12" )
    in13 = tf.placeholder( dtype = tf.int32,shape = (), name = "in13" )
    in14 = tf.placeholder( dtype = tf.int32,shape = (), name = "in14" )
    in15 = tf.placeholder( dtype = tf.int32,shape = (), name = "in15" )
    in16 = tf.placeholder( dtype = tf.int32,shape = (), name = "in16" )
    
    input_tuple = (in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11,in12,in13,in14,in15,in16)
    
    # Initialize parameters
    parameters = initialize_parameters()

    # Forward propagation
    Z3 = forward_propagation(X,parameters,alex_prob,input_tuple)
    
    # Accuracy Counting 
    # Calculate the correct predictions
    predict_op = tf.argmax(Z3, 1) # max of columns (axis = 1) for any row is the prediction
    correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1)) # Creating a boolean array with 1 for matched and 0 for not matched
    number = tf.reduce_sum(tf.cast(correct_prediction, "float"),0) # Counting how many mathced
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
    
    # Creating the saving object 
    saver = tf.train.Saver(max_to_keep = 10000)
     
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        #writer = tf.summary.FileWriter("/home/user/Desktop/log", sess.graph)

        # Run the initialization
        sess.run(init)
        
        max_accuracy = -1
        glove_model = get_glove_model()
        
        # Do the training loop
        for epoch in range(num_epochs):
            
            # resetting the global_counter for list of lengths
            accuracy_count = 0
            global_counter = -1
            minibatch_cost = 0.0
            minibatches = create_minibatches(X_train,Y_train,minibatch_size)
            num_minibatches = len(minibatches)
            
            
            for minibatch in minibatches:
                
                (minibatch_X,minibatch_Y) = minibatch                
    
                # incrementing the global_counter
                global_counter += 1
                
                # num = total number of sentences
                num = 0
                for sol in range(batch_size*(global_counter),batch_size*(global_counter+1)):
                    if sol >=  len(len_train):
                        print("Something wrong : This should not happen")
                    else:
                        num += len_train[sol]
                                
                
                #initialization of input
                minibatch_X_modified = np.zeros( shape =(num,n_H0,1,300) )
                
                counter = 0
                for sol in range(len(minibatch_X)):
                                        
                    # Get Shorter Sentences
                    actual_words_of_sents = get_broken_sentences(n_H0,breaker_length,minibatch_X,sol)
                    
                    # for sentence in essay
                    for broken_sentence in actual_words_of_sents:
                        minibatch_X_modified[counter,:,:,:] = get_embedding_equivalent(n_H0,broken_sentence,glove_model)
                        counter += 1
                                        
                assert( counter == num )
                        
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
            
                
                _ ,temp_cost,minibatch_correct = sess.run([optimizer,cost,number], feed_dict = {X:minibatch_X_modified,Y:minibatch_Y, alex_prob : 0.7, in1:(len_train[batch_size*(global_counter)]),in2:(len_train[batch_size*(global_counter) + 1]),in3:(len_train[batch_size*(global_counter) + 2]),in4:(len_train[batch_size*(global_counter) + 3]),in5:(len_train[batch_size*(global_counter) + 4]),
                in6:(len_train[batch_size*(global_counter) + 5]),in7:(len_train[batch_size*(global_counter) + 6]),in8:(len_train[batch_size*(global_counter) + 7]),in9:(len_train[batch_size*(global_counter) + 8]),in10:(len_train[batch_size*(global_counter) + 9]),in11:(len_train[batch_size*(global_counter) + 10]),in12:(len_train[batch_size*(global_counter) + 11]),
                in13:(len_train[batch_size*(global_counter) + 12]),in14:(len_train[batch_size*(global_counter) + 13]),in15:(len_train[batch_size*(global_counter) + 14]),in16:(len_train[batch_size*(global_counter) + 15])} )                
                

                accuracy_count += minibatch_correct
                minibatch_cost += temp_cost / num_minibatches
                
                
            train_accuracy = accuracy_count/1984
            print("Epoch : " + str(epoch))
            print("Train Accuracy : " + str(train_accuracy) )
            
            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
       
            # Saving the accuracies of the train and test data in a file
            if (epoch+1)%5 == 0 :
                
                # Finding out the test Accuracy 
                accuracy_count_test = 0
                f_counter = -1
            
                minibatches_test = create_minibatches(X_test,Y_test,batch_size)
                    
                for minibatch_test in minibatches_test:
                        (minibatch_X_test,minibatch_Y_test) = minibatch_test                
                        
                        f_counter += 1
                        
                        # num = total number of sentences
                        num = 0
                        for sol in range(batch_size*(f_counter),batch_size*(f_counter+1)):
                            if sol >=  len(len_test):
                                print("Something wrong : This should not happen")
                            else:
                                num += len_test[sol]
                        
                        #initialization of input
                        minibatch_X_modified_test = np.zeros( shape =(num,n_H0,1,300) )
                        
                        counter = 0
                        for sol in range(len(minibatch_X_test)):

                            # Get Shorter Sentences
                            actual_words_of_sents = get_broken_sentences(n_H0,breaker_length,minibatch_X_test,sol)

                            # For each sentence in the essay
                            for broken_sentence in actual_words_of_sents:
                                minibatch_X_modified_test[counter,:,:,:] = get_embedding_equivalent(n_H0,broken_sentence,glove_model)
                                counter += 1
                        
                        assert(counter == num)
                        
                        minibatch_correct_test = sess.run([number], feed_dict = {X:minibatch_X_modified_test,Y:minibatch_Y_test, alex_prob : 1.0, in1:(len_test[batch_size*(f_counter)]),in2:(len_test[batch_size*(f_counter) + 1]),in3:(len_test[batch_size*(f_counter) + 2]),in4:(len_test[batch_size*(f_counter) + 3]),in5:(len_test[batch_size*(f_counter) + 4]),
                                                     in6:(len_test[batch_size*(f_counter) + 5]),in7:(len_test[batch_size*(f_counter) + 6]),in8:(len_test[batch_size*(f_counter) + 7]),in9:(len_test[batch_size*(f_counter) + 8]),in10:(len_test[batch_size*(f_counter) + 9]),in11:(len_test[batch_size*(f_counter) + 10]),in12:(len_test[batch_size*(f_counter) + 11]),
                                                     in13:(len_test[batch_size*(f_counter) + 12]),in14:(len_test[batch_size*(f_counter) + 13]),in15:(len_test[batch_size*(f_counter) + 14]),in16:(len_test[batch_size*(f_counter) + 15])} )                
                        
                        accuracy_count_test += minibatch_correct_test[0]                
                        
                test_accuracy = (accuracy_count_test/480 ) 
                print("Test Accuracy:", test_accuracy )
                  
                if epoch == 4 :
                    path = os.path.join(os.getcwd(),'SOP/Personality/Model/OPN/accuracy'+ '_' + str(n_H0) + '_' + str(breaker_length) + '/accuracies.txt')
                    with open(path,'w') as f23:
                        string0 = "\n----" + "\n"
                        string1 = "This is epoch " + str(epoch) + "\n" 
                        string2 = "Train Accuracy: " + str(train_accuracy) + "\n"
                        string3 = "Test Accuracy: " + str( test_accuracy ) + "\n"
                        string4 = "----\n"
                        
                        f23.write( string0 )
                        f23.write( string1 )
                        f23.write( string2 )
                        f23.write( string3 )
                        f23.write( string4 )
                        
                        f23.close()   
                else:
                    path = os.path.join(os.getcwd(),'SOP/Personality/Model/OPN/accuracy'+ '_' + str(n_H0) + '_' + str(breaker_length) + '/accuracies.txt')
                    with open(path,'a') as f23:
                        string0 = "\n----" + "\n"
                        string1 = "This is epoch " + str(epoch) + "\n"
                        string2 = "Train Accuracy: " + str(train_accuracy) + "\n"
                        string3 = "Test Accuracy: " + str( test_accuracy ) + "\n"
                        string4 = "----\n"
                        
                        f23.write( string0 )
                        f23.write( string1 )
                        f23.write( string2 )
                        f23.write( string3 )
                        f23.write( string4 )
                        
                        f23.close()
            
            if (epoch+1)%5 == 0 :
                if max_accuracy < test_accuracy and train_accuracy >= 0.75:
                    max_accuracy = test_accuracy
                    path = os.path.join(os.getcwd(),'SOP/Personality/Model/OPN/parameters'+ '_' + str(n_H0) + '_' + str(breaker_length) + '/personality-OPN-model.ckpt')
                    saver.save(sess,path, global_step=epoch)
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        return 

if __name__ == '__main__':  
    
    n_H0 = 64
    breaker_length = 32

    df =  get_latest_essays(n_H0,breaker_length)    
    X_train,Y_train,len_train,X_test,Y_test,len_test = convert_to_forms(df)
    Y_train_transpose = convert_to_one_hot( Y_train, 2 ).T
    Y_test_transpose = convert_to_one_hot( Y_test, 2 ).T
    batch_size = 16
    model(n_H0,breaker_length,X_train,Y_train_transpose,X_test,Y_test_transpose,len_train,len_test,0.001,100,batch_size, print_cost=True)


