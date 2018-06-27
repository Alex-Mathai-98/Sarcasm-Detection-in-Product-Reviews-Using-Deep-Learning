import numpy as np
from nltk import sent_tokenize
import json, requests

# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000000

class StanfordCoreNLP:
    """
    Modified from https://github.com/smilli/py-corenlp
    """
 
    def __init__(self, server_url):
        # TODO: Error handling? More checking on the url?
        if server_url[-1] == '/':
            server_url = server_url[:-1]
        self.server_url = server_url
 

    def annotate(self, text, properties=None):
        assert isinstance(text, str)
        if properties is None:
            properties = {}
        else:
            assert isinstance(properties, dict)
 
        # Checks that the Stanford CoreNLP server is started.
        try:
            requests.get(self.server_url)
        except requests.exceptions.ConnectionError:
            raise Exception('Check whether you have started the CoreNLP server e.g.\n'
                            '$ cd <path_to_core_nlp_folder>/stanford-corenlp-full-2016-10-31/ \n'
                            '$ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port <port> -timeout <timeout_in_ms>')
 
        data = text.encode()
        r = requests.post(
            self.server_url, params={
                'properties': str(properties)
            }, data=data, headers={'Connection': 'close'})
        output = r.text
        if ('outputFormat' in properties
            and properties['outputFormat'] == 'json'):
            try:
                output = json.loads(output, encoding='utf-8', strict=True)
            except:
                pass
        return output


class sentiment_classifier() :

	def __init__ (self,text) :

		self.text = text
		

	def sentiment_analysis_on_sentence(self,sentence):
    
	    # The StanfordCoreNLP server is running on http://127.0.0.1:9000
	    nlp = StanfordCoreNLP('http://127.0.0.1:9000')
	    
	    # Json response of all the annotations
	    output = nlp.annotate(sentence, properties={
	        "annotators": "tokenize,ssplit,parse,sentiment",
	        
	        "outputFormat": "json",
	        
	        # Setting enforceRequirements to skip some annotators and make the process faster
	        "enforceRequirements": "false"
	    })

	    # In JSON, 'sentences' is a list of Dictionaries, the second number is basically the index of the sentence you want the result of, and each sentence has a 'sentiment' attribute and 'sentimentValue' attribute  
	    # "Very negative" = 0 "Negative" = 1 "Neutral" = 2 "Positive" = 3 "Very positive" = 4  (Corresponding value of sentiment and sentiment value)
	    
	    assert isinstance(output['sentences'], list)
	    
	    return output['sentences']

	def sentence_sentiment(self,sentence):

		# checking if the sentence is of type string
		assert isinstance(sentence, str)

		# getting the json ouput of the different sentences. Type "List"
		result = self.sentiment_analysis_on_sentence(sentence)

		num_of_sentences = len(result)

		sentiment_vec = np.zeros((1,num_of_sentences), dtype = "int64" )

		for i in range(0,num_of_sentences):		
			sentiment_vec[0,i] = ( int(result[i]['sentimentValue']) )

		#print(sentiment_vec[0])

		return sentiment_vec

	def paragraph_sentiment(self):

		sents = sent_tokenize(self.text)

		final_vector = []

		for sent in sents :
			
			vec  = self.sentence_sentiment(sent)
			modified_vec =  vec[0]
			
			if len(modified_vec) > 1 :

				average = 0

				for value in modified_vec :
					average += value

				average = average/len(modified_vec)

				final_vector.append(average)

			else : 
				
				final_vector.append(modified_vec[0])

		return final_vector
	
	def display_value_meanings(self):

		setiment_meaning = {'0':'Very Negative','1': 'Negative','2':'Normal','3':'Good','4':'Very Good'}

		for i in range(len(setiment_meaning)):

			print("{} stands for {}".format(str(i),setiment_meaning[str(i)]))


if __name__ == '__main__':
	
	text = "You are stupid! You're smart and handsome. This is a tool. Rohan is a fantastic person and a great person!"

	text = "I think she makes some good points, and I think some things are just put out there and that she doesn't listen. She just wants only her opinion to be right to an extreme.  She's good at analyzing situations, but she would not be good for a government position requiring much trust to keep stability, that is for sure.  On the other hand, you probably want her to be your Republican lobbyist.  A \"friend\" a \"Coulter Jr.\" told me about how great this book is.  He acts just like Coulter, but just doesn't publish books and goes out and speaks like she does.  Otherwise, he would probably be doing at least okay- (Coulter created and kept her niche first.)  I am not particularly Democrat or Republican, but I try to give everything a chance.  This book, while giving some fresh perspectives I would not have thought of, is quite hit or miss, too opinionated, and not always reasoning things out enough."

	senti = sentiment_classifier(text)
	senti.display_value_meanings()

	vector = senti.paragraph_sentiment()

	print(vector)