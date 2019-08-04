'''
!wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
!unzip stanford-corenlp-full-2018-10-05.zip
!pip install stanfordcorenlp
'''

#IMPORTS
import re
import nltk
from stanfordcorenlp import StanfordCoreNLP
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#INITIALIZING
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
nlp = StanfordCoreNLP('./stanford-corenlp-full-2018-10-05')




'''
FUNCTIONS
'''

#PREPROCESSING
def pre_process(doc):
  noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
  noun_pairs = []  
  new_doc = ''
  
  doc = re.sub('\S*@\S*\s?', '', doc)
  doc = re.sub('\s+', ' ', doc)
  doc = re.sub("\'", "", doc) 
  
  doc_lower = doc.lower() 
     
  pos_tags = list(nlp.pos_tag(doc_lower))
  
  for i in range(len(pos_tags)-1):
    if (pos_tags[i][1] in noun_tags) & (pos_tags[i+1][1] in noun_tags):
      noun_pairs.append(i)
     
  for i in range(len(pos_tags)):
    if i in noun_pairs:
      new_doc += str(pos_tags[i][0]) + '-'
    else:
      new_doc += str(pos_tags[i][0]) + ' '
      
  return new_doc
 
 
#COREFRENCE RESOLUTION
def coref(doc):
  coref_list = nlp.coref(doc)
  coref_set = []
  for i in range(len(coref_list)):
    similar_tokens = []
    cluster = coref_list[i]
    for j in range(len(cluster)):
      similar_tokens.append(cluster[j][-1])
    
    coref_set.append(similar_tokens)
  
  return coref_set
  
#GETTING OPINION PAIRS
def get_opinion_pairs(doc, polar_threshold):
  doc = pre_process(doc)
  coref_list = coref(doc)
  parses = nlp.dependency_parse(doc)
  
  targets = []
  opinion_pairs = []
  final_opinion_pairs = {}
  id2token = {}
  
  tokens = nlp.word_tokenize(doc)
  
  for i in range(len(tokens)):
    id2token.update({ i+1 : tokens[i]})
  
  for triplet in parses:
    relation  = triplet[0] 
    if relation == 'ROOT':
      continue
      
    governor, dependent = id2token[(triplet[1])], id2token[(triplet[2])]
    
    gov_polarity = (sid.polarity_scores(governor))['compound']
    dep_polarity = (sid.polarity_scores(dependent))['compound']
    
    if  (relation == 'nsubj') & (abs(gov_polarity) >= polar_threshold):
      opinion_pairs.append((dependent, governor))
      
    elif (relation in ['amod']) & (bool(nlp.pos_tag(dependent)[0][1] in ['JJ', 'JJR', 'JJS'])):
      opinion_pairs.append((governor, dependent))
      
    elif (relation in ['dobj']) & (bool(nlp.pos_tag(governor)[0][1] in ['JJ', 'JJR', 'JJS'])):
      opinion_pairs.append((dependent, governor))
      
  opinion_pairs = list(opinion_pairs)
  
  for i in range(len(opinion_pairs)):
    opinion_pairs[i] = list(opinion_pairs[i])
    for cluster in coref_list:
      if opinion_pairs[i][0] in cluster:
        for item in cluster:
          if bool(nlp.pos_tag(item)[0][1] not in ['PRP', 'PRP$', 'WP', 'WP$']):
            opinion_pairs[i][0] = item
  
  for i in range(len(opinion_pairs)):
    if opinion_pairs[i][0] not in targets:
      targets.append(opinion_pairs[i][0])      
      final_opinion_pairs.update({opinion_pairs[i][0] : (opinion_pairs[i][1]).split()})
    else:   
      if opinion_pairs[i][1] not in final_opinion_pairs[opinion_pairs[i][0]]:
        final_opinion_pairs[opinion_pairs[i][0]].append(opinion_pairs[i][1])
  
  
  return final_opinion_pairs
  
  
'''
USAGE EXAMPLE:

get_opinion_pairs("Your Text", THRESHOLD FOR POLARITY OF OPINION-WORDS)
'''