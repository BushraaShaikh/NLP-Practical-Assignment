#NLP PRACTICAL ASSIGNMENT
##Assignment No.01##
#Title:"Text Pre-Processing using NLP operations:perform Tokenization
# Stop word removal,Lemmatization ,Part-of-Speech Tagging use any sample text"

#import libraries
import spacy

# Load the language model
nlp = spacy.load("en_core_web_sm")

# Define the input text with spaces between sentences
about_text = (
   "India is my country. "
   "Maharashtra is my state."
)

# 1. Tokenization:
about_doc = nlp(about_text)
print("1. Tokenization:")
for token in about_doc:
    print(token, token.idx)

# 2. Stop Words Removal:
about_doc = nlp(about_text)
print("\n2. Stop Words Removal:")
print([token for token in about_doc if not token.is_stop])

# 3. Lemmatization:
about_doc = nlp(about_text)
print("\n3. Lemmatization:")
for token in about_doc:
    if str(token) != str(token.lemma_):
        print(f"{str(token):>20} : {str(token.lemma_)}")

# 4. Part of Speech Tagging:
about_doc = nlp(about_text)
print("\n4. Part of Speech Tagging:")
for token in about_doc:
    print(
        f"""
TOKEN: {str(token)}
=====
TAG: {str(token.tag_):10} POS: {token.pos_}
EXPLANATION: {spacy.explain(token.tag_)}"""
    )

#----------output-------#
"""India 0
is 6
my 9
country 12
. 18
Maharashtra 20
is 32
my 35
state 38
. 43
[India, country, ., Maharashtra, state, .]
is : be
is : be
TOKEN: India
=====
TAG: NNP        POS: PROPN
EXPLANATION: noun, proper singular

TOKEN: is
=====
TAG: VBZ        POS: AUX
EXPLANATION: verb, 3rd person singular present

TOKEN: my
=====
TAG: PRP$       POS: DET
EXPLANATION: pronoun, possessive

TOKEN: country
=====
TAG: NN         POS: NOUN
EXPLANATION: noun, singular or mass

TOKEN: .
=====
TAG: .          POS: PUNCT
EXPLANATION: punctuation mark, sentence closer

TOKEN: Maharashtra
=====
TAG: NNP        POS: PROPN
EXPLANATION: noun, proper singular

TOKEN: is
=====
TAG: VBZ        POS: AUX
EXPLANATION: verb, 3rd person singular present

TOKEN: my
=====
TAG: PRP$       POS: DET
EXPLANATION: pronoun, possessive

TOKEN: state
=====
TAG: NN         POS: NOUN
EXPLANATION: noun, singular or mass

TOKEN: .
=====
TAG: .          POS: PUNCT
EXPLANATION: punctuation mark, sentence closer"""



##Assignment No.02##

#Title:Assignment to implement Bag of Words and TFIDF using Gensim library.
import gensim
from gensim import corpora
from gensim.utils import simple_preprocess

text2 = ["""I love programming
         Python is my favorite programming language.
         Programming allows me to solve real-world problems."""]

tokens2 = [[item for item in line.split()] for line in text2]
g_dict2 = corpora.Dictionary(tokens2)

print("The dictionary has: " + str(len(g_dict2)) + " tokens\n")
print(g_dict2.token2id)

g_bow2 = [g_dict2.doc2bow(token, allow_update=True) for token in tokens2]
print("Bag of Words : ", g_bow2)

text3 = ["""I love programming
         Python is my favorite programming language.
         Programming allows me to solve real-world problems."""]

g_dict3 = corpora.Dictionary([simple_preprocess(line) for line in text3])
g_bow3 = [g_dict3.doc2bow(simple_preprocess(line)) for line in text3]

print("\nDictionary : ")
for item in g_bow3:
    print([[g_dict3[id], freq] for id, freq in item])

    
##OUTPUT##
'''
The dictionary has: 12 tokens

{'I': 0, 'Python': 1, 'allows': 2, 'favorite': 3, 'is': 4, 'language.': 5, 'love': 6, 'me': 7, 'my': 8, 'programming': 9, 'real-world': 10, 'solve': 11}
Bag of Words :  [[(0, 1), (6, 1), (9, 1)], [(1, 1), (3, 1), (4, 1), (5, 1), (8, 1), (9, 1)], [(2, 1), (7, 1), (10, 1), (11, 1), (9, 1)]]

Dictionary : 
[['I', 1], ['love', 1], ['programming', 1]]
[['Python', 1], ['favorite', 1], ['is', 1], ['language', 1], ['my', 1], ['programming', 1]]
[['allows', 1], ['me', 1], ['real-world', 1], ['solve', 1], ['programming', 1]]

'''


###  Assignment No 3 ###
#Assignment Title : Name Entity Recognition in python with spacy

import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

def perform_ner(text):
    # Process the text using SpaCy
    doc = nlp(text)
    
    # Extract named entities and their labels
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    return entities

if __name__ == "__main__":
    # Example text
    text = "Earth is the third planet from the Sun in our solar system and the only known celestial body to support life. With a diverse range of ecosystems, it is home to a vast array of plant and animal species, including humans."

    # Perform Named Entity Recognition
    named_entities = perform_ner(text)

    # Print the results
    print("Named Entities:")
    for entity, label in named_entities:
        print(f"{entity} - {label}")

'''
**************    OUTPUT

Named Entities:
Earth - LOC
third - ORDINAL
Sun - ORG

'''



###  Assignment No 4 ###

"""Assignment Title : Implement Bi-gram, Tri-gram word sequence and its count in text input
data using NLTK library"""

from nltk import ngrams
from nltk.util import ngrams

#unigram model
n = 1
sentence = 'Earth is the third planet from the Sun in our solar system and the only known celestial body to support life. With a diverse range of ecosystems, it is home to a vast array of plant and animal species, including humans.'

unigrams = ngrams(sentence.split(), n)
print(f"\n***********   UNIGRAM    ************************")
for item in unigrams:
    print(item)
#bigram model
n = 2
sentence = 'Earth is the third planet from the Sun in our solar system and the only known celestial body to support life. With a diverse range of ecosystems, it is home to a vast array of plant and animal species, including humans.'

unigrams = ngrams(sentence.split(), n)
print(f"\n***********   BIGRAM    ************************")
for item in unigrams:
    print(item)

#trigram model
n = 3
sentence = 'Earth is the third planet from the Sun in our solar system and the only known celestial body to support life. With a diverse range of ecosystems, it is home to a vast array of plant and animal species, including humans.'
unigrams = ngrams(sentence.split(), n)
print(f"\n***********   TRIGRAM    ************************")
for item in unigrams:
    print(item)

'''
************    OUTPUT    ********************

***********   UNIGRAM    ************************
('Earth',)
('is',)
('the',)
('third',)
('planet',)
('from',)
('the',)
('Sun',)
('in',)
('our',)
('solar',)
('system',)
('and',)
('the',)
('only',)
('known',)
('celestial',)
('body',)
('to',)
('support',)
('life.',)
('With',)
('a',)
('diverse',)
('range',)
('of',)
('ecosystems,',)
('it',)
('is',)
('home',)
('to',)
('a',)
('vast',)
('array',)
('of',)
('plant',)
('and',)
('animal',)
('species,',)
('including',)
('humans.',)

***********   BIGRAM    ************************
('Earth', 'is')
('is', 'the')
('the', 'third')
('third', 'planet')
('planet', 'from')
('from', 'the')
('the', 'Sun')
('Sun', 'in')
('in', 'our')
('our', 'solar')
('solar', 'system')
('system', 'and')
('and', 'the')
('the', 'only')
('only', 'known')
('known', 'celestial')
('celestial', 'body')
('body', 'to')
('to', 'support')
('support', 'life.')
('life.', 'With')
('With', 'a')
('a', 'diverse')
('diverse', 'range')
('range', 'of')
('of', 'ecosystems,')
('ecosystems,', 'it')
('it', 'is')
('is', 'home')
('home', 'to')
('to', 'a')
('a', 'vast')
('vast', 'array')
('array', 'of')
('of', 'plant')
('plant', 'and')
('and', 'animal')
('animal', 'species,')
('species,', 'including')
('including', 'humans.')

***********   TRIGRAM    ************************
('Earth', 'is', 'the')
('is', 'the', 'third')
('the', 'third', 'planet')
('third', 'planet', 'from')
('planet', 'from', 'the')
('from', 'the', 'Sun')
('the', 'Sun', 'in')
('Sun', 'in', 'our')
('in', 'our', 'solar')
('our', 'solar', 'system')
('solar', 'system', 'and')
('system', 'and', 'the')
('and', 'the', 'only')
('the', 'only', 'known')
('only', 'known', 'celestial')
('known', 'celestial', 'body')
('celestial', 'body', 'to')
('body', 'to', 'support')
('to', 'support', 'life.')
('support', 'life.', 'With')
('life.', 'With', 'a')
('With', 'a', 'diverse')
('a', 'diverse', 'range')
('diverse', 'range', 'of')
('range', 'of', 'ecosystems,')
('of', 'ecosystems,', 'it')
('ecosystems,', 'it', 'is')
('it', 'is', 'home')
('is', 'home', 'to')
('home', 'to', 'a')
('to', 'a', 'vast')
('a', 'vast', 'array')
('vast', 'array', 'of')
('array', 'of', 'plant')
('of', 'plant', 'and')
('plant', 'and', 'animal')
('and', 'animal', 'species,')
('animal', 'species,', 'including')
('species,', 'including', 'humans.')

'''



###  Assignment No 5 ###

"""Assignment Title :  Implement regular expression function to find URL, IP address, Date,
PAN number in textual data using python libraries"""

import spacy
import re

# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")

# Define regular expressions
url_pattern = re.compile(r'https?://\S+|www\.\S+')

ip_address_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')

date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')
pan_number_pattern = re.compile(r'[A-Z]{5}[0-9]{4}[A-Z]')

def extract_entities(text):
    # Tokenize the text using spaCy
    doc = nlp(text)

    # Find entities using regular expressions
    urls = re.findall(url_pattern, text)
    ip_addresses = re.findall(ip_address_pattern, text)
    dates = re.findall(date_pattern, text)
    pan_numbers = re.findall(pan_number_pattern, text)

    # Extract spaCy entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    return {
        'urls': urls,
        'ip_addresses': ip_addresses,
        'dates': dates,
        'pan_numbers': pan_numbers,
        'spaCy_entities': entities
    }

# Example usage
text_data = """
Here is a sample text with a URL: https://www.Sample.com. 
Also, an IP address: 192.168.789.102. 
The date is 2023-01-01.
A PAN number is BBRPL4574H.
"""

results = extract_entities(text_data)

print("URLs:", results['urls'])
print("IP Addresses:", results['ip_addresses'])
print("Dates:", results['dates'])
print("PAN Numbers:", results['pan_numbers'])
print("Entities:", results['spaCy_entities'])

'''
**************   OUTPUT     ****************

URLs: ['https://www.Sample.com.']
IP Addresses: ['192.168.789.102']
Dates: ['2023-01-01']
PAN Numbers: ['BBRPL4574H']
Entities: [('IP', 'ORG'), ('192.168.789.102', 'CARDINAL'), ('2023-01-01', 'DATE'), ('PAN', 'ORG')]

'''



###  Assignment No 6 ###

"""Assignment Title : : Implement and visualize Dependency Parsing of Textual Input
using Stan- ford CoreNLP and Spacy library"""

import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

multiline_text = """
Earth is the third planet from the Sun in our solar system and the only known celestial body to support life. 
With a diverse range of ecosystems, it is home to a vast array of plant and animal species, including humans. 
Earth's atmosphere, composed mainly of nitrogen and oxygen, sustains life by providing the necessary conditions for biological processes to thrive.
"""

multiline_doc = nlp(multiline_text)

for token in multiline_doc:
    print(
        f"""
TOKEN: {token.text}
=====
{token.tag_ = }
{token.head.text = }
{token.dep_ = }"""
    )

displacy.serve(multiline_doc, style="dep")

'''
******************    OUTPUT      *************************

TOKEN:

=====
token.tag_ = '_SP'
token.head.text = 'Earth'
token.dep_ = 'dep'

TOKEN: Earth
=====
token.tag_ = 'NNP'
token.head.text = 'is'
token.dep_ = 'nsubj'

TOKEN: is
=====
token.tag_ = 'VBZ'
token.head.text = 'is'
token.dep_ = 'ROOT'

TOKEN: the
=====
token.tag_ = 'DT'
token.head.text = 'planet'
token.dep_ = 'det'

TOKEN: third
=====
token.tag_ = 'JJ'
token.head.text = 'planet'
token.dep_ = 'amod'

TOKEN: planet
=====
token.tag_ = 'NN'
token.head.text = 'is'
token.dep_ = 'attr'

TOKEN: from
=====
token.tag_ = 'IN'
token.head.text = 'planet'
token.dep_ = 'prep'

TOKEN: the
=====
token.tag_ = 'DT'
token.head.text = 'Sun'
token.dep_ = 'det'

TOKEN: Sun
=====
token.tag_ = 'NNP'
token.head.text = 'from'
token.dep_ = 'pobj'

TOKEN: in
=====
token.tag_ = 'IN'
token.head.text = 'planet'
token.dep_ = 'prep'

TOKEN: our
=====
token.tag_ = 'PRP$'
token.head.text = 'system'
token.dep_ = 'poss'

TOKEN: solar
=====
token.tag_ = 'JJ'
token.head.text = 'system'
token.dep_ = 'amod'

TOKEN: system
=====
token.tag_ = 'NN'
token.head.text = 'in'
token.dep_ = 'pobj'

TOKEN: and
=====
token.tag_ = 'CC'
token.head.text = 'planet'
token.dep_ = 'cc'

TOKEN: the
=====
token.tag_ = 'DT'
token.head.text = 'body'
token.dep_ = 'det'

TOKEN: only
=====
token.tag_ = 'JJ'
token.head.text = 'body'
token.dep_ = 'amod'

TOKEN: known
=====
token.tag_ = 'VBN'
token.head.text = 'body'
token.dep_ = 'amod'

TOKEN: celestial
=====
token.tag_ = 'JJ'
token.head.text = 'body'
token.dep_ = 'amod'

TOKEN: body
=====
token.tag_ = 'NN'
token.head.text = 'planet'
token.dep_ = 'conj'

TOKEN: to
=====
token.tag_ = 'TO'
token.head.text = 'support'
token.dep_ = 'aux'

TOKEN: support
=====
token.tag_ = 'VB'
token.head.text = 'is'
token.dep_ = 'advcl'

TOKEN: life
=====
token.tag_ = 'NN'
token.head.text = 'support'
token.dep_ = 'dobj'

TOKEN: .
=====
token.tag_ = '.'
token.head.text = 'is'
token.dep_ = 'punct'

TOKEN:

=====
token.tag_ = '_SP'
token.head.text = '.'
token.dep_ = 'dep'

TOKEN: With
=====
token.tag_ = 'IN'
token.head.text = 'is'
token.dep_ = 'prep'

TOKEN: a
=====
token.tag_ = 'DT'
token.head.text = 'range'
token.dep_ = 'det'

TOKEN: diverse
=====
token.tag_ = 'JJ'
token.head.text = 'range'
token.dep_ = 'amod'

TOKEN: range
=====
token.tag_ = 'NN'
token.head.text = 'With'
token.dep_ = 'pobj'

TOKEN: of
=====
token.tag_ = 'IN'
token.head.text = 'range'
token.dep_ = 'prep'

TOKEN: ecosystems
=====
token.tag_ = 'NNS'
token.head.text = 'of'
token.dep_ = 'pobj'

TOKEN: ,
=====
token.tag_ = ','
token.head.text = 'is'
token.dep_ = 'punct'

TOKEN: it
=====
token.tag_ = 'PRP'
token.head.text = 'is'
token.dep_ = 'nsubj'

TOKEN: is
=====
token.tag_ = 'VBZ'
token.head.text = 'is'
token.dep_ = 'ROOT'

TOKEN: home
=====
token.tag_ = 'RB'
token.head.text = 'is'
token.dep_ = 'advmod'

TOKEN: to
=====
token.tag_ = 'IN'
token.head.text = 'home'
token.dep_ = 'prep'

TOKEN: a
=====
token.tag_ = 'DT'
token.head.text = 'array'
token.dep_ = 'det'

TOKEN: vast
=====
token.tag_ = 'JJ'
token.head.text = 'array'
token.dep_ = 'amod'

TOKEN: array
=====
token.tag_ = 'NN'
token.head.text = 'to'
token.dep_ = 'pobj'

TOKEN: of
=====
token.tag_ = 'IN'
token.head.text = 'array'
token.dep_ = 'prep'

TOKEN: plant
=====
token.tag_ = 'NN'
token.head.text = 'species'
token.dep_ = 'nmod'

TOKEN: and
=====
token.tag_ = 'CC'
token.head.text = 'plant'
token.dep_ = 'cc'

TOKEN: animal
=====
token.tag_ = 'NN'
token.head.text = 'plant'
token.dep_ = 'conj'

TOKEN: species
=====
token.tag_ = 'NNS'
token.head.text = 'of'
token.dep_ = 'pobj'

TOKEN: ,
=====
token.tag_ = ','
token.head.text = 'species'
token.dep_ = 'punct'

TOKEN: including
=====
token.tag_ = 'VBG'
token.head.text = 'species'
token.dep_ = 'prep'

TOKEN: humans
=====
token.tag_ = 'NNS'
token.head.text = 'including'
token.dep_ = 'pobj'

TOKEN: .
=====
token.tag_ = '.'
token.head.text = 'is'
token.dep_ = 'punct'

TOKEN:

=====
token.tag_ = '_SP'
token.head.text = '.'
token.dep_ = 'dep'

TOKEN: Earth
=====
token.tag_ = 'NNP'
token.head.text = 'atmosphere'
token.dep_ = 'poss'

TOKEN: 's
=====
token.tag_ = 'POS'
token.head.text = 'Earth'
token.dep_ = 'case'

TOKEN: atmosphere
=====
token.tag_ = 'NN'
token.head.text = 'sustains'
token.dep_ = 'nsubj'

TOKEN: ,
=====
token.tag_ = ','
token.head.text = 'atmosphere'
token.dep_ = 'punct'

TOKEN: composed
=====
token.tag_ = 'VBN'
token.head.text = 'atmosphere'
token.dep_ = 'acl'

TOKEN: mainly
=====
token.tag_ = 'RB'
token.head.text = 'of'
token.dep_ = 'advmod'

TOKEN: of
=====
token.tag_ = 'IN'
token.head.text = 'composed'
token.dep_ = 'prep'

TOKEN: nitrogen
=====
token.tag_ = 'NN'
token.head.text = 'of'
token.dep_ = 'pobj'

TOKEN: and
=====
token.tag_ = 'CC'
token.head.text = 'nitrogen'
token.dep_ = 'cc'

TOKEN: oxygen
=====
token.tag_ = 'NN'
token.head.text = 'nitrogen'
token.dep_ = 'conj'

TOKEN: ,
=====
token.tag_ = ','
token.head.text = 'atmosphere'
token.dep_ = 'punct'

TOKEN: sustains
=====
token.tag_ = 'VBZ'
token.head.text = 'sustains'
token.dep_ = 'ROOT'

TOKEN: life
=====
token.tag_ = 'NN'
token.head.text = 'sustains'
token.dep_ = 'dobj'

TOKEN: by
=====
token.tag_ = 'IN'
token.head.text = 'sustains'
token.dep_ = 'prep'

TOKEN: providing
=====
token.tag_ = 'VBG'
token.head.text = 'by'
token.dep_ = 'pcomp'

TOKEN: the
=====
token.tag_ = 'DT'
token.head.text = 'conditions'
token.dep_ = 'det'

TOKEN: necessary
=====
token.tag_ = 'JJ'
token.head.text = 'conditions'
token.dep_ = 'amod'

TOKEN: conditions
=====
token.tag_ = 'NNS'
token.head.text = 'providing'
token.dep_ = 'dobj'

TOKEN: for
=====
token.tag_ = 'IN'
token.head.text = 'conditions'
token.dep_ = 'prep'

TOKEN: biological
=====
token.tag_ = 'JJ'
token.head.text = 'processes'
token.dep_ = 'amod'

TOKEN: processes
=====
token.tag_ = 'NNS'
token.head.text = 'for'
token.dep_ = 'pobj'

TOKEN: to
=====
token.tag_ = 'TO'
token.head.text = 'thrive'
token.dep_ = 'aux'

TOKEN: thrive
=====
token.tag_ = 'VB'
token.head.text = 'conditions'
token.dep_ = 'relcl'

TOKEN: .
=====
token.tag_ = '.'
token.head.text = 'sustains'
token.dep_ = 'punct'

TOKEN:

=====
token.tag_ = '_SP'
token.head.text = '.'
token.dep_ = 'dep'

'''


