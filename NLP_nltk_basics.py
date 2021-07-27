# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 15:21:31 2021

@author: goka
"""

##############################################################################
## Working on Tokenization
##############################################################################
from nltk.tokenize import word_tokenize, sent_tokenize

sample_text = "Hello there, how are you? The weather is nice today. How is your mother-in-law doing?"

tokenized_sentences = sent_tokenize(sample_text)

for sentence in tokenized_sentences:
    print(sentence)
    
tokenized_word = word_tokenize(sample_text)
for word in tokenized_word:
    print(word)
    
## Working on stemming
from nltk.stem import PorterStemmer

ps = PorterStemmer()

sample_words = ["legal", "illegal", "legalize", "legality"]
for word in sample_words:
    print(ps.stem(word))


##############################################################################
## Working on Lemmatization
##############################################################################
from nltk.stem import WordNetLemmatizer
import nltk 

nltk.download("wordnet")
nltk.download()

lemmatizer = WordNetLemmatizer()

sample_words = ["puppies", "celebrities", "dancing", "immortality"]
for word in sample_words:
    print(lemmatizer.lemmatize(word))
    

print(lemmatizer.lemmatize("taller", pos="a"))


##############################################################################
## Working on Part of Speech Tagging
##############################################################################
from nltk import word_tokenize, sent_tokenize
from nltk import pos_tag
nltk.download('averaged_perceptron_tagger')
nltk.download()

sample_text = "I was walking along the road when I saw the cuttest puppies on the other side"
tokenized_words = word_tokenize(sample_text)
tag = pos_tag(tokenized_words)
print(tag)


with open("2001aspaceodyssey.txt", "r") as f:
    sample_text = f.read()

tokenized_sentences = sent_tokenize(sample_text)
tag_words = list()
for sentence in tokenized_sentences:
    tokenized_words = word_tokenize(sentence)
    tag = pos_tag(tokenized_words)
    tag_words.append(tag)
    

##############################################################################
## Working on StopWords
##############################################################################
from nltk.corpus import stopwords

text_1 = "Stopwords are a pretty way to get rid of words we do not want."

stop_words = set(stopwords.words("english"))
print(stop_words)    

tokenize_words_1 = word_tokenize(text_1)
filtered_sent = list()

for word in tokenize_words_1:
    if word not in stop_words:
        filtered_sent.append(word)
print(filtered_sent)

## Removing the stop words in an entire corpus
with open("2001aspaceodyssey.txt", "r") as f:
    sample_text = f.read()

tokenized_sentences = sent_tokenize(sample_text)
sentences_filtered = list()
for sentence in tokenized_sentences:
    tokenized_words = word_tokenize(sentence)

    filtered_sent = list()

    for word in tokenized_words:
        if word not in stop_words:
            filtered_sent.append(word)
    sentences_filtered.extend(filtered_sent)
print(sentences_filtered)


##############################################################################
## Working on Named Entities Recognition
##############################################################################
from nltk import pos_tag
from nltk import word_tokenize
import nltk


sample_text = "The breakthrough in talks came a day after France was again shaken on Saturday by protests against the rules that saw over 160,000 rally and dozens arrested"
tokenized_words = word_tokenize(sample_text)
tag = pos_tag(tokenized_words)

namedEnt = nltk.ne_chunk(tag,)
# namedEnt = nltk.ne_chunk(tag, binary=True)
namedEnt.draw()