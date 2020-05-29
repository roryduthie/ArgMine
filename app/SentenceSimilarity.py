"""
Created on Fri Apr 12 15:10:40 2019

@author: nihitsaxena
"""

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from pywsd.lesk import simple_lesk
import numpy as np

class SentenceSimilarity:

    def __init__(self):
        self.word_order = False


    def identifyWordsForComparison(self, sentence):
        #Taking out Noun and Verb for comparison word based
        tokens = nltk.word_tokenize(sentence)
        pos = nltk.pos_tag(tokens)
        pos = [p for p in pos if p[1].startswith('N') or p[1].startswith('V')]
        return pos

    def wordSenseDisambiguation(self, sentence):
        # removing the disambiguity by getting the context
        pos = self.identifyWordsForComparison(sentence)
        sense = []
        for p in pos:
            sense.append(simple_lesk(sentence, p[0], pos=p[1][0].lower()))
        return set(sense)

    def getSimilarity(self, arr1, arr2, vector_len):
        #cross multilping all domains
        vector = [0.0] * vector_len
        count = 0
        for i,a1 in enumerate(arr1):
            all_similarityIndex=[]
            for a2 in arr2:
                if a1 is not None and a2 is not None:
                    similarity = wn.synset(a1.name()).wup_similarity(wn.synset(a2.name()))
                else:
                    similarity = None
                if similarity != None:
                    all_similarityIndex.append(similarity)
                else:
                    all_similarityIndex.append(0.0)
            all_similarityIndex = sorted(all_similarityIndex, reverse = True)
            if len(all_similarityIndex) < 1:
                vector[i] = 0
            else:
                vector[i]=all_similarityIndex[0]
            if vector[i] >= 0.804:
                count +=1
        return vector, count


    def shortestPathDistance(self, sense1, sense2):
        #getting the shortest path to get the similarity
        if len(sense1) >= len(sense2):
            grt_Sense = len(sense1)
            v1, c1 = self.getSimilarity(sense1, sense2, grt_Sense)
            v2, c2 = self.getSimilarity(sense2, sense1, grt_Sense)
        if len(sense2) > len(sense1):
            grt_Sense = len(sense2)
            v1, c1 = self.getSimilarity(sense2, sense1, grt_Sense)
            v2, c2 = self.getSimilarity(sense1, sense2, grt_Sense)
        return np.array(v1),np.array(v2),c1,c2

    def main(self, sentence1, sentence2):
        sense1 = self.wordSenseDisambiguation(sentence1)
        sense2 = self.wordSenseDisambiguation(sentence2)
        v1,v2,c1,c2 = self.shortestPathDistance(sense1,sense2)
        dot = np.dot(v1,v2)
        #print("dot", dot) # getting the dot product
        tow = (c1+c2)/1.8
        final_similarity = dot/tow
        if tow == 0 or dot == 0:
            return 0
        return final_similarity

    def sentence_similarity(self,sentence1, sentence2):
        """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
        sentence1 = pos_tag(word_tokenize(sentence1))
        sentence2 = pos_tag(word_tokenize(sentence2))

    # Get the synsets for the tagged words
        synsets1 = [self.tagged_to_synset(*tagged_word) for tagged_word in sentence1]
        synsets2 = [self.tagged_to_synset(*tagged_word) for tagged_word in sentence2]

    # Filter out the Nones
        synsets1 = [ss for ss in synsets1 if ss]
        synsets2 = [ss for ss in synsets2 if ss]

        score, count = 0.0, 0

    # For each word in the first sentence
        for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
            vals = [synset.path_similarity(ss) if synset.path_similarity(ss) is not None else 0 for ss in synsets2]
            best_score = max(vals)

        # Check that the similarity could have been computed
            if best_score is not None:
                score += best_score
                count += 1

    # Average the values
        score /= count
        return score

    def penn_to_wn(self,tag):
        """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
        if tag.startswith('N'):
            return 'n'

        if tag.startswith('V'):
            return 'v'

        if tag.startswith('J'):
            return 'a'

        if tag.startswith('R'):
            return 'r'

        return None


    def tagged_to_synset(self,word, tag):
        wn_tag = self.penn_to_wn(tag)
        if wn_tag is None:
            return None

        try:
            return wn.synsets(word, wn_tag)[0]
        except:
            return None

    def symmetric_sentence_similarity(self, sentence1, sentence2):
        """ compute the symmetric sentence similarity using Wordnet """
        return (self.sentence_similarity(sentence1, sentence2) + self.sentence_similarity(sentence2, sentence1)) / 2
