# models.py

import time
import numpy as np
from utils import *
from collections import Counter
from nerdata import *
from optimizers import *
from typing import List
from constants import id_labels, label_idx, label_vectors
import pandas as pd
import ast

class CountBasedClassifier(object):
    """
    Classifier that takes counts of how often a word was observed with different labels.
    Unknown tokens or ties default to O label, and then person, location, organization and then MISC.
    Attributes:
        per_counts: how often each token occurred with the label PERSON in training
        loc_counts: how often each token occurred with the label LOC in training
        org_counts: how often each token occurred with the label ORG in training
        misc_counts: how often each token occurred with the label MISC in training
        null_counts: how often each token occurred with the label O in training
    """
    def __init__(self, 
            per_counts: Counter, 
            loc_counts: Counter,
            org_counts: Counter,
            misc_counts: Counter,
            null_counts: Counter
            ):
        self.per_counts = per_counts
        self.loc_counts = loc_counts
        self.org_counts = org_counts
        self.misc_counts = misc_counts
        self.null_counts = null_counts

    def predict(self, tokens: List[str], idx: int):
        per_count = self.per_counts[tokens[idx]]
        loc_count = self.loc_counts[tokens[idx]]
        org_count = self.org_counts[tokens[idx]]
        misc_count = self.misc_counts[tokens[idx]]
        null_count = self.null_counts[tokens[idx]]
        max_count = max(per_count, loc_count, org_count, misc_count, null_count)
        if null_count == max_count:
            return 'O'
        elif per_count == max_count:
            return 'PER'
        elif loc_count == max_count:
            return 'LOC'
        elif org_count == max_count:
            return 'ORG'
        elif misc_count == max_count:
            return 'MISC'
        else:
            print('ERROR?')    
        return 'O'

def train_count_based_classifier(ner_exs: List[NERExample]) -> CountBasedClassifier:
    """
    :param ner_exs: training examples to build the count-based classifier from
    :param ner_exs: training examples to build the count-based classifier from
    :return: A CountBasedClassifier using counts collected from the given examples
    """
    per_counts = Counter()
    loc_counts = Counter()
    org_counts = Counter()
    misc_counts = Counter()
    null_counts = Counter()
    for ex in ner_exs:
        for idx in range(0, len(ex)):
            if ex.labels[idx] == 'PER':
                per_counts[ex.tokens[idx]] += 1.0
            elif ex.labels[idx] == 'LOC':
                loc_counts[ex.tokens[idx]] += 1.0
            elif ex.labels[idx] == 'ORG':
                org_counts[ex.tokens[idx]] += 1.0
            elif ex.labels[idx] == 'MISC':
                misc_counts[ex.tokens[idx]] += 1.0
            else:
                null_counts[ex.tokens[idx]] += 1.0
    return CountBasedClassifier(per_counts, loc_counts, org_counts, misc_counts, null_counts)



class NERClassifier(object):
    """
    Classifier to classify a token in a sentence as a PERSON token or not.
    Constructor arguments are merely suggestions; you're free to change these.
    """

    def __init__(self, weights: np.ndarray, indexer: Indexer):
        self.weights = weights
        self.indexer = indexer

    def predict(self, tokens: List[str], idx: int):
        """
        Makes a prediction for token at position idx in the given NERExample
        :param tokens:
        :param idx:
        :return: 0 if not a person token, 1 if a person token
        """
        graddict={};
       file = open("FeatureSet.txt", "r")
       contents = file.read()
       dictionary = ast.literal_eval(contents)
       file.close()
       IndxList=[];
       file = open("Label.txt", "r")
       labls = file.read();
       dict_labels = ast.literal_eval(labls);
       file.close();
        for ner_exssentence in ner_exs:
            ner_exstoken=ner_exssentence.split();
            for idx in range(0, len(ner_exstoken)):
                if ner_exstoken[idx] in dictionary:
                    IndxList.append(dictionary[ner_exstoken[idx]]);
                else :
                    IndxList.append(dictionary["unknown"]);

            for Indx in IndxList:
                val = dict_labels[Indx];
                person=0;
                org=0;
                loc=0;
                mis=0;
                othr=0;
            
                if val=="PER":
                    person=1;
                elif val=="ORG":
                    org=1;
                elif val=="LOC":
                    loc=1;
                elif val=="MIS":
                    mis=1;
                elif val=="OTH":
                    othr=1;
                
                list1=[person,org,loc,mis,othr];  
                labelVector = np.array(list1);
                print(labelVector);
                WeightMatrix = np.zeros((len(dictionary), 5));
                print(WeightMatrix);
                for j in range(0,5):
                    sum=sum+WeightMatrix[Indx][j];
                score=sum;
              # softmax function    
                f_x = np.exp(score) / np.sum(np.exp(score));
        raise Exception("Implement me!")


def train_classifier(ner_exs: List[NERExample]) -> NERClassifier:
    # Extracting Feature
   optm=UnregularizedAdagradTrainer(WeightMatrix);
   graddict={};
   file = open("FeatureSet.txt", "r")
   contents = file.read()
   dictionary = ast.literal_eval(contents)
   file.close()
   IndxList=[];
   file = open("Label.txt", "r")
   labls = file.read();
   dict_labels = ast.literal_eval(labls);
   file.close();

    for epoches in range(0.50):
        for ner_exssentence in ner_exs:
            ner_exstoken=ner_exssentence.split();
            for idx in range(0, len(ner_exstoken)):
                if ner_exstoken[idx] in dictionary:
                    IndxList.append(dictionary[ner_exstoken[idx]]);
                else :
                    IndxList.append(dictionary["unknown"]);

            for Indx in IndxList:
                val = dict_labels[Indx];
                person=0;
                org=0;
                loc=0;
                mis=0;
                othr=0;
            
                if val=="PER":
                    person=1;
                elif val=="ORG":
                    org=1;
                elif val=="LOC":
                    loc=1;
                elif val=="MIS":
                    mis=1;
                elif val=="OTH":
                    othr=1;
                
                list1=[person,org,loc,mis,othr];  
                labelVector = np.array(list1);
                print(labelVector);
                WeightMatrix = np.zeros((len(dictionary), 5));
                print(WeightMatrix);
                for j in range(0,5):
                    sum=sum+WeightMatrix[Indx][j];
                score=sum;
              # softmax function    
                f_x = np.exp(score) / np.sum(np.exp(score));
            
                
                '''for i in range(5):
                    if list1[i]==1:
                        colindx=i;
                        break;'''
                        
              #  listvctor=[WeightMatrix[Indx][0],WeightMatrix[Indx][1],WeightMatrix[Indx][2],WeightMatrix[Indx][3],WeightMatrix[Indx][4]];    
               # sumweightvector=np.array(listvctor);
                #print(sumweightvector);
                
                gradRes = np.subtract(labelVector,f_x);
                graddict[Indx]=gradRes;         
                loss= np.log(f_x);                              
    
    optm.apply_gradient_update((graddict), batch_size);
            

            
    raise Exception("Implement me!")
