# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 10:16:58 2018

@author: Tiancheng Hu

"""

"""
Every function for this homework assignment is in this file.
This program first read the training and test data set file. Then, it builds 
the naive bayes classifier in a dictionary. For example, record['1']['wesley']
['1'] stores the probability of P(wesley=1|1). After building this model,
the function moves on to calculate the accuracy and print out the model.

 
"""
import csv
import sys
import math


"""
this function read the training data as well as the test data from file
trainaddr is the address of the training data and testaddr is the address of
the test data. It returns the training and test data read from files.
"""
def readfile(trainaddr,testaddr):
    traindata=[]
    testdata=[]
    with open(trainaddr) as f1:
        csv_f1=csv.reader(f1)
        for row in csv_f1:
            temp=row[0].split()
            traindata.append(temp)  
    with open(testaddr) as f2:
        csv_f2=csv.reader(f2)
        for row2 in csv_f2:
            temp2=row2[0].split()
            testdata.append(temp2) 
    return(traindata,testdata)
"""
this function counts the numer of 1s and 0s of an attribute attr among the dataset 
train. It returns the number of samples when attr=1 and 0
"""
def count(train,attr):
    countzero=0
    countone=0
    for row in train[1:]:
        if row[train[0].index(attr)]==str(1):
            countone+=1
        else:
            countzero+=1
    return countone,countzero   
"""
this function actually builds the naive bayes model given the input list of 
training data set train. It returns record, a dictionary that stored the model,
posprob, P(class=1) and negprob, P(class=0).

"""
         
def naivebayes(train):
    record={}
    positive=[r for r in train[1:] if r[-1]==str(1)]
    possize=len(positive)
    posprob=possize/(len(train)-1)
    positive.insert(0,train[0])
    negative=[r for r in train[1:] if r[-1]==str(0)]
    negsize=len(negative)
    negprob=negsize/(len(train)-1)
    negative.insert(0,train[0])
    record['1']={}
    record['0']={}
    for i in positive[0][:-1]:
        record['1'][i]={}
        countone,countzero=count(positive,i)
        record['1'][i]['1']=countone/possize
        record['1'][i]['0']=countzero/possize
        record['0'][i]={}
        countonen,countzeron=count(negative,i)
        record['0'][i]['1']=countonen/negsize
        record['0'][i]['0']=countzeron/negsize
        
    return record,posprob,negprob
"""
this function returns the prediction given a model that is already trained. 
Notice that list1 should be a list of which the first line is the name of the 
attributes and the second line is the value of those attributes. In other
words, it should be similar to train[0:2]

"""

def inference(list1,record,posprob,negprob):
    class0=math.log10(negprob)
    class1=math.log10(posprob)
    for index,name in enumerate(list1[0][:-1]):
        class0=class0+math.log10(record['0'][name][str(list1[1][index])])
        class1=class1+math.log10(record['1'][name][str(list1[1][index])])
    if class0>class1:
        return '0'
    else:
        return '1'
"""
this function returns the accuracy of a trained model on a given dataset list1
"""    
    
def evaluate(list1,record,posprob,negprob):
    name=list1[0]
    test=[]
    test.append(name)
    count=0
    for row in list1[1:]:
        test=[]
        test.append(name)
        test.append(row)
        res=inference(test,record,posprob,negprob)
        if res==row[-1]:
            count+=1
    return count/(len(list1)-1)
"""
this function prints out the model which is stored in a dictionary to the 
format required
"""            

def printmodel(train,record,posprob,negprob):
    sys.stdout.write('P('+train[0][-1]+'=0)='+str(round(negprob,2))+' ')
    for i in train[0][:-1]:
        sys.stdout.write('P('+i+'=0'+'|0)='+str("%.2f" % round(record['0'][i]['0'],2))+' ')
        sys.stdout.write('P('+i+'=1'+'|0)='+str("%.2f" % round(record['0'][i]['1'],2))+' ')
    sys.stdout.write('\n')
    sys.stdout.write('P('+train[0][-1]+'=1)='+str(round(posprob,2))+' ')
    for i in train[0][:-1]:
        sys.stdout.write('P('+i+'=0'+'|1)='+str("%.2f" % round(record['1'][i]['0'],2))+' ')
        sys.stdout.write('P('+i+'=1'+'|1)='+str("%.2f" % round(record['1'][i]['1'],2))+' ')
    sys.stdout.write('\n')

if __name__ == "__main__":
    a = (sys.argv[1])
    b = (sys.argv[2])
    train,test=readfile(a,b)
    record,posprob,negprob=naivebayes(train)
    printmodel(train,record,posprob,negprob)
    sys.stdout.write('\n')
    trainaccr=evaluate(train,record,posprob,negprob)
    testaccr=evaluate(test,record,posprob,negprob)
    sys.stdout.write('Accuracy on training set (')
    sys.stdout.write(str(len(train)-1)+' instances): '+str("%.2f" % round(trainaccr*100,2))+'%')
    sys.stdout.write('\n')
    sys.stdout.write('\n')
    sys.stdout.write('Accuracy on test set (')
    sys.stdout.write(str(len(test)-1)+' instances): '+str("%.2f" % round(testaccr*100,2))+'%')
    
    
    

    
