# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 19:47:30 2018

@author: Tiancheng Hu
"""

"""
Every function for this homework assignment is in this file.
This program first read the training and test data from file. Then, it builds 
a sigmoid unit with weights saved in a list. After building this model,
the program moves on to calculate the accuracy.

 
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
this function serves the role of a sigmoid function
"""    
def sigmoid(t):
    return 1/(1+math.exp(-1*t))  
"""
this function performs matrix multiplication(element wise), mat1 and mat2 are
the 2 input matrices while res is the output result. 
"""    
def matmul(mat1,mat2):
    res=0
    for i,j in zip(mat1,mat2):
        res=res+i*int(j)
    return res
"""
this function implements the threshold. "dataset“ is an input dataset. It
can be train data or test data. The function returns the number of 
the correctly classified instances in that dataset
"""  

def evaluate(dataset,weight):
    true=0
    for i in dataset[1:]:
        groundtruth=int(i[-1])
        sample=i[:-1]
        predict=sigmoid(matmul(weight,sample))
        if predict >=0.5:
            predict=1
        else:
            predict=0
        if predict==groundtruth:
            true+=1
    return true      
"""
this is the bulk of this whole program. This function performs the forward 
calculation, back propogation and printing of the training process. "train" is
the train dataset. Lr is the learning rate. numiter is the numer of iterations.

"""  
def training(train,lr,numiter):
    weightnum=len(train[0])-1
    weight=[0]*weightnum
#    weightnum=13
    pointer=1
    attrname=train[0][:-1]
    for i in range(numiter):
        sample=train[pointer][:-1]
        groundtruth=train[pointer][-1]
        prediction=matmul(weight,sample)
        loss=int(groundtruth)-sigmoid(prediction)
        for j in range(weightnum):
            weight[j]=weight[j]+lr*(int(sample[j])*loss*sigmoid(prediction)*(1-sigmoid(prediction)))
#                print('iteration',i, file=f)
#                print('weight',j,'=weight', j, '+',lr,'*', int(sample[j]),'*',loss,'*',sigmoid(prediction), '*', (1-sigmoid(prediction)),file=f)
        pointer+=1
        if pointer==len(train):
            pointer=1
        print("After iteration ",i+1,": ",end ="",sep='')
        for j,k in enumerate(attrname):
            print("w(",k,") = ",("%.4f" % round(weight[j],4)), ", ",end ="",sep='')
        newpredict=sigmoid(matmul(weight,sample))
        print("output = ",("%.4f" % round(newpredict,4)),end="",sep='')
        print("",sep='')
    return weight
"""
This is the function that prints out the evaluation of the model. 

"""  
def printaccu(weight,train,test):
#    with open('outset2lr0.3+2500.txt', 'a') as f:
    print("")
    truetrain=evaluate(train,weight)
    acctrain=truetrain/(len(train)-1)
    print("Accuracy on training set (",len(train)-1, " instances): ",("%.2f" % round(acctrain*100,2)),"%",sep='')
    print("",sep='')
    truetest=evaluate(test,weight)
    acctest=truetest/(len(test)-1)
    print("Accuracy on test set (",len(test)-1, " instances): ",("%.2f" % round(acctest*100,2)),"%",sep='')
    #print("",sep='')
"""
The main function puts everything together.

""" 

if __name__ == "__main__":
    a = (sys.argv[1])
    b = (sys.argv[2])
    c = float(sys.argv[3])      
    d = int(sys.argv[4])
    train,test=readfile(a,b)
    weight=training(train,c,d)
    printaccu(weight,train,test)
