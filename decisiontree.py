# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 19:03:54 2018

@author: Tiancheng Hu

Every function of this homework is in this file.
This program takes 3 arguments: the location of training set, the location of 
test set and the number of training set to be used. It then builds a decision 
tree based on the ID3 algorithm. Afterwards, it is evaluated on the training 
set used as well as the test set. This program uses nested dictionary as the 
building block for the tree. The name of the parent node is stored as a key in
the dictionary while the value of this key is the children. When the value of a
key is a number(stored as a string) instead of another dictionary, we reach the 
leaf node. The value corresponding to the key means the prediction in this 
specific case. We assume that the branch corresponding to a "0" in the feature
always goes before the branch corresponding to a "1"
An example is the follwing: 
{'nigeria': {'0': {'learning': {'0': '0',
    '1': {'viagra': {'0': '1', '1': '0'}}}},
  '1': {'viagra': {'0': '1', '1': '0'}}}}

The tree first split on 'nigeria'. Then, the left branch(0 branch) split on 
"learning" and so on. One example of a leaf node is nigeria(0)-learning(0).




"""
import csv
import math
import sys

"""
this function read the training data as well as the test data from file
"""
def readfile(trainaddr,testaddr):
    traindata=[]
    testdata=[]
    with open(trainaddr) as f1:
        csv_f1=csv.reader(f1, delimiter='\t')
        for row in csv_f1:
            traindata.append(row)  
    with open(testaddr) as f2:
        csv_f2=csv.reader(f2, delimiter='\t')
        for row2 in csv_f2:
            testdata.append(row2) 
    return(traindata,testdata)

"""
this function calculate the entropy given a list of samples
"""

def calculate_entropy(list1):
    if not list1:
        return 0
    ones=0
    zeros=0
    for row in list1:
        if int(row[-1])==1:
            ones+=1
        if int(row[-1])==0:
            zeros+=1
    p1=ones/(ones+zeros)
    p2=zeros/(ones+zeros)
    if (p1==0) or (p2==0):
        entropy=0
    else:
        entropy=-(p1*math.log2(p1))-(p2*math.log2(p2))
    return entropy

"""
This function calculate the best information gain given a list of samples and 
return two lists after the split, along with the splitting feature, and a 
binary value. If this value is 0, it means the split will result in no 
information gain. If it is 1, it means a split should be done. Notice that a 
global variable available_feature is used to keep track of what features have 
been used

"""

def bestinfogain(list1):
    global available_feature
    maxig=0
    oldentropy=calculate_entropy(list1[1:])
    splitfeature=list1[0][0]
    feature1_true_best=[]
    feature1_true_best.append(list1[0])
    feature1_false_best=[]
    feature1_false_best.append(list1[0])
    for i in range(len(list1[0])-1):
        feature1=list1[0][i]
        feature1_true=[]
        feature1_false=[]
        count_true=0
        count_false=0
        feature1_true.append(list1[0])
        feature1_false.append(list1[0])
        
        for row in list1[1:]:

            if (int(row[i])==1):
                feature1_true.append(row)
                count_true+=1
            else:
                feature1_false.append(row)
                count_false+=1

        if (feature1_true==[list1[0]]) or (feature1_false==[list1[0]]):
            continue
        
        entropy_true=calculate_entropy(feature1_true[1:])
        entropy_false=calculate_entropy(feature1_false[1:])
       
        ptrue=count_true/(count_true+count_false)
        pfalse=count_false/(count_true+count_false)

        ig=oldentropy-(ptrue*entropy_true+entropy_false*pfalse)
        if ig>maxig:
            maxig=ig
            splitfeature=feature1
            feature1_true_best=feature1_true[:]
            feature1_false_best=feature1_false[:]
    
    if (maxig==0):
        feature1=available_feature[0]
        featureindex=list1[0].index(feature1)
        feature1_true=[]
        feature1_false=[]
        feature1_true.append(list1[0])
        feature1_false.append(list1[0])
        for row in list1[1:]:

            if (int(row[featureindex])==1):
                feature1_true.append(row)
            else:
                feature1_false.append(row)
        return (feature1_true,feature1_false,feature1,0)
    else:
        
        return (feature1_true_best,feature1_false_best,splitfeature,1)
    
"""
This function determines if a given list is pure
"""
    
def checkifpure(list1):
    ones=0
    zeros=0
    for row in list1[1:]:
        if (int(row[-1])==1):
            ones+=1
        elif (int(row[-1])==0):
            zeros+=1
    if (ones==0) or (zeros==0):
        return 1
    else:
        return 0
"""
This is the function that actually build the tree into the dictionary. It
selects the feature with the highest information gain. If no feature gives out
>0 information gain then the tree will be built based on the remaining features
in the order they are in the first row of the training data.
It is written based on a number of cases, e.g: whether a majority vote is 
needed etc.
"""
       
    
def buildtree(list1,tree1=None):
        global available_feature
        global mostcommonclass
        feature1_true,feature1_false,splitfeature,worthsplit=bestinfogain(list1)

        if (tree1 is None):
            tree1={}
            tree1[splitfeature]={}
        if (worthsplit==0) and ((len(available_feature))<=1):
            tree1[splitfeature][str(0)]=str(most_common_class(list1))
            tree1[splitfeature][str(1)]=str(most_common_class(list1))
            return tree1
        elif (worthsplit==0) and ((len(available_feature))>1):
            available_feature.remove(splitfeature)
            tree1[splitfeature][str(0)]=buildtree(feature1_false)
            tree1[splitfeature][str(1)]=buildtree(feature1_true)
            available_feature.insert(0,splitfeature)
            return tree1
        elif (checkifpure(feature1_false)):
            tree1[splitfeature][str(0)]=str(feature1_false[-1][-1])
        elif ((checkifpure(feature1_false))==False) and (checkcannotsplit(feature1_false)):   
            countzero=0
            countone=0
            
            for row in feature1_false[1:]:
                if (int(row[-1])==0):
                    countzero+=1
                elif (int(row[-1])==1):
                    countone+=1

            if countzero>countone:
                tree1[splitfeature][str(0)]=str(0)
            elif countone>countzero:
                tree1[splitfeature][str(0)]=str(1)
            else:
                tree1[splitfeature][str(0)]=str(mostcommonclass)

        else: 
            available_feature.remove(splitfeature)
            tree1[splitfeature][str(0)]=buildtree(feature1_false)
            available_feature.insert(0,splitfeature)
             
        if (checkifpure(feature1_true)):
            tree1[splitfeature][str(1)]=str(feature1_true[-1][-1])
        elif ((checkifpure(feature1_true))==False) and (checkcannotsplit(feature1_true)): 
            countzero=0
            countone=0
            for row in feature1_true[1:]:
                if (int(row[-1])==0):
                    countzero+=1
                elif (int(row[-1])==1):
                    countone+=1
            if countzero>countone:
                tree1[splitfeature][str(1)]=str(0)
            elif countone>countzero:
                tree1[splitfeature][str(1)]=str(1)
            else:
                tree1[splitfeature][str(1)]=str(mostcommonclass)

        else:
            available_feature.remove(splitfeature)
            tree1[splitfeature][str(1)]=buildtree(feature1_true)
            available_feature.insert(0,splitfeature)
        return tree1
    
"""
This funciton returns the majority class of a list of dataset
"""

def most_common_class(list1):
    ones=0
    zeros=0
    for row in list1[1:]:
        if (int(row[-1])==1):
            ones+=1
        else:
            zeros+=1

    if ones>zeros:
        return 1
    else:
        return 0
    
"""
This funciton determines whether this list cannot be split
"""
def checkcannotsplit(tree1):
    flag1=True
    for i in range(1,len(tree1[0:]),2):
        if (i==(len(tree1[0:])-1)):
            if (tree1[i][:-1]!=tree1[i-1][:-1]):
                flag1=False
            return (flag1)
        if (tree1[i][:-1]!=tree1[i+1][:-1]):
            flag1=False
    return (flag1)
"""
This funciton prints out the tree from the dictionary data structure into
the format asked in the prompt using recursion
"""
def printtree(tree1,depth=0):
    attributename=list(tree1.keys())[0]
    attribute=tree1[list(tree1.keys())[0]]
    leftchild=attribute[list(attribute.keys())[0]]
    rightchild=attribute[list(attribute.keys())[1]]
    if isinstance(leftchild,str):
        for i in range(depth):
            sys.stdout.write("| ")
        sys.stdout.write(attributename)
        sys.stdout.write(' = 0 :  ')
        sys.stdout.write(leftchild)
        sys.stdout.write('\n')

        #
    else:
        for i in range(depth):
            sys.stdout.write("| ")
        depth+=1
        sys.stdout.write(attributename)
        sys.stdout.write(' = 0 :')
        sys.stdout.write('\n')
        printtree(leftchild,depth)
        depth-=1
    if isinstance(rightchild,str):
        for i in range(depth):
            sys.stdout.write("| ")
        sys.stdout.write(attributename)
        sys.stdout.write(' = 1 :  ')
        sys.stdout.write(rightchild)
        sys.stdout.write('\n')
    else: 
        for i in range(depth):
            sys.stdout.write("| ")
        depth+=1
        sys.stdout.write(attributename)
        sys.stdout.write(' = 1 :')
        sys.stdout.write('\n')
        printtree(rightchild,depth)
        depth-=1
"""
This funciton is used to infer the class after the tree is built.
"""
def inference(tree1,list1):
    attnamelist=list1[0]   
    attributename=list(tree1.keys())[0]
    attval=list1[1][attnamelist.index(attributename)]
    nextlevel=tree1[attributename][attval]
    if isinstance(nextlevel,str):
        return nextlevel
    else:
        return (inference(nextlevel,list1))
"""
This funciton is used to test the tree. It uses the inference function. 
"""
def testtree(tree1,list1):
    correct=0
    total=len(list1)-1
    for row in list1[1:]:    
        list2=[list1[0],row]
        res=inference(tree1,list2)
        if (res==row[-1]):
            correct+=1
    return (correct/total)

  
"""
This funciton serves as the main function. It does some necessary calculations
and calls the appropriate functions.
"""                       
   
if __name__ == "__main__":
    a = (sys.argv[1])
    b = (sys.argv[2])
    c = int(sys.argv[3])
    train,test=readfile(a,b)
    mostcommonclass=most_common_class(train[:c+1])
    available_feature=train[0][0:-1]
    tree1=buildtree(train[:c+1])
    printtree(tree1)
    trainres=testtree(tree1,train[:c+1])
    sys.stdout.write('\n')
    sys.stdout.write('Accuracy on training set (')
    sys.stdout.write(str(c)+' instances):  '+('%.2f' % (trainres*100))+'%')
    sys.stdout.write('\n')
    sys.stdout.write('\n')
    testres=testtree(tree1,test)
    sys.stdout.write('Accuracy on test set (')
    sys.stdout.write(str(len(test)-1)+' instances):  '+('%.2f' % (testres*100))+'%')
    
