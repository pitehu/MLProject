# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 14:30:25 2018

@author: Tiancheng Hu
"""

import csv
import sys
import math


"""
Every function for this homework assignment is in this file.
This program first read the training from file. Then, it loads the rewards into
a dictionary called reward. It also loads all the probabilities into a
dictionary file called prob. For example, prob['s1']['a1']['s1'] stores the
probability of taking action  a1 at state s1 and stays at a1. Then, using value
iteration, I build a dictionary structure called table that stores all the J
values, or expected discounted sum of rewards over the next 1-20 time step in 
this case. At the same time, I store the best action at each state into another
dictionary file called bestact.
 
"""


"""
this function read the training data from file and parse them into a list 
for later processing
"""
def readfile(trainaddr):
    traindata=[]
    with open(trainaddr) as f1:
        csv_f1=csv.reader(f1)
        for row in csv_f1:
            temp=row[0].replace('(',' ').replace(')',' ').split()
            traindata.append(temp)  
    return(traindata)

"""
this function populates the dictionaries that store the rewards and 
probabilities. The input of this function is the the output of the previous
function, a list, and the output is two dictionaries.

"""

def processdata(traindata):
#    reward=[0]*len(train)
#    prob=[0,0,0]*len(train)
    reward={}
    prob={}
    for i in range(len(traindata)):
        snumber=traindata[i][0]
#        print(snumber)
        rew=traindata[i][1]
#        print(rew)
        reward[snumber]=rew
        prob[snumber]={}
        for j in range(2,len(traindata[i])-2,3):
#            print(j)
            act=traindata[i][j]
#            print('act',act)
            moveto=traindata[i][j+1]
#            print('moveto',moveto)
            withaprobof=traindata[i][j+2]
#            print('withaprobof',withaprobof)
            if not(act in prob[snumber]):
                prob[snumber][act]={}
            prob[snumber][act][moveto]=withaprobof
            
    return reward,prob
"""
this function builds the dictionaries that store the J values along with 
another dictionary that stores the best action at any given state.
The input of this function is the the outputs of the previous
function, the reward and prob dictionary and the discount factor gamma.

"""


def buildtable(reward,prob,gamma):
    table={}
    table['1']={}
    bestact={}
    bestact['1']={}
    for k,v in reward.items():
        table['1'][k]=float(v)
        bestact['1'][k]='a1'      
    for j in range(2,21):
        bestact[str(j)]={}
        table[str(j)]={}
        for state in list(prob.keys()):
            maxvalue=-math.inf
            rew=float(reward[state])
            for action in list(prob[state].keys()):
                temp=rew
                for destinestate in list(prob[state][action].keys()):
#                    print('state',state)
#                    print('action',action)
#                    print('des',destinestate)
                    temp+=gamma*float(prob[state][action][destinestate])*(table[str(j-1)][destinestate])
#                    print(temp)
                if temp>maxvalue:
                    maxvalue=temp
                    bestact[str(j)][state]=action
#                    print("state",state,"action",action,"maxvalue",maxvalue)
            table[str(j)][state]=maxvalue
                    
    return table,bestact

"""
this function prints out the outputs of this whole program in the required 
format. All inputs are dictionaries from the previous functions.
"""

def printres(prob,table,bestact):
    for i in range(1,21):
        sys.stdout.write('After iteration '+str(i)+': ')
        for state in (list(prob.keys())):
            value=round(table[str(i)][state],4)
            sys.stdout.write('('+state+' '+bestact[str(i)][state]+' '+str("%.4f" % value)+') ')
        if not i==(20):
            sys.stdout.write('\n')
"""
this part serves as the main() of this entire program. It calls the appropriate
functions.
"""                     
if __name__ == "__main__":
    a = (sys.argv[1])
    gamma = float((sys.argv[2]))
    train=readfile(a)
    reward,prob=processdata(train)
    table,bestact=buildtable(reward,prob,gamma)
    printres(prob,table,bestact)
        
        