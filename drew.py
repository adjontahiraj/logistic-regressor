import numpy as np
import random
import math
from math import exp

def initModel(k):
    return np.array([[random.gauss(0, .5)] for x in range(k)])

def addIntercept(x):
    intercept = np.ones((x.shape[0], 1))
    return np.concatenate(( x,intercept), axis=1)

def higherProb(prob1,prob2,prob3):
    largest=max(prob1,prob2,prob3)
    if largest==prob1:
        return 0
    elif largest==prob2: 
        return 1
    else: 
        return 2

def predict(x, models):
    x=addIntercept(x)
    predicts = np.zeros(len(x))
    for i in range(len(x)):
        probs = []
        for model in models:
            probs.append((predictProb(x[i],model)))
        probInd = higherProb(probs[0][0],probs[1][0],probs[2][0])
        #print((probs[0][0],probs[1][0],probs[2][0]))
        if(probInd==2):
            predicts[i]=2
        elif(probInd==1):
            predicts[i]=1
        else:
            predicts[i]=0
    #print(predicts)
    return predicts

def sigmoid(z):
    for i in z:
        if(i == 1.0):
            i = 0.99999
        
    return 1 / (1 + np.exp(-z))

def getLoss(h, y,lam,theta):
    e = .00001
    part1=np.dot(y,np.log(h+e).T)
    part2=np.dot((1-y),np.log(1-h+e).T)
    part3=(part1+part2).mean()
    return part3 + (lam*(theta**2)).mean()/2


def fit(x,y,alpha,lam,nepochs,epsilon,modelNo,param):
    x=addIntercept(x)
    #theta = np.zeros(x.shape[1])
    theta=initModel(len(x[0]))
    if(param!=None):
        theta=np.reshape(np.array(param),(len(np.array(param)),1))-theta
    newY=[]
    count,others=0,0
    for i in range(len(y)):
        if(modelNo == 0):
            if(y[i][0] == 0):
                newY.append([.99])
                count+=1
            else:
                newY.append([0.001])
                others+=1
        elif (modelNo == 1):
            if(y[i][0] == 1):
                newY.append([.99])
                count+=1
            else:
                newY.append([0.001])
                others+=1
        else:
            if(y[i][0] == 2):
                newY.append([.99])
                count+=1
            else:
                newY.append([0.001])
                others+=1
    newY=np.array(newY)
    if(modelNo==0):
        scale = 100/(count/(count+others))
    elif(modelNo==1):
        scale = .7/(count/(count+others))
    else:
        scale = 10/(count/(count+others))
    for epoch in range(nepochs):
        # epochLoss=0.0
        # prevLoss=0
        # for row in range(len(x)):
        #     h = predictProb(x[row],theta)
        #     #gradient = (x[row].T* (h - y[row])) #/ len(y)
        #     #theta = theta - (alpha * gradient)
        #     if(y[row]==0):
        #         loss=prevLoss
        #     else:
        #         loss = getLoss(h, y[row])
        #         print(loss)
        #         prevLoss=loss
        #     #print(loss)
        #     epochLoss+=loss
        #     for i in range(len(theta)):
        #         theta[i] += alpha*(loss*x[row][i] - lam*theta[i])
        h = predictProb(x,theta)
        loss=getLoss(h,newY,lam,theta)
        if(abs(loss)>epsilon):
            break
        gradient=((h - newY)).mean() + (lam*theta)/len(newY)
        diag=np.ones(len(x[0]))
        diag[0]=0
        diag=np.diag(diag)
        part1=np.dot(h.T,(1-h)).mean()
        part2=np.dot(x.T,x)
        hessian=np.dot(part1,part2) + (lam/len(newY))*diag
        theta = theta - scale*alpha*np.dot(gradient.T,hessian).T
        #theta = theta - scale*alpha*loss
        
        # for i in range(len(theta)):
            
        #     theta[i] += alpha*(loss*x[row][i] - lam*theta[i])
    return theta,abs(loss)

def predictProb(x, model):
    return sigmoid(np.dot(x, model))


def train(x,y,alpha,lam,nepochs,epsilon,param=None):
    finalAlpha = 0
    finalLam = 0
    finalModels = []    #Features and a constant
    lrStep=(alpha[1]-alpha[0])/20
    lStep=(lam[1]-lam[0])/20
    alphas=[0]*3
    lams=[0]*3
    y = y.reshape(len(y),1)
    for modelNo in range(3):
        bestModel = []    #Features and a constant
        bestLoss = 99999999
        for learnRate in np.arange(alpha[0],alpha[1],lrStep):
            for regRate in np.arange(lam[0],lam[1],lStep):
                model,loss = fit(x,y,learnRate,regRate,nepochs,epsilon,modelNo,param)
                if(np.log(loss) < np.log(bestLoss)):
                    bestLoss = loss
                    bestModel = model
                    alphas[modelNo]=learnRate
                    lams[modelNo]=regRate
        finalModels.append(bestModel)
    finalAlpha=(np.array(alphas)).mean()
    finalLam=(np.array(lams)).mean()
    #print(finalModels)
    return finalModels,finalAlpha,finalLam

def validate(x,y,alpha,lam,nepochs,epsilon,param):
    lossSum=0.0
    y = y.reshape(len(y),1)
    predicts = predict(x,param)
    for i in range(len(x)):
        lossSum+=(y[i][0]-predicts[i])**2
    return lossSum/len(predicts)

def test(x,y,alpha,lam,nepochs,epsilon,param):
    predicts = predict(x,param)
    return np.array(predicts)

def SGDSolver(phase, x, y, alpha,lam,nepochs,epsilon,param):
    if(phase == 'Training'):
        print('In Training Phase')
        if(len(param)==0):
            param=None
        finalParam,finalAlpha,finalLam = train(np.array(x),np.array(y),alpha,lam,nepochs,epsilon,param)
        return finalParam,finalAlpha,finalLam
    elif(phase == 'Validation'):
        print('In Validation Phase')
        mse = validate(np.array(x),np.array(y),alpha,lam,nepochs,epsilon,param)
        return mse
    elif(phase == 'Testing'):
        print('In Testing Phase')
        predictions = test(np.array(x),np.array(y),alpha,lam,nepochs,epsilon,param)
        return predictions
    else:
        print("Invalid phase")
        