import numpy as np
import math
import random

# def load_data(filename):
#     """
#     arg: filename - filename of file you want to load data from
#             e.g. red_train.npy 
#     Return: x values (numpy array: n x n)
#             y values (numpy array: n x 1)
#     """
#     data = np.load(filename)
#     #print(data)
#     data_x = np.zeros((len(data), 11))
#     data_y = np.zeros(len(data))
#     for i in range(len(data)):
#         data_x[i] = data[i][0:11]
#         data_y[i] = data[i][-1]
#         # if(data[i][-1] <= 4):
#         #     data_y[i] = 0
#         # elif(data[i][-1] <= 6):
#         #     data_y[i] = 1
#         # else:
#         #     data_y[i] = 2
#     return data_x, data_y

def initial_values():
    #out = [float(random.gauss(0,0.015)) for x in range(11)]
    #out.append(0)
    out = [0,0,0,0,0,0,0,0,0,0,0,0]
    return(np.array(out))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#function to compute predicted output with given model
def compute_y_hat(model, feature_set):
    #setting it to b
    out = model[-1]
    for i in range(11):
        out += (model[i] * feature_set[i])
    out = sigmoid(out)
    #print("\nPrediction: ")
    #print(out)
    return out 

#function to compute loss function with given prediction, actual output, lambda, and current model weights.
def compute_loss(y_hat, y, lam, model):
    if(y_hat == 1):
        y_hat = 0.9999
    loss = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat));


    avg_weights = 0
    for i in range(11):
        avg_weights += model[i];
    avg_weights = avg_weights/11;
    loss += (lam * (avg_weights*avg_weights))
    return loss

#updating weights by calculating the gradient
def update_weights(model, inputs, y_hat, y, learning_rate, regularization_weight):
    new_model = np.zeros(12)
    for i in range(len(model)-1):
        if( i == 2 or i == 3 or i == 5 or i == 9):
            continue
        new_model[i] = model[i] - (learning_rate*( (2*inputs[i]*(y_hat - y)) + 2*regularization_weight*model[i]))
    new_model[-1] = model[-1] - (learning_rate * ( (2*(y_hat - y))) )
    return new_model

def SGDSolver(phase, x, y, alpha, lamb, nepoch, epsilon, params):
    best_alpha = -1
    best_lam = -1
    if (phase == "Training"):
        #print("\n\nTraining...\n\n")
        models = []
        errors = []
        class_counts = []
        
        for c in range(3):
            best_curr_model = initial_values()
            initial_error = 100000000
            lowest_error = 100000000
            for alph in np.arange(alpha[0], alpha[1], ((alpha[1]-alpha[0])/10)):
                for lam in np.arange(lamb[0], lamb[1], ((lamb[1]-lamb[0])/10)):
                    #looping through each epoch
                    model = initial_values()
                    error = 0
                    #curr_class_cout = 0
                    for ep in range(nepoch):
                        #print("Epoch: " ,ep, " out of " ,nepoch, " " )

                        for feature_set in range(len(x)):
                            #print("Batch " ,feature_set+1, " out of ",len(x)," ")

                            #computing y^ with current model
                            y_hat = compute_y_hat(model, x[feature_set]);
                            real_out = -1
                            #computing the loss with current model
                            if(c == 0 and (y[feature_set] >= 5)):
                                real_out = 0
                            elif(c == 1 and ((y[feature_set] != 5) and (y[feature_set] != 6) ) ):
                                real_out = 0
                            elif(c == 2 and (y[feature_set] < 7)):
                                real_out = 0
                            else:
                                real_out = 1
                            #print("\n Real_Out class = ")
                            #print(real_out)
                            loss = compute_loss(y_hat, real_out, lam, model)
                            # if(loss < epsilon):
                            #     break
                            #print("\nLoss: ",loss,"\n")

                            #updating weights using SGD
                            model = update_weights(model, x[feature_set], y_hat, real_out, alph, lam) 

                    #checking the error here so I can use the data for 
                    for feature_set in range(len(x)):
                        rc = -1
                        if(c == 0 and (y[feature_set] >= 5)):
                            rc = 0
                        elif(c == 1 and ((y[feature_set] != 5) and (y[feature_set] != 6) )):
                            rc = 0
                        elif(c == 2 and (y[feature_set] < 7)):
                            rc = 0
                        else:
                            rc = 1
                            #curr_class_count +=1
                        predicted_out = compute_y_hat(model, x[feature_set])
                        if predicted_out < 0.5:
                            predicted_out = 0
                        else:
                            predicted_out = 1
                        error += abs(predicted_out - rc)
                    #error = error/len(x)
                    #error = error*error
                    #saving the best data to use
                    if error <= lowest_error:
                        lowest_error = error
                        best_curr_model = model
                        best_alpha = alph
                        best_lam = lam

            models.append(best_curr_model)
            # errors.append(lowest_error)
            # class_counts.append(curr_class_count)
        return(models,best_alpha,best_lam)
    elif(phase == "Validation"):
        #print("\n\nValidation...\n\n")
        acc_class1 = 0
        acc_class2 = 0
        acc_class3 = 0
        predict_class1 = 0
        predict_class2 = 0
        predict_class3 = 0
        for i in range(len(x)):
            predict_class1 = compute_y_hat(params[0], x[i])
            predict_class2 = compute_y_hat(params[1], x[i])
            predict_class3 = compute_y_hat(params[2], x[i])
            if((predict_class1 > predict_class2) and (predict_class1 > predict_class3)):
                if y[i] <5:
                    acc_class1 +=1
            if((predict_class2 > predict_class1) and (predict_class2 > predict_class3)):
                if ((y[i] > 4) and (y[i] <7)):
                    acc_class2 +=1
            if((predict_class3 > predict_class1) and (predict_class3 > predict_class2)):
                if y[i] >=7:
                    acc_class3 +=1
        total_acc = acc_class1 + acc_class2 + acc_class3
        mse = ((len(x)-total_acc) / len(x))**2
        return mse
    else:
        #print("\n\nTesting...\n\n")
        out = np.zeros(len(x))
        predict_class1 = 0
        predict_class2 = 0
        predict_class3 = 0
        for i in range(len(x)):
            for i in range(len(x)):
                predict_class1 = compute_y_hat(params[0], x[i])
                predict_class2 = compute_y_hat(params[1], x[i])
                predict_class3 = compute_y_hat(params[2], x[i])
                if((predict_class1 > predict_class2) and (predict_class1 > predict_class3)):
                    out[i] = 0
                elif((predict_class2 > predict_class1) and (predict_class2 > predict_class3)):
                    out[i] = 1
                else:
                    out[i] = 2
        return out


# lam     = [0, 0.001]        # regularization weight [min, max]
# alpha   = [0,0.1]        # learning rate [min, max]
# nepochs = 1       # sample # of epochs
# epsilon = 0.0       # epsilon value
# param   = []
# train_red_x, train_red_y= load_data('hw2_winequality-red_train.npy')
# bad = 0
# med = 0
# good = 0
# for i in train_red_y:
#     if i <5:
#         bad+=1
#     elif i >6:
#         good+=1
#     else:
#         med +=1
# #print(train_red_y)
# models, errors,class_c = SGDSolver('Training', train_red_x, train_red_y, alpha, lam, nepochs, epsilon, param)

# print("\nModel count ", len(models))
# print("\nError count" , len(errors))
# print("\nClass counts" , len(class_c))

# print("\nModel for Bad: ")
# print(models[0])
# print("\nError for Bad: ")
# print(errors[0])
# print("\nClass count")
# print(class_c[0])

# print("\nModel for Medium: ")
# print(models[1])
# print("\nError for Medium: ")
# print(errors[1])
# print("\nClass count")
# print(class_c[1])


# print("\nModel for Good: ")
# print(models[2])
# print("\nError for Good: ")
# print(errors[2])
# print("\nClass count")
# print(class_c[2])
