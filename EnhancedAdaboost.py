# Comparison of 5 ensemble classifiers. These are:
# 1. Adaboost with logistic regression as a weak learner
# 2. Adaboost with decision tree as a weak learner
# 3. Adaboost with Neural Network classifier as a weak learner
# 4. Adaboost with Plurality voting classifier (using the above three classifiers) as a weak learner
# 5. Enhanced adaboost: Adaboost with the best of the above four classifiers chosen as a weak learner for each iteration


import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import math
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import ttest_rel


# Class for Logistic regression classifier
class LogisticRegressionModel:

    def __init__(self, random_state=0):
        self.model = LogisticRegression(random_state=random_state)

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        # Train the Logistic Regression model
        self.model.fit(X_train, y_train)
        # Predict on the test data
        y_pred = self.model.predict(X_test)
        # Calculate accuracy
        accuracy_LR = accuracy_score(y_test, y_pred)

        #Returns predicted list and accuracy
        return y_pred, accuracy_LR


# Class for Decision tree classifier
class DecisionTreeModel:
    def __init__(self, max_depth=5, random_state=0):
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        # Train the Decision Tree model
        self.model.fit(X_train, y_train)
        # Predict on the test data
        y_pred = self.model.predict(X_test)
        # Calculate accuracy
        accuracy_DT = accuracy_score(y_test, y_pred)

        # Returns predicted list and accuracy
        return y_pred, accuracy_DT


# Class for Neural Network classifier
class NeuralNetworkModel:
    def __init__(self, hidden_layer_sizes=(10,10), max_iter=360, random_state=0):
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state)

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        # Train the Neural Network model
        self.model.fit(X_train, y_train)
        # Predict on the test data
        y_pred = self.model.predict(X_test)
        # Calculate accuracy
        accuracy_NN = accuracy_score(y_test, y_pred)

        return y_pred, accuracy_NN


# Class implementing Plurality voting classifier which uses logistic regression, decision tree, and neural network classifiers
class PluralityVoting:
    def __init__(self, y_pred_lr, y_pred_dt, y_pred_nn, y_test):
        self.y_pred_lr = y_pred_lr
        self.y_pred_dt = y_pred_dt
        self.y_pred_nn = y_pred_nn
        self.y_test = y_test

    def plurality_voting(self):

        # List for storing predicted values
        y_pred_ensemble = []

        # Identify the vote for each item
        for i in range(len(self.y_pred_lr)):
            votes = [self.y_pred_lr[i], self.y_pred_dt[i], self.y_pred_nn[i]]
            most_common = mode(votes).mode
            if isinstance(most_common, np.ndarray):
                y_pred_ensemble.append(most_common[0])
            else:
                y_pred_ensemble.append(most_common)
        return y_pred_ensemble

    def train_and_evaluate(self):
        # Perform plurality voting
        y_pred_ensemble = self.plurality_voting()

        # Calculate accuracy
        accuracy_ensemble = accuracy_score(self.y_test, y_pred_ensemble)

        return y_pred_ensemble, accuracy_ensemble


# Function used by different versions of Adaboost to help pick a random sample
# THIS FUNCTION HAD BEEN PROVIDED TO THE STUDENTS MY DR. MARCOLINO IN THE PAST
def sample(p):

    # First, get a random value between 0 and 1.0:
    randomValue = random.random()

    left = 0
    right = p[0]

    # Check in which interval the random value lies.
    # I check here up to one before the last feature, so that the update in the end of the for loop works
    for item in range(len(p)-1):
        if (randomValue >= left and randomValue < right):
            return item

        left = right

        right = right + p[item+1]


    # If nothing was selected previously, the randomValue must be in the last interval.
    return (len(p)-1)


# ***CODE OF ALL FIVE VERSIONS OF ADABOOST IS WRITTEN FROM SCRATCH BY ME***
# 1. First Ensemble classifier class: Adaboost using logistic regression classifier as a weak learner
class AdaBoost_logistic:

    def __init__(self, nItems=2400):
        self.nItems = nItems
        self.logistic_classifier = LogisticRegressionModel()
        self.error = 0  # Initial value of error is set to 0
        self.weak_classifier = []  # List to store all trained weak classifiers
        self.beta_list = []  # To store values of Beta

    def train(self, X_train, y_train, T):           # T = Number of weak classifiers used

        prob_list = [1/len(X_train)] * len(X_train)  # Use a temporary variable 'prob_list' to store the probability list
        iterations = 0

        while iterations < T:  # Checking if number of classifiers trained are less than T
            deviation = []  # Stores deviation between predicted and true labels
            weights = []
            sum_weights = 0
            prob = []
            index_list = []
            self.error = 0  # Resets self.error to 0 before training every weak classifier
            self.logistic_classifier = LogisticRegressionModel()  # Creates a new instance of Logistic regression class every time a new weal classifier is trained

            # Generate training data and labels for corresponding probabilities using sample function with repetition
            for i in range(self.nItems):
                index = sample(prob_list)
                index_list.append(index)
            X_train_ensemble = X_train.iloc[index_list]
            Y_train_ensemble = y_train.iloc[index_list]

            # Train the weak classifier
            y_pred_LR, accuracy_LR = self.logistic_classifier.train_and_evaluate(X_train_ensemble, X_train, Y_train_ensemble, y_train)

            # Calculate deviations, and error:
            for k in range(len(y_pred_LR)):
                deviation.append(abs(y_pred_LR[k] - y_train.iloc[k]))
                self.error = self.error + prob_list[k] * deviation[k]

            # Store the weak classifier in a list
            self.weak_classifier.append(self.logistic_classifier)

            beta = max((self.error / (1 - self.error)), 0.001)      #Calculate Beta; use minimum beta as 0.001 to avoid division by 0 in the 'classify' method

            self.beta_list.append(beta)

            #Calculate and store weights of items in a list
            for l in range(len(prob_list)):
                weight_i = prob_list[l]*beta**(1-deviation[l])
                weights.append(weight_i)
                sum_weights = sum_weights + weight_i

            #Normalise the weights to convert them into probabilities
            for m in range(len(weights)):
                prob.append(weights[m] / sum_weights)

            prob_list = prob        #Update the probability list for creating the next training data set

            iterations = iterations + 1     #Update the number of weak classifiers already trained

    def classify(self, X_test):
            return X_test.apply(self.classify_row, axis=1).tolist()

    # Helper method:
    def classify_row(self, row):
        left_sum = 0            #Initialise the left side of Adaboost inequality used for last classification step to 0
        right_sum = 0           #Initialise the right side of Adaboost inequality used for last classification step to 0
        for n in range(len(self.weak_classifier)):

            row_df = pd.DataFrame([row])
            # Update the expression on the left side of the inequality used for final classification
            left_sum += (self.weak_classifier[n].model.predict(row_df)[0]) * math.log(1/self.beta_list[n])
            # Update the expression on the right side of the inequality used for final classification
            right_sum += (math.log(1/self.beta_list[n])) * 0.5

        if left_sum >= right_sum:           #Check the inequality condition for final classification and return the class
            return 1
        else:
            return 0


# 2. Second Ensemble classifier class: Adaboost using Decision Tree classifier as a weak learner
class AdaBoost_DT:

    def __init__(self, nItems=2400):
        self.nItems = nItems
        self.DT_classifier = DecisionTreeModel()
        self.error = 0  # Initial value of error is set to 0
        self.weak_classifier = []  # List to store all trained weak classifiers
        self.beta_list = []  # To store values of Beta

    def train(self, X_train, y_train, T):           # T = Number of weak classifiers used

        prob_list = [1/len(X_train)] * len(X_train)  # Use a temporary variable 'prob_list' to store the probability list
        iterations = 0

        while iterations < T:  # Checking if number of classifiers trained are less than T
            deviation = []  # Stores deviation between predicted and true labels
            weights = []
            sum_weights = 0
            prob = []
            index_list = []
            self.error = 0  # Resets self.error to 0 before training every weak classifier
            self.DT_classifier = DecisionTreeModel()

            # Generate training data and labels for corresponding probabilities using sample function with repetition
            for i in range(self.nItems):
                index = sample(prob_list)
                index_list.append(index)
            X_train_ensemble = X_train.iloc[index_list]
            Y_train_ensemble = y_train.iloc[index_list]

            # Train the weak classifier
            y_pred_DT, accuracy_DT = self.DT_classifier.train_and_evaluate(X_train_ensemble, X_train, Y_train_ensemble, y_train)

            # Calculate the error in prediction for the weak classifiers
            for k in range(len(y_pred_DT)):
                deviation.append(abs(y_pred_DT[k] - y_train.iloc[k]))
                self.error = self.error + prob_list[k] * deviation[k]

            self.weak_classifier.append(self.DT_classifier)

            beta = max((self.error / (1 - self.error)), 0.001)      #Calculate Beta; use minimum beta as 0.001 to avoid division by 0 in the 'classify' method

            self.beta_list.append(beta)

            #Calculate and store weights of items in a list
            for l in range(len(prob_list)):
                weight_i = prob_list[l]*beta**(1-deviation[l])
                weights.append(weight_i)
                sum_weights = sum_weights + weight_i

            #Normalise the weights to convert them into probabilities
            for m in range(len(weights)):
                prob.append(weights[m] / sum_weights)

            prob_list = prob        #Update the probability list for creating the next training data set

            iterations = iterations + 1     #Update the number of weak classifiers already trained

    def classify(self, X_test):
            return X_test.apply(self.classify_row, axis=1).tolist()

    # Helper method:
    def classify_row(self, row):
        left_sum = 0            #Initialise the left side of Adaboost inequality used for last classification step to 0
        right_sum = 0           #Initialise the right side of Adaboost inequality used for last classification step to 0
        for n in range(len(self.weak_classifier)):

            row_df = pd.DataFrame([row])
            # Update the expression on the left side of the inequality used for final classification
            left_sum += (self.weak_classifier[n].model.predict(row_df)[0]) * math.log(1/self.beta_list[n])
            # Update the expression on the right side of the inequality used for final classification
            right_sum += (math.log(1/self.beta_list[n])) * 0.5

        if left_sum >= right_sum:           #Check the inequality condition for final classification and return the class
            return 1
        else:
            return 0


# 3. Third Ensemble classifier class: Adaboost using Neural Network classifier as a weak learner
class AdaBoost_NN:

    def __init__(self, nItems=2400):
        self.nItems = nItems
        self.NN_classifier = NeuralNetworkModel()
        self.error = 0  # Initial value of error is set to 0
        self.weak_classifier = []  # List to store all trained weak classifiers
        self.beta_list = []  # To store values of Beta

    def train(self, X_train, y_train, T):           # T = Number of weak classifiers used

        prob_list = [1/len(X_train)] * len(X_train)  # Use a temporary variable 'prob_list' to store the probability list
        iterations = 0

        while iterations < T:  # Checking if number of classifiers trained are less than T
            deviation = []  # Stores deviation between predicted and true labels
            weights = []
            sum_weights = 0
            prob = []
            index_list = []
            self.error = 0  # Resets self.error to 0 before training every weak classifier
            self.NN_classifier = NeuralNetworkModel()

            # Generate training data and lables for corresponding probablilities using sample function with repetition
            for i in range(self.nItems):
                index = sample(prob_list)
                index_list.append(index)
            X_train_ensemble = X_train.iloc[index_list]
            Y_train_ensemble = y_train.iloc[index_list]

            # Train the weak classifier
            y_pred_NN, accuracy_NN = self.NN_classifier.train_and_evaluate(X_train_ensemble, X_train, Y_train_ensemble, y_train)

            # Calculate the error in prediction for the weak classifiers
            for k in range(len(y_pred_NN)):
                deviation.append(abs(y_pred_NN[k] - y_train.iloc[k]))
                self.error = self.error + prob_list[k] * deviation[k]

            # Store the weak classifier in a list
            self.weak_classifier.append(self.NN_classifier)

            beta = max((self.error / (1 - self.error)), 0.001)      #Calculate Beta; use minimum beta as 0.001 to avoid division by 0 in the 'classify' method

            self.beta_list.append(beta)

            #Calculate and store weights of items in a list
            for l in range(len(prob_list)):
                weight_i = prob_list[l]*beta**(1-deviation[l])
                weights.append(weight_i)
                sum_weights = sum_weights + weight_i

            #Normalise the weights to convert them into probabilities
            for m in range(len(weights)):
                prob.append(weights[m] / sum_weights)

            prob_list = prob        #Update the probability list for creating the next training data set

            iterations = iterations + 1     #Update the number of weak classifiers already trained

    def classify(self, X_test):
            return X_test.apply(self.classify_row, axis=1).tolist()

    def classify_row(self, row):
        left_sum = 0            #Initialise the left side of Adaboost inequality used for last classification step to 0
        right_sum = 0           #Initialise the right side of Adaboost inequality used for last classification step to 0
        for n in range(len(self.weak_classifier)):

            row_df = pd.DataFrame([row])
            # Update the expression on the left side of the inequality used for final classification
            left_sum += (self.weak_classifier[n].model.predict(row_df)[0]) * math.log(1/self.beta_list[n])
            # Update the expression on the right side of the inequality used for final classification
            right_sum += (math.log(1/self.beta_list[n])) * 0.5

        if left_sum >= right_sum:           #Check the inequality condition for final classification and return the class
            return 1
        else:
            return 0


# 4. Fourth Ensemble classifier class: Adaboost using Plurality Voting ensemble classifier as a weak learner.
# The plurality voting classifier using votes of Logistic regression, Decision Tree, amd Neural Network classifiers
class AdaBoost_plurality_voting:

    def __init__(self, nItems=2400):
        self.nItems = nItems

        # Three base classifiers for plurality voting:
        self.logistic_classifier = LogisticRegressionModel()
        self.DT_classifier = DecisionTreeModel()
        self.NN_classifier = NeuralNetworkModel()

        self.error = 0  # Initial value of error is set to 0

        # Lists for storing base classifiers
        self.weak_classifier_LR_list = []
        self.weak_classifier_DT_list = []
        self.weak_classifier_NN_list = []

        self.weak_classifier = []  # List to store all trained weak classifiers
        self.beta_list = []  # To store values of Beta

    def train(self, X_train, y_train, T):           # T = Number of weak learners used

        prob_list = [1/len(X_train)] * len(X_train)  # Use a temporary variable 'prob_list' to store the probability list
        iterations = 0

        while iterations < T:  # Checking if number of classifiers trained are less than T
            deviation = []  # Stores deviation between predicted and true labels
            weights = []
            sum_weights = 0
            prob = []
            index_list = []
            self.error = 0  # Resets self.error to 0 before training every weak classifier

            # Creates a new instance of base classifier classes every time a new weak classifier is trained
            self.logistic_classifier = LogisticRegressionModel()
            self.DT_classifier = DecisionTreeModel()
            self.NN_classifier = NeuralNetworkModel()

            # Generate training data and labels for corresponding probabilities using sample function with repetition
            for i in range(self.nItems):
                index = sample(prob_list)
                index_list.append(index)
            X_train_ensemble = X_train.iloc[index_list]
            Y_train_ensemble = y_train.iloc[index_list]

            # Predictions of base classifiers
            y_pred_LR, accuracy_LR = self.logistic_classifier.train_and_evaluate(X_train_ensemble, X_train, Y_train_ensemble, y_train)
            y_pred_DT, accuracy_DT = self.DT_classifier.train_and_evaluate(X_train_ensemble, X_train, Y_train_ensemble, y_train)
            y_pred_NN, accuracy_NN = self.NN_classifier.train_and_evaluate(X_train_ensemble, X_train, Y_train_ensemble, y_train)

            # Get predictions of plurality voting classifier for the current iteration of Adaboost ensemble
            self.PV_classifier = PluralityVoting(y_pred_LR, y_pred_DT, y_pred_NN, y_train)
            y_pred_PV, accuracy_PV = self.PV_classifier.train_and_evaluate()

            # Calculate the error in prediction for the weak classifiers
            for k in range(len(y_pred_PV)):
                deviation.append(abs(y_pred_PV[k] - y_train.iloc[k]))
                self.error = self.error + prob_list[k] * deviation[k]

            # Store all base classifiers in their respective lists
            self.weak_classifier_LR_list.append(self.logistic_classifier)
            self.weak_classifier_DT_list.append(self.DT_classifier)
            self.weak_classifier_NN_list.append(self.NN_classifier)
            self.weak_classifier.append(self.PV_classifier)  # Store the weak classifiers in a list

            beta = max((self.error / (1 - self.error)), 0.001)    #Calculate Beta; use minimum beta as 0.001 to avoid division by 0 in the 'classify' method

            self.beta_list.append(beta)

            #Calculate and store weights of items in a list
            for l in range(len(prob_list)):
                weight_i = prob_list[l]*beta**(1-deviation[l])
                weights.append(weight_i)
                sum_weights = sum_weights + weight_i

            #Normalise the weights to convert them into probabilities
            for m in range(len(weights)):
                prob.append(weights[m] / sum_weights)

            prob_list = prob        #Update the probability list for creating the next training data set

            iterations = iterations + 1     #Update the number of weak classifiers already trained

    def classify(self, X_test):
            return X_test.apply(self.classify_row, axis=1).tolist()

    # Helper function
    def classify_row(self, row):
        left_sum = 0            #Initialise the left side of Adaboost inequality used for last classification step to 0
        right_sum = 0           #Initialise the right side of Adaboost inequality used for last classification step to 0
        for n in range(len(self.weak_classifier)):

            row_df = pd.DataFrame([row])
            # Update the expression on the left side of the inequality used for final classification

            # Get prediction from base classifiers used for plurality voting
            weak_LR_prediction = self.weak_classifier_LR_list[n].model.predict(row_df)[0]
            weak_DT_prediction = self.weak_classifier_DT_list[n].model.predict(row_df)[0]
            weak_NN_prediction = self.weak_classifier_NN_list[n].model.predict(row_df)[0]
            predictions = mode([weak_LR_prediction, weak_DT_prediction, weak_NN_prediction]).mode
            prediction = predictions[0] if isinstance(predictions, np.ndarray) else predictions

            left_sum += prediction * math.log(1/self.beta_list[n])
            # Update the expression on the right side of the inequality used for final classification
            right_sum += (math.log(1/self.beta_list[n])) * 0.5

        # Check the inequality condition for final classification and return the prediction
        if left_sum >= right_sum:
            return 1
        else:
            return 0


# 5. Fifth and final Ensemble classifier class: Adaboost using the best performing weak classifier among...
# Logistic regression, Decision Tree, Neural Network, and Plurality Voting classifier for each iteration.
class AdaBoost_enhanced:

    def __init__(self, nItems=2400):
        self.nItems = nItems
        self.logistic_classifier = LogisticRegressionModel()
        self.DT_classifier = DecisionTreeModel()
        self.NN_classifier = NeuralNetworkModel()
        self.error = [0, 0, 0, 0]  # Initial value of error is set to 0

        # Create lists for storing base classifiers used by plurality voting ensemble
        self.weak_classifier_LR_list = []
        self.weak_classifier_DT_list = []
        self.weak_classifier_NN_list = []

        self.weak_classifier = []  # List to store all trained weak classifiers
        self.beta_list = []  # To store values of Beta

    def train(self, X_train, y_train, T):

        prob_list = [1/len(X_train)] * len(X_train)  # Use a temporary variable 'prob_list' to store the probability list
        iterations = 0

        while iterations < T:  # Checking if number of classifiers trained are less than T
            deviation = [[], [], [], []]  # Stores deviation between predicted and true labels
            weights = []
            sum_weights = 0
            prob = []
            index_list = []
            self.error = [0, 0, 0, 0]  # Resets self.error to 0 before training every weak classifier

            # Creates a new instance of base classifiers every time a new weak classifier is trained
            self.logistic_classifier = LogisticRegressionModel()
            self.DT_classifier = DecisionTreeModel()
            self.NN_classifier = NeuralNetworkModel()

            # Generate training data and labels for corresponding probabilities using sample function with repetition
            for i in range(self.nItems):
                index = sample(prob_list)
                index_list.append(index)
            X_train_ensemble = X_train.iloc[index_list]
            Y_train_ensemble = y_train.iloc[index_list]

            # Get predictions of base classifiers
            y_pred_LR, accuracy_LR = self.logistic_classifier.train_and_evaluate(X_train_ensemble, X_train, Y_train_ensemble, y_train)
            y_pred_DT, accuracy_DT = self.DT_classifier.train_and_evaluate(X_train_ensemble, X_train, Y_train_ensemble, y_train)
            y_pred_NN, accuracy_NN = self.NN_classifier.train_and_evaluate(X_train_ensemble, X_train, Y_train_ensemble, y_train)

            # Get prediction of plurality voting ensemble classifier
            self.PV_classifier = PluralityVoting(y_pred_LR, y_pred_DT, y_pred_NN, y_train)
            y_pred_PV, accuracy_PV = self.PV_classifier.train_and_evaluate()

            classifier_list = [self.logistic_classifier, self.DT_classifier, self.NN_classifier, self.PV_classifier]

            # Create a list storing lists of predictions by the four potential weak classifiers
            y_prediction_list = [y_pred_LR, y_pred_DT, y_pred_NN, y_pred_PV]

            # Calculate the error in prediction for the potential weak classifiers
            for j in range(len(y_prediction_list)):
                for k in range(len(y_prediction_list[j])):
                    deviation[j].append(abs(y_prediction_list[j][k] - y_train.iloc[k]))
                    self.error[j] = self.error[j] + prob_list[k] * deviation[j][k]

            # Select the weak classifier for the current iteration that has minimum error and store it in the weak classiifer list
            classifier_index = self.error.index(min(self.error))
            self.weak_classifier_LR_list.append(self.logistic_classifier)
            self.weak_classifier_DT_list.append(self.DT_classifier)
            self.weak_classifier_NN_list.append(self.NN_classifier)
            self.weak_classifier.append(classifier_list[classifier_index])  # Store the trained classifiers in a list

            beta = max((self.error[classifier_index] / (1 - self.error[classifier_index])), 0.001)      #Calculate Beta; use minimum beta as 0.001 to avoid division by 0 in the 'classify' method

            self.beta_list.append(beta)

            #Calculate and store weights of items in a list
            for l in range(len(prob_list)):
                weight_i = prob_list[l]*beta**(1-deviation[classifier_index][l])
                weights.append(weight_i)
                sum_weights = sum_weights + weight_i

            #Normalise the weights to convert them into probabilities
            for m in range(len(weights)):
                prob.append(weights[m] / sum_weights)

            prob_list = prob        #Update the probability list for creating the next training data set

            iterations = iterations + 1     #Update the number of weak classifiers already trained

    def classify(self, X_test):
            return X_test.apply(self.classify_row, axis=1).tolist()

    # Helper mathod:
    def classify_row(self, row):
        left_sum = 0            #Initialise the left side of Adaboost inequality used for last classification step to 0
        right_sum = 0           #Initialise the right side of Adaboost inequality used for last classification step to 0
        for n in range(len(self.weak_classifier)):

            row_df = pd.DataFrame([row])

            # Get prediction if the weak classifier is logistic, decision tree, or Neural network:
            if isinstance(self.weak_classifier[n], PluralityVoting):
                weak_LR_prediction = self.weak_classifier_LR_list[n].model.predict(row_df)[0]
                weak_DT_prediction = self.weak_classifier_DT_list[n].model.predict(row_df)[0]
                weak_NN_prediction = self.weak_classifier_NN_list[n].model.predict(row_df)[0]

                predictions = mode([weak_LR_prediction, weak_DT_prediction, weak_NN_prediction]).mode
                prediction = predictions[0] if isinstance(predictions, np.ndarray) else predictions

            # Get prediction if the weak classifier is plurality voting classifier:
            else:
                prediction = (self.weak_classifier[n].model.predict(row_df)[0])

            # Update the expression on the left side of the inequality used for final classification
            left_sum += prediction * math.log(1/self.beta_list[n])
            # Update the expression on the right side of the inequality used for final classification
            right_sum += (math.log(1/self.beta_list[n])) * 0.5

        if left_sum >= right_sum:           #Check the inequality condition for final classification and return the class
            return 1
        else:
            return 0


# Function for obtaining predictions from each of the five ensemble classifiers:
def generate_data(n, X_train, y_train, X_test, y_test):

    adaboost_data = []
    random.seed(0)

    # Prediction from the first ensemble classifier:
    adaBoost_logistic = AdaBoost_logistic()
    adaBoost_logistic.train(X_train, y_train, T=n)
    y_pred_adaBoost_logistic = adaBoost_logistic.classify(X_test)
    accuracy_adaBoost_logistic = accuracy_score(y_test, y_pred_adaBoost_logistic)
    adaboost_data.append([accuracy_adaBoost_logistic, n, 1])            #This classifier is assigned code = 1

    # Prediction from the second ensemble classifier:
    adaBoost_DT = AdaBoost_DT()
    adaBoost_DT.train(X_train, y_train, T=n)
    y_pred_adaboost_DT = adaBoost_DT.classify(X_test)
    accuracy_adaboost_DT = accuracy_score(y_test, y_pred_adaboost_DT)
    adaboost_data.append([accuracy_adaboost_DT, n, 2])          #This classifier is assigned code = 2

    # Prediction from the third ensemble classifier:
    adaBoost_NN = AdaBoost_NN()
    adaBoost_NN.train(X_train, y_train, T=n)
    y_pred_adaboost_NN = adaBoost_NN.classify(X_test)
    accuracy_adaboost_NN = accuracy_score(y_test, y_pred_adaboost_NN)
    adaboost_data.append([accuracy_adaboost_NN, n, 3])          #This classifier is assigned code = 3

    # Prediction from the fourth ensemble classifier:
    adaBoost_PV = AdaBoost_plurality_voting()
    adaBoost_PV.train(X_train, y_train, T=n)
    y_pred_adaboost_PV = adaBoost_PV.classify(X_test)
    accuracy_adaboost_PV = accuracy_score(y_test, y_pred_adaboost_PV)
    adaboost_data.append([accuracy_adaboost_PV, n, 4])          #This classifier is assigned code = 4

    # Prediction from the fifth ensemble classifier:
    adaBoost_enhanced = AdaBoost_enhanced()
    adaBoost_enhanced.train(X_train, y_train, T=n)
    y_pred_adaboost_enhanced = adaBoost_enhanced.classify(X_test)
    accuracy_adaboost_enhanced = accuracy_score(y_test, y_pred_adaboost_enhanced)
    adaboost_data.append([accuracy_adaboost_enhanced, n, 5])            #This classifier is assigned code = 5

    return adaboost_data


def split_data(data):
    # Split the data into features and labels

    # First three columns of each dataset used are features
    X = data.iloc[:, :3]
    # Last column has the labels
    y = data.iloc[:, 3]
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

# Load and split the first dataset:
data1 = pd.read_csv('Stroke_BRFSS.csv')
X_train1, X_test1, y_train1, y_test1 = split_data(data1)

# I. Run all five ensemble classifiers for different number of iterations:

iterations = [10, 20, 30, 40, 50, 60 ,70, 80, 90, 100]

final_data = []         # List for storing accuracy data of all five ensemble classifiers

# Get prediction accuracy values from all five ensemble classifiers for various number of weak learners using the first dataset
for n in iterations:
    ensemble_data = generate_data(n, X_train1, y_train1, X_test1, y_test1)
    for element in ensemble_data:
        final_data.append(element)

# Convert list of lists storing accuracy numbers into a dataframe:
adaboost_versions_df = pd.DataFrame(final_data, columns=['Accuracy', 'Iterations', 'Classifier'])

# Display the DataFrame
print("Accuracy data for 5 different ensemble classifiers by various number of iterations:")
print(adaboost_versions_df)
print("\n")

# Convert the last columnn of ensemble codes into categorical variable:
adaboost_versions_df['Classifier'] = adaboost_versions_df['Classifier'].astype('category')

# Create a line plot for accuracy of all ensembles for various number of weak learners:
plt.figure(figsize=(10, 6))
sns.lineplot(data=adaboost_versions_df, x='Iterations', y='Accuracy', hue='Classifier', marker='o', palette='bright')
# Add labels and title
plt.xlabel('Iterations', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Line Plot of Accuracy vs Iterations by Classifier', fontsize=16)
plt.legend(title='Classifier', title_fontsize='13', fontsize='12')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# Save the plot
plt.savefig('line_plot.png')
# Show plot
plt.show()


# Create box-plot of accuracy versus ensemble system
plt.figure(figsize=(10, 6))
colours = ["#33FF57", "#FF5733", "#3357FF", "#FF33A1", "#33FFF5"]
sns.boxplot(x='Classifier', y='Accuracy', data=adaboost_versions_df, hue='Classifier', palette=colours, legend=False)
plt.title('Boxplot of Accuracy by Classifier', fontsize=16)
plt.xlabel('Classifier', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.savefig('box_plot.png')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()



# Run a linear model with accuracy as a dependent variable and Iterations (number of weak learners) and ensemble system type as features
lin_model = smf.ols(formula='Accuracy ~ Iterations + C(Classifier)', data=adaboost_versions_df).fit()
# Print the summary of the model
print(lin_model.summary())

# Draw the Q-Q plot to check for normality
# Chat-GPt was used for this part
residuals = lin_model.resid
sm.qqplot(residuals, line='45')
plt.title('Q-Q plot of residuals')
plt.savefig('QQ_plot.png')
plt.show()


# II. Running all five ensemble systems each using 100 weak learners, for four different datasets and comparing their prediction accuracy
# Load and split data
data1 = pd.read_csv('Stroke_BRFSS.csv')
X_train1, X_test1, y_train1, y_test1 = split_data(data1)
data_for_evaluation1 = generate_data(100, X_train1, y_train1, X_test1, y_test1)

# Load and split data
data2 = pd.read_csv('Pulsar_star.csv')
X_train2, X_test2, y_train2, y_test2 = split_data(data2)
data_for_evaluation2 = generate_data(100, X_train2, y_train2, X_test2, y_test2)

# Load and split data
data3 = pd.read_csv('Sepsis.csv')
X_train3, X_test3, y_train3, y_test3 = split_data(data3)
data_for_evaluation3 = generate_data(100, X_train3, y_train3, X_test3, y_test3)

# Load and split data
data4 = pd.read_csv('Skin_NonSkin.csv')
X_train4, X_test4, y_train4, y_test4 = split_data(data4)
data_for_evaluation4 = generate_data(100, X_train4, y_train4, X_test4, y_test4)



# Extract the first element (score) from each sublist and combine them into a DataFrame
data = {
    'Stroke_BRFSS': [x[0] for x in data_for_evaluation1],
    'Pulsar_star' : [x[0] for x in data_for_evaluation2],
    'Sepsis'      : [x[0] for x in data_for_evaluation3],
    'Skin_NonSkin': [x[0] for x in data_for_evaluation4]
}

# Create a DataFrame for storing prediction accuracy of all five ensemble systems for four distinct datasets
accuracy_df = pd.DataFrame(data, index=['Adaboost_logistic', 'Adaboost_DT', 'Adaboost_NN', 'Adaboost_PV', 'Enhanced_Adaboost'])

# Round off values
accuracy_df_rounded = accuracy_df.round(3)

print("\n")
print("Accuracy table for five ensemble classifiers and four datasets:\n")
print(accuracy_df_rounded)

Adaboost_logistic = accuracy_df.loc['Adaboost_logistic']
Adaboost_DT = accuracy_df.loc['Adaboost_DT']
Adaboost_NN = accuracy_df.loc['Adaboost_NN']
Adaboost_PV = accuracy_df.loc['Adaboost_PV']
Enhanced_Adaboost = accuracy_df.loc['Enhanced_Adaboost']

p_values = []

# Run paired t-tests between all possible pairs of ensemble systems
t_stat_L_DT, p_value_L_DT = ttest_rel(Adaboost_logistic, Adaboost_DT)
p_values.append(["Adaboost_logistic, Adaboost_DT", t_stat_L_DT, p_value_L_DT])
t_stat_L_NN, p_value_L_NN = ttest_rel(Adaboost_logistic, Adaboost_NN)
p_values.append(["Adaboost_logistic, Adaboost_NN", t_stat_L_NN, p_value_L_NN])
t_stat_L_PV, p_value_L_PV = ttest_rel(Adaboost_logistic, Adaboost_PV)
p_values.append(["Adaboost_logistic, Adaboost_PV", t_stat_L_PV, p_value_L_PV])
t_stat_L_Enh, p_value_L_Enh = ttest_rel(Adaboost_logistic, Enhanced_Adaboost)
p_values.append(["Adaboost_logistic, Enhanced_Adaboost", t_stat_L_Enh, p_value_L_Enh])

t_stat_DT_NN, p_value_DT_NN = ttest_rel(Adaboost_DT, Adaboost_NN)
p_values.append(["Adaboost_DT, Adaboost_NN", t_stat_DT_NN, p_value_DT_NN])
t_stat_DT_PV, p_value_DT_PV = ttest_rel(Adaboost_DT, Adaboost_PV)
p_values.append(["Adaboost_DT, Adaboost_PV", t_stat_DT_PV, p_value_DT_PV])
t_stat_DT_Enh, p_value_DT_Enh = ttest_rel(Adaboost_DT, Enhanced_Adaboost)
p_values.append(["Adaboost_DT, Enhanced_Adaboost", t_stat_DT_Enh, p_value_DT_Enh])

t_stat_NN_PV, p_value_NN_PV = ttest_rel(Adaboost_NN, Adaboost_PV)
p_values.append(["Adaboost_NN, Adaboost_PV", t_stat_NN_PV, p_value_NN_PV])
t_stat_NN_Enh, p_value_NN_Enh = ttest_rel(Adaboost_NN, Enhanced_Adaboost)
p_values.append(["Adaboost_NN, Enhanced_Adaboost", t_stat_NN_Enh, p_value_NN_Enh])

t_stat_PV_Enh, p_value_PV_Enh = ttest_rel(Adaboost_PV, Enhanced_Adaboost)
p_values.append(["Adaboost_PV, Enhanced_Adaboost", t_stat_PV_Enh, p_value_PV_Enh])

# Name the columns
p_values_df = pd.DataFrame(p_values, columns=['Model pair', 't-statistic', 'p-value'])

p_values_df_rounded = p_values_df.round(3)

print("\n")
print("p-value table for paired t-test between all possible pairs of five ensemble classifiers:")
print(p_values_df_rounded)