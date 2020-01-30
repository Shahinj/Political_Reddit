import argparse
import os
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier  

###
import numpy as np
from sklearn.model_selection import KFold
###


def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    # print ('TODO')
    return np.trace(C) / np.sum(np.sum(C, axis = 1), axis = 0)


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    result = []
    for i in range(0, C.shape[0]):
        accuracy = C[i,i] / np.sum(C, axis = 0)[i]
        result.insert(i,accuracy)
    return result
    # print ('TODO')


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    result = []
    for i in range(0, C.shape[0]):
        accuracy = C[i,i] / np.sum(C, axis = 1)[i]
        result.insert(i,accuracy)
    return result
    # print ('TODO')
    

def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:      
       i: int, the index of the supposed best classifier
    '''
    # print('TODO Section 3.1')
    classifiers = ['SGDClassifier', 'GaussianNB', 'RandomForestClassifier', 'MLPClassifier', 'AdaBoostClassifier']
    
    best = 0
    best_acc = 0
    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        for classifier_name in classifiers:
            if  classifier_name == 'SGDClassifier':
                classifier = SGDClassifier()
            elif classifier_name == 'GaussianNB':
                classifier = GaussianNB()
            elif classifier_name == 'RandomForestClassifier':
                classifier = RandomForestClassifier()
            elif classifier_name == 'MLPClassifier':
                classifier = MLPClassifier()
            elif classifier_name == 'AdaBoostClassifier':
                classifier = AdaBoostClassifier()
            else:
                raise Exception('Classifier does not exist')
    
            
            classifier.fit(X_train, y_train)
            conf_matrix = confusion_matrix(y_test,classifier.predict(X_test))
            acc = accuracy(conf_matrix)
            rec = recall(conf_matrix)
            prec = precision(conf_matrix)
            
            if( acc > best_acc):
                best = classifiers.index(classifier_name)
                best_acc = acc

            # For each classifier, compute results and write the following output:
            outf.write(f'Results for {classifier_name}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {acc:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in rec]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in prec]}\n')
            outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')
            # pass
    
    iBest = best
    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    # print('TODO Section 3.2')
    classifiers = ['SGDClassifier', 'GaussianNB', 'RandomForestClassifier', 'MLPClassifier', 'AdaBoostClassifier']
    increments = [1,5,10,15,20]
    
    classifier_name = classifiers[iBest]
    if  classifier_name == 'SGDClassifier':
        classifier = SGDClassifier()
    elif classifier_name == 'GaussianNB':
        classifier = GaussianNB()
    elif classifier_name == 'RandomForestClassifier':
        classifier = RandomForestClassifier()
    elif classifier_name == 'MLPClassifier':
        classifier = MLPClassifier()
    elif classifier_name == 'AdaBoostClassifier':
        classifier = AdaBoostClassifier()
    else:
        raise Exception('Classifier does not exist')
    
    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        for inc in increments:
            idx = np.random.choice(X_train.shape[0], size= inc * 1000, replace = False)
            if(inc == 1):
                X_1k = X_train[idx,:]
                y_1k = y_train[idx]
            classifier.fit(X_train[idx,:], y_train[idx])
            conf_matrix = confusion_matrix(y_test,classifier.predict(X_test))
            acc = accuracy(conf_matrix)
            # For each number of training examples, compute results and write
            # the following output:
            outf.write(f'{inc}: {acc:.4f}\n')
            # pass

    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    # print('TODO Section 3.3')
    
    classifiers = ['SGDClassifier', 'GaussianNB', 'RandomForestClassifier', 'MLPClassifier', 'AdaBoostClassifier']

    classifier_name = classifiers[i]
    if  classifier_name == 'SGDClassifier':
        classifier = SGDClassifier()
    elif classifier_name == 'GaussianNB':
        classifier = GaussianNB()
    elif classifier_name == 'RandomForestClassifier':
        classifier = RandomForestClassifier()
    elif classifier_name == 'MLPClassifier':
        classifier = MLPClassifier()
    elif classifier_name == 'AdaBoostClassifier':
        classifier = AdaBoostClassifier()
    else:
        raise Exception('Classifier does not exist')
    
    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.
        search_range = [5,50]
        for k in search_range:
            selector = SelectKBest(f_classif, k)
            X_new = selector.fit_transform(X_train, y_train)
            pp = selector.pvalues_
            k_feat = k
            p_values = pp[np.argpartition(pp, k)[:k]]
        # for each number of features k_feat, write the p-values for
        # that number of features:
            outf.write(f'{k_feat} p-values: {[round(pval, 4) for pval in p_values]}\n')
        
        k = 5
        selector = SelectKBest(f_classif, k)
        
        #full dataset
        X_new_train = selector.fit_transform(X_train, y_train)
        pp_full = selector.pvalues_
        X_new_test  = selector.transform(X_test)
        
        classifier.fit(X_new_train, y_train)
        conf_matrix = confusion_matrix(y_test,classifier.predict(X_new_test))
        accuracy_full = accuracy(conf_matrix)
        
        #1k dataset
        X_new_train = selector.fit_transform(X_1k, y_1k)
        pp_1k = selector.pvalues_
        classifier.fit(X_new_train, y_1k)
        conf_matrix = confusion_matrix(y_test,classifier.predict(X_new_test))
        accuracy_1k = accuracy(conf_matrix)
        
        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        
        
        #indices
        feature_intersection = np.intersect1d(np.argpartition(pp_full, k)[:k] , np.argpartition(pp_1k, k)[:k])
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        top_5 = np.argpartition(pp_full, k)[:k]
        outf.write(f'Top-5 at higher: {top_5}\n')
        
    return 
        # pass


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    # print('TODO Section 3.4')
    
    classifiers = ['SGDClassifier', 'GaussianNB', 'RandomForestClassifier', 'MLPClassifier', 'AdaBoostClassifier']

    classifier_name = classifiers[i]
    
    
    kf = KFold(n_splits=5, shuffle = True)
    classifier_accuracies = [[] for i in range(0, len(classifiers))]
    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        for train_index, test_index in kf.split(X_train):
            kfold_accuracies = []
            X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
            y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
            for classifier_index, classifier_name in enumerate(classifiers):
                if  classifier_name == 'SGDClassifier':
                    classifier = SGDClassifier()
                elif classifier_name == 'GaussianNB':
                    classifier = GaussianNB()
                elif classifier_name == 'RandomForestClassifier':
                    classifier = RandomForestClassifier()
                elif classifier_name == 'MLPClassifier':
                    classifier = MLPClassifier()
                elif classifier_name == 'AdaBoostClassifier':
                    classifier = AdaBoostClassifier()
                else:
                    raise Exception('Classifier does not exist')
                classifier.fit(X_train_fold, y_train_fold)
                conf_matrix = confusion_matrix(y_test_fold,classifier.predict(X_test_fold))
                acc = accuracy(conf_matrix)
                classifier_accuracies[classifier_index].append(acc)
                kfold_accuracies.append(acc)
            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')                
            
        kfold_accuracies = [np.mean(i) for i in classifier_accuracies]
        
        best = np.argmax(kfold_accuracies)
        p_values = []
        for i, accs in enumerate(classifier_accuracies):
            if best != i:
                S = ttest_rel(classifier_accuracies[i], classifier_accuracies[best])
                p_values.append(S.pvalue)
        
        outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
        outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')
        # pass
    return

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to."
        # ,default=os.path.dirname(os.path.dirname(__file__))
        )
    args = parser.parse_args()       
    
    ###
    #args = parser.parse_args(['-i',r'C:\Users\Shahin\Documents\School\Skule\Year 4\Winter\CSC401\A1\part2.npz','-o',r'C:\Users\Shahin\Documents\School\Skule\Year 4\Winter\CSC401\A1\Output'])
    ### 
    
    # TODO: load data and split into train and test.
    with np.load(args.input) as data:
        keys = [i for i in data.iterkeys()]
        features = data[keys[0]]
    np.random.seed(0)
    train_80, test_20 = train_test_split(features, train_size = 0.8, test_size=0.2)
    train_100 = features.copy()
    #create the directory if does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # TODO : complete each classification experiment, in sequence.
    iBest      = class31(args.output_dir, train_80[:,:173], test_20[:,:173], train_80[:,173], test_20[:,173])
    X_1k, y_1k = class32(args.output_dir, train_80[:,:173], test_20[:,:173], train_80[:,173], test_20[:,173], iBest)
    class33(args.output_dir, train_80[:,:173], test_20[:,:173], train_80[:,173], test_20[:,173], iBest, X_1k, y_1k)
    class34(args.output_dir, train_100[:,:173], test_20[:,:173], train_100[:,173], test_20[:,173], iBest)
