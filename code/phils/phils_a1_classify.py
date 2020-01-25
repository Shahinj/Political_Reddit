from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from scipy import stats
import numpy as np
import argparse
import sys
import os
import csv
import time

models = []
initialized = False

def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    return np.trace(C) / sum(sum(C))

def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    return np.diagonal(C) / np.sum(C, axis=1)

def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    return np.diagonal(C) / np.sum(C, axis=0)

def init_models():
    global models
    global initialized
    initialized = True
    svc_linear        = LinearSVC()
    svc_rbf           = SVC(kernel='rbf', gamma = 2, max_iter=5000)
    random_forest_clf = RandomForestClassifier(n_estimators=10, max_depth=5)
    mlp_clf           = MLPClassifier(alpha=0.05)
    adaboost_clf      = AdaBoostClassifier()
    models = [svc_linear, svc_rbf, random_forest_clf, mlp_clf, adaboost_clf]

def class31(filename):
    ''' This function performs experiment 3.1
    
    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''

    data = np.load(filename)
    X = data['arr_0'][:,:173]
    y = data['arr_0'][:,173]
    X_train, X_test, y_train, y_test, = train_test_split(X,y, test_size=0.2,
                                                         random_state=0)

    init_models()

    results = np.empty((5,26))
    results[:,0] = range(1,6)

    for i, model in enumerate(models,start=1):
        s = time.time()
        model.fit(X_train, y_train)
        # joblib.dump(model, 'model'+str(i)+'.pkl')
        C = confusion_matrix(y_test, model.predict(X_test))
        results[i-1][1] = accuracy(C)
        results[i-1][2:6] = recall(C)
        results[i-1][6:10] = precision(C)
        results[i-1][10:] = C.ravel()
        print(str(time.time() - s))

    # iBest is the index of the classifier with maximum accuracy
    iBest = np.argmax(results[:,1])
    print('Best:',iBest)

    # Save results to csv
    np.savetxt('a1_3.1.csv', results, delimiter=',')    

    return (X_train, X_test, y_train, y_test,iBest)


def class32(X_train, X_test, y_train, y_test,iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    if not initialized:
        init_models()
    model = models[iBest]
    # model = joblib.load('model'+str(iBest)+'.pkl')
    accuracies = []
    training_sizes = [1000, 5000, 10000, 15000, 20000]

    for sz in training_sizes:
        print(sz)
        model.fit(X_train[:sz], y_train[:sz])
        C = confusion_matrix(y_test, model.predict(X_test))
        accuracies.append(accuracy(C))

    np.savetxt('a1_3.2.csv', accuracies, fmt='%.5f', delimiter=',')
    X_1k, y_1k = X_train[:1000], y_train[:1000]

    return (X_1k, y_1k)
    
def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    # model = joblib.load('model'+str(i)+'.pkl')
    if not initialized:
        init_models()
    model = models[i]
    k_values = [5, 10, 20, 30, 40, 50]
    all_pvalues = []

    # 3.3.1 Compute pvalues for k features
    for k in k_values:
        selector = SelectKBest(f_classif, k)
        X_new = selector.fit_transform(X_train, y_train)
        feature_indices = selector.get_support(indices=True)
        pp = selector.pvalues_
        print(feature_indices)
        topk = sorted(pp[feature_indices])
        topk.insert(0, k)
        all_pvalues.append(topk)

    # 3.3.2 Train on k=5 best features
    selector = SelectKBest(f_classif, 5)

    # 32 k set
    X_new = selector.fit_transform(X_train, y_train)
    X_new_test = selector.transform(X_test)
    feature_indices = selector.get_support(indices=True)
    model.fit(X_new, y_train)
    C = confusion_matrix(y_test, model.predict(X_new_test))
    acc_32k = accuracy(C)
    print('32k top pvalues:',selector.pvalues_[feature_indices])
    print('32k best features:', feature_indices)
    # 1 k set
    X_new = selector.fit_transform(X_1k, y_1k)
    X_new_test = selector.transform(X_test)
    feature_indices = selector.get_support(indices=True)
    model.fit(X_new, y_1k)
    C = confusion_matrix(y_test, model.predict(X_new_test))
    acc_1k = accuracy(C)
    print('1k top pvalues:',selector.pvalues_[feature_indices])
    print('1k best features:', feature_indices)

    with open('a1_3.3.csv', 'w') as csvf:
        csvw = csv.writer(csvf, delimiter=',')
        csvw.writerows(all_pvalues)
        csvw.writerow([acc_1k, acc_32k])

def class34( filename, i ):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    if not initialized:
        init_models()
     
    data = np.load(filename)
    X = data['arr_0'][:,:173]
    y = data['arr_0'][:,173]
    kf = KFold(n_splits=5, shuffle=True, random_state=401)
    foldnum = 1
    cv_results = []

    # 5-Fold Cross Validation
    for train_idx, test_idx in kf.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        currfold_results = [] # tracks each model's accuracy for current fold
        for j, model in enumerate(models,start=1):
            s = time.time()
            model.fit(X_train, y_train)
            # joblib.dump(model, 'CVmodel'+str(j)+'fold'+str(foldnum)+'.pkl')
            C = confusion_matrix(y_test, model.predict(X_test))
            currfold_results.append(accuracy(C))
            print('Time for model', j, str(time.time() - s))

        cv_results.append(currfold_results)
        foldnum += 1

    # Compute pvalues 
    pvalues = []
    best_model_cv_results = [acc_list[i] for acc_list in cv_results]

    for model_idx in range(len(cv_results)):
        if model_idx == i:
            continue
        S = stats.ttest_rel(best_model_cv_results, [a[model_idx] for a in cv_results])
        pvalues.append(S.pvalue)

    # Save to csv
    with open('a1_3.4.csv', 'w') as csvf:
        csvw = csv.writer(csvf, delimiter=',')
        csvw.writerows(cv_results)
        csvw.writerow(pvalues)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()
    feats_file = args.input
    # Experiment 1
    print('Experiment 3.1')
    X_train, X_test, y_train, y_test, iBest = class31(feats_file)
    # Experiment 2
    print('Experiment 3.2')
    X_1k, y_1k = class32(X_train, X_test, y_train, y_test, iBest) 
    # Experiment 3
    print('Experiment 3.3')
    class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    # Experiment 4
    print('Experiment 3.4')
    class34(feats_file, iBest)
