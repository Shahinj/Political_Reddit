import numpy as np
import argparse
import json
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier  


##inspired by: https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d

stop_words_loc = '/h/u1/cs401/Wordlists/StopWords'
with open(stop_words_loc, 'r') as f:
    stop_words = f.readlines()
stop_words = [i.replace('\n','') for i in stop_words]


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
    classifiers = ['SGDClassifier', 
    'GaussianNB', 
    'RandomForestClassifier', 
    'AdaBoostClassifier',
    'SVM'
    ]
    
    best = 0
    best_acc = 0
    with open(f"{output_dir}/a1_bonus.txt", "w") as outf:
        for classifier_name in classifiers:
            if  classifier_name == 'SGDClassifier':
                classifier = SGDClassifier()
            elif classifier_name == 'GaussianNB':
                classifier = GaussianNB()
            elif classifier_name == 'RandomForestClassifier':
                classifier = RandomForestClassifier()
            elif classifier_name == 'SVM':
                classifier = LinearSVC()
            elif classifier_name == 'AdaBoostClassifier':
                classifier = AdaBoostClassifier()
            else:
                raise Exception('Classifier does not exist')
    
            
            classifier.fit(X_train, y_train)
            conf_matrix = confusion_matrix(y_test,classifier.predict(X_test))
            acc = accuracy(conf_matrix)
            rec = recall(conf_matrix)
            prec = precision(conf_matrix)
            
            print(f'Results for {classifier_name}:\n')  # Classifier name
            print(f'\tAccuracy: {acc:.4f}\n')
            print('----')
            
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



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

def plot_coefficients(output_dir,classifier, feature_names, max_features=20):
    coefs = classifier.coef_
    feature_names = np.array(feature_names)
    classes = {3: 'Alt', 1:'Center', 2:'Right', 0:'Left'}
    keywords = {}
    values = {}
    with open(f"{output_dir}/a1_bonus_features.txt", "w") as outf:
        for i in range(0,4):
            top_keywords = np.argsort(coefs[i,:])[-max_features:]
            keywords[classes[i]] = feature_names[top_keywords].tolist()
            values[classes[i]]   = coefs[i,top_keywords].tolist()
            outf.write('{0}: {1}\n'.format(classes[i], keywords[classes[i]]) )  # Classifier name
                
    # create plot
    fig, ([ax1,ax2],[ax3,ax4]) = plt.subplots(2,2,figsize=(15,15))
    axes = [ax1,ax2,ax3,ax4]
    colors = ['red','green','blue','orange']
    for i in range(0,4):
        axis = axes[i]
        axis.scatter(values[classes[i]], np.arange(1 * max_features),  color=colors[i])
        axis.set_yticks(np.arange(0.3, 0.3 + max_features), keywords[classes[i]])
        axis.set_yticklabels(keywords[classes[i]], fontdict=None, minor=True, rotation = '0', fontsize = 8, ha='right')
        axis.set_title(classes[i], color='black')
        
    plt.savefig('{0}\\features.png'.format(output_dir))
    
def svc_classifier_with_features(output_dir, X_train, X_test, y_train, y_test, feature_names):
    svm = LinearSVC()
    svm.fit(X_train, y_train)
    conf_matrix = confusion_matrix(y_test,svm.predict(X_test))
    acc = accuracy(conf_matrix)
    rec = recall(conf_matrix)
    prec = precision(conf_matrix)
    plot_coefficients(output_dir,svm, feature_names, 50)

def main(args):
    #prepare data
    data = pd.DataFrame(json.load(open(args.input)))
    classes = {'Alt': 3, 'Center': 1, 'Right': 2, 'Left': 0}
    data['Label'] = data['cat'].apply(lambda x: classes[x])
    tfidf = TfidfVectorizer(analyzer = 'word', token_pattern = '\w+(?=\/)', ngram_range = (1,3), max_features = 10000, stop_words = stop_words)

    np.random.seed(0)
    train_80, test_20 = train_test_split(data, train_size = 0.8, test_size=0.2)
    train_100 = data.copy()

    training_tfidf = tfidf.fit_transform(train_80['body'])
    test_tfidf     = tfidf.transform(test_20['body'])
    feature_names  = tfidf.get_feature_names()
    
    #do classifications
    svc_classifier_with_features(args.output, training_tfidf.todense(), test_tfidf.todense(), train_80['Label'], test_20['Label'],feature_names)
    class31(args.output, training_tfidf.todense(), test_tfidf.todense(), train_80['Label'], test_20['Label'])

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    
    
    args = parser.parse_args()       
    
    ###
    # args = parser.parse_args(['-i',r'C:\Users\Shahin\Documents\School\Skule\Year 4\Winter\CSC401\A1\preproc.json','-o',r'C:\Users\Shahin\Documents\School\Skule\Year 4\Winter\CSC401\A1\bonus.npz','--a1_dir',r'C:\Users\Shahin\Documents\School\Skule\Year 4\Winter\CSC401\A1'])
    ### 

    main(args)

