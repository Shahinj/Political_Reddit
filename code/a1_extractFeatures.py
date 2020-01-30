import numpy as np
import argparse
import json

###
import time
import re
import csv
puncs = {'SYM',
"`","`",
"'","'",
",",
"-LRB-","-RRB-",
".", ":",
"HYPH", "NFP",
'#','$','.',',',':', '(', ')', '"', "`", "\\","'", '"'}
###

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}

NORMS_dir = '/h/u1/cs401/Wordlists/'
FEATS_dir = '/h/u1/cs401/A1/feats/'



def read_norms():
    #read the norm files
    bristol_path = NORMS_dir  + 'BristolNorms+GilhoolyLogie.csv'
    warriner_path = NORMS_dir + 'Ratings_Warriner_et_al.csv'

    global bristol_index
    global bristol_norms
    global bristol_keys
    bristol_index = {}
    bristol_norms = np.empty( (0,3) , np.float )
    with open(bristol_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        next(readCSV, None)
        for i,row in enumerate(readCSV):
            bristol_index[row[1]] = i
            newrow = [row[3],row[4],row[5]]
            bristol_norms = np.vstack([bristol_norms, newrow])  
    bristol_norms = np.nan_to_num(bristol_norms)
    bristol_keys = bristol_index.keys()
            
    global warriner_index
    global warriner_norms
    global warriner_keys
    warriner_index = {}
    warriner_norms = np.empty( (0,3) , np.float )
    with open(warriner_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        next(readCSV, None)
        for i,row in enumerate(readCSV):
            warriner_index[row[1]] = i
            newrow = [row[2],row[5],row[8]]
            warriner_norms = np.vstack([warriner_norms, newrow])
    warriner_norms = np.nan_to_num(warriner_norms)
    warriner_keys = warriner_index.keys()
    
    global ids
    global liwc
    
    alt_ids         = np.loadtxt(FEATS_dir + 'Alt_IDs.txt', str)
    alt_values      = np.load(FEATS_dir + 'Alt_feats.dat.npy')
    center_ids      = np.loadtxt(FEATS_dir + 'Center_IDs.txt', str)
    center_values   = np.load(FEATS_dir + 'Center_feats.dat.npy')
    left_ids        = np.loadtxt(FEATS_dir + 'Left_IDs.txt', str)
    left_values     = np.load(FEATS_dir + 'Left_feats.dat.npy')
    right_ids       = np.loadtxt(FEATS_dir + 'Right_IDs.txt', str)
    right_values    = np.load(FEATS_dir + 'Right_feats.dat.npy')
    
    ids  = {'Alt': alt_ids, 'Center': center_ids, 'Right': right_ids, 'Left': left_ids}
    liwc = {'Alt': alt_values, 'Center': center_values, 'Right': right_values, 'Left': left_values}
    
    return

read_norms()
    


def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''   
    # TODO: Extract features that rely on capitalization.
    #get sentences
    feats = np.zeros((1, 173))
    if(comment == ''):
        return feats
    if(comment[-1] == '\n'):
        comment = comment[:-1]

    comments = comment.split('\n')
    splitted = []
    for sent in comments:
        splitted += sent.split(' ')
    ###1 caps
    uppers = [ i[:i.rfind('/')] for i in splitted if str.isupper( i[:i.rfind('/')]) == True and len(i[:i.rfind('/')]) >= 3]
    feats[0,0] = len(uppers)
    # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
    lower = [ str.lower(  i[:i.rfind('/')]  ) +'/'+ i[i.rfind('/')+1:] for i in splitted]
    lemmas =  [i[:i.rfind('/')] for i in lower]
    pos    =  [i[i.rfind('/')+1:] for i in lower]
    # TODO: Extract features that do not rely on capitalization.
    ###2 first
    feats[0,1] = len([i for i in lemmas if i in FIRST_PERSON_PRONOUNS])
    ###3 second
    feats[0,2] = len([i for i in lemmas if i in SECOND_PERSON_PRONOUNS])
    ###4 third
    feats[0,3] = len([i for i in lemmas if i in THIRD_PERSON_PRONOUNS])
    ###5 coordinating conjunctions
    feats[0,4] = len([i for i in pos if i == 'CC'])
    ###6 past tense verbes
    feats[0,5] = len([i for i in pos if i == 'VBD'])
    ###7 future tense verbs
    feats[0,6] = len([i for i in lemmas if i in ['will'] ])   #'ll will be taken care by spacy, gonna is going to be converted to go to by spacy
    goings = [index for index,word in enumerate(lower) if 'go' in word]
    for going in goings:
        if(going+2 >= len(lower)):
            continue
        if lemmas[going+1] != 'to' or pos[going+2] != 'VB':
            continue
        else:
            feats[0,6] += 1
    ###8 commas
    feats[0,7] = len([i for i in lemmas if i == ','])
    ###9 multi-character punctuation tokens
    feats[0,8] = len([index for index,tag in enumerate(pos) if tag in puncs and len(lemmas[index]) > 1])
    ###10 common nouns
    feats[0,9] =  len([i for i in pos if i in ['NN','NNS'] ])
    ###11 proper nouns
    feats[0,10] = len([i for i in pos if i in ['NNP','NNPS'] ])
    ###12 adverbs
    feats[0,11] = len([i for i in pos if i in ['RB', 'RBR', 'RBS'] ])
    ###13 wh-words
    feats[0,12] = len([i for i in pos if i in ['WDT', 'WP', 'WP$', 'WRB'] ])
    ###14 slang acronyms
    feats[0,13] = len([i for i in lemmas if i in SLANG ])
    ###15 Average length of sentences, in tokens
    num_sent = len(comments)
    feats[0,14] = float(len(lower) / num_sent)
    ###16 Average length of tokens, excluding punctuation-only tokens, in characters
    no_puncs = [ i[:i.rfind('/')] for i in lower if  i[i.rfind('/')+1:] not in puncs]
    if(comment == "I/PRP be/VBP go/VBG to/TO admit/VB ,/, as/RB much/RB as/IN I/PRP love/VBP the/DT /r/NNP //SYM woman/NNS subreddit/NNS ,/, I/PRP want/VBP to/TO indulge/VB in/IN the/DT less/RBR serious/JJ thing/NNS about/IN be/VBG girly/RB ./."):
        print(no_puncs)
    
    if(len(no_puncs) == 0):
        feats[0,15] = 0
    else:
        feats[0,15] = float(len( ''.join(no_puncs) ) / len(no_puncs))
    ###17 # of sentences
    feats[0,16] = num_sent
    ###18,19,20 average AoA,IMG,FAM
    bristol_words = [bristol_index[i] for i in lemmas if i in bristol_keys]
    feats[0,17],feats[0,18],feats[0,19] = np.average(bristol_norms[bristol_words,:].astype(np.float), axis = 0)
    ###21,22,23 std AoA,IMG,FAM
    feats[0,20],feats[0,21],feats[0,22] = np.std(bristol_norms[bristol_words,:].astype(np.float), axis = 0)
    ###24,25,26 average V,A,D mean.sum
    warriner_words = [warriner_index[i] for i in lemmas if i in warriner_keys]
    feats[0,23],feats[0,24],feats[0,25] = np.average(warriner_norms[warriner_words,:].astype(np.float), axis = 0)
    ###27,28,29 std V,A,D mean.sum
    feats[0,26],feats[0,27],feats[0,28] = np.std(warriner_norms[warriner_words,:].astype(np.float), axis = 0)
    
    
    feats = np.nan_to_num(feats)
    return feats
    
def extract2(feats, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feats: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    '''    
    feats[29:] = liwc[comment_class][np.where(ids[comment_class] == comment_id)][0,:]
    return feats

def main(args):
    start = time.time()
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))
     
    classes = {'Alt': 3, 'Center': 1, 'Right': 2, 'Left': 0}

    # TODO: Use extract1 to find the first 29 features for each
    # data point. Add these to feats.
    # TODO: Use extract2 to copy LIWC features (features 30-173)
    # into feats. (Note that these rely on each data point's class,
    # which is why we can't add them in extract1).
    # print('TODO')
    for index,post in enumerate(data):
        e1_feats = extract1(post['body'])
        feats[index,0:-1] = e1_feats
        feats[index,:-1] = extract2(feats[index,:-1], post['cat'], post['id'])
        feats[index,173] = classes[post['cat']]

    np.savez_compressed(args.output, feats)
    print('Took {0} seconds.'.format(time.time() - start))

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    
    
    args = parser.parse_args()       
    
    ###
    #args = parser.parse_args(['-i',r'C:\Users\Shahin\Documents\School\Skule\Year 4\Winter\CSC401\A1\preproc.json','-o',r'C:\Users\Shahin\Documents\School\Skule\Year 4\Winter\CSC401\A1\part2.npz','--a1_dir',r'C:\Users\Shahin\Documents\School\Skule\Year 4\Winter\CSC401\A1'])
    ### 

    main(args)

