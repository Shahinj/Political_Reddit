from __future__ import division
import numpy as np
import sys
import argparse
import os
import json
import re
import string
import csv

######## GLOBAL VARIABLES/CONSTANTS #########
# bristol_dir     = '/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv'
# warringer_dir   = '/u/cs401/Wordlists/Ratings_Warriner_et_al.csv'
# feats_dir       = '/u/cs401/A1/feats/'
bristol_dir   = '/home/philipqlu/courses/csc401/a1/code/Wordlists/BristolNorms+GilhoolyLogie.csv'
warringer_dir = '/home/philipqlu/courses/csc401/a1/code/Wordlists/Ratings_Warriner_et_al.csv'
feats_dir     = '/home/philipqlu/courses/csc401/a1/feats/'
feats_file_dirs = {cat:(feats_dir + cat + '_feats.dat.npy', feats_dir+cat+'_IDs.txt') \
                   for cat in ['Alt','Center', 'Left', 'Right']}
# wordlists_dir = '/u/cs401/Wordlists/'
wordlists_dir = 'Wordlists/'

CLASS_DICT      = {'Left': 0, 'Center': 1, 'Right': 2, 'Alt': 3}
FIRST_PRONOUNS  = ['I', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
SECOND_PRONOUNS = ['you', 'your', 'yours', 'u', 'ur', 'urs']
THIRD_PRONOUNS  = ['he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its',
                  'they', 'them', 'their', 'theirs']
SLANG_ACROS =    ['smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf',
               'bff', 'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw',
               'bw', 'imho', 'fyi', 'ppl', 'sob', 'ttyl', 'imo', 'ltr',
               'thx', 'kk', 'omg', 'omfg', 'ttys', 'afn', 'bbs', 'cya', 'ez',
               'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya', 'nm', 'np','plz',
               'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml']

punctuation = string.punctuation
FIRST_PRONOUN_PATTERN    = ''
SECOND_PRONOUN_PATTERN   = ''
THIRD_PRONOUN_PATTERN    = ''
COOR_CONJUNCTION_PATTERN = ''
PASTTENSE_VB_PATTERN     = ''
FUTURE_VB_PATTERN        = ''
COMMA_PATTERN            = ''
MULTICHAR_PUNC_PATTERN   = ''
COMMON_NOUN_PATTERN      = ''
PROPER_NOUN_PATTERN      = ''
ADVERB_PATTERN           = ''
WH_WORD_PATTERN          = ''
SLANG_ACROS_PATTERN      = ''
UPPERCASE_PATTERN        = ''
ANY_TOKEN_PATTERN        = ''
NONPUNC_TOKEN_PATTERN    = ''
EXTRACT_TOKEN_PATTERN    = re.compile(r'(?i)(?:(?<=\s)|(?<=^))(\w+)(?=\/\S+)')
INITIALIZED = False

BRISTOL_DATA   = {}
WARRINGER_DATA = {}

FEATURE_IDS    = {}
LIWC_FEATURES  = {}

##### END OF GLOBAL VARIABLES/CONSTANTS #####

def check_wordlists(fdir):
    ''' Checks if we're missing any of the words for features that is in
    Wordlists and updates the lists accordingly for each feature.
    '''
    global FIRST_PRONOUN_PATTERN
    global SECOND_PRONOUN_PATTERN
    global THIRD_PRONOUN_PATTERN
    global SLANG_ACROS

    with open(fdir+'First-person', 'r') as f:
        words = f.read().split()
        FIRST_PRONOUNS.extend([word for word in words if word not in FIRST_PRONOUNS])
    with open(fdir+'Second-person', 'r') as f:
        words = f.read().split()
        SECOND_PRONOUNS.extend([word for word in words if word not in SECOND_PRONOUNS])
    with open(fdir+'Third-person', 'r') as f:
        words = f.read().split()
        THIRD_PRONOUNS.extend([word for word in words if word not in THIRD_PRONOUNS])
    for fname in ['Slang','Slang2']:
        with open(fdir+fname, 'r') as f:
            words = f.read().split()
            SLANG_ACROS.extend([word for word in words if word not in SLANG_ACROS])

def download_feats_data():
    ''' Reads the feature id files and the feats data.npy files from disk.
    Stores the data into dictionaries with key-value pairs as follows:


        BRISTOL_DATA:
            key = string, the word in the norms list
            value = dictionary of (string:int) -> (norm:norm_value)

        WARRINGER_DATA:
            key = string, the word in the norms list
            value = dictionary of (string:float) -> (norm:norm_value)

        FEATURE_IDS:
            key = string, one of the 4 classes
            value = list of comment ids

        LIWC_FEATURES:
            key = string, one of the 4 classes
            value = numpy array (Nx144) where N is number of comments in class
    '''
    global WARRINGER_DATA
    global BRISTOL_DATA
    global LIWC_FEATURES
    global FEATURE_IDS

    # Bristol et al
    with open(bristol_dir, 'r') as f:
        reader = csv.DictReader(f)
        BRISTOL_DATA = {row['WORD']: {'aoa': int(row['AoA (100-700)']),
                                      'img': int(row['IMG']),
                                      'fam': int(row['FAM'])} for row in reader \
                                       if row['WORD'] != ''}
    # Warringer
    with open(warringer_dir, 'r') as f:
        reader = csv.DictReader(f)
        WARRINGER_DATA = {row['Word']: {'vmean': float(row['V.Mean.Sum']),
                                        'amean': float(row['A.Mean.Sum']),
                                        'dmean': float(row['D.Mean.Sum'])} for row in reader}
    # LIWC features
    for cat, files in feats_file_dirs.items():
        # read ID file
        with open(files[1], 'r') as f:
            FEATURE_IDS[cat] = f.read().split()
        # load npy file
        LIWC_FEATURES[cat] = np.load(files[0])


def compile_all_patterns():
    ''' Compiles all of the patterns for counting the occurrences for features
    1 to 17.
    '''
    global FIRST_PRONOUN_PATTERN
    global SECOND_PRONOUN_PATTERN
    global THIRD_PRONOUN_PATTERN
    global COOR_CONJUNCTION_PATTERN
    global PASTTENSE_VB_PATTERN
    global FUTURE_VB_PATTERN
    global COMMA_PATTERN
    global MULTICHAR_PUNC_PATTERN
    global COMMON_NOUN_PATTERN
    global PROPER_NOUN_PATTERN
    global ADVERB_PATTERN
    global WH_WORD_PATTERN
    global SLANG_ACROS_PATTERN
    global UPPERCASE_PATTERN
    global ANY_TOKEN_PATTERN
    global NONPUNC_TOKEN_PATTERN

    FIRST_PRONOUN_PATTERN    = compile_token_pattern(FIRST_PRONOUNS)
    SECOND_PRONOUN_PATTERN   = compile_token_pattern(SECOND_PRONOUNS)
    THIRD_PRONOUN_PATTERN    = compile_token_pattern(THIRD_PRONOUNS)
    COOR_CONJUNCTION_PATTERN = compile_POS_pattern(['CC'])
    PASTTENSE_VB_PATTERN     = compile_POS_pattern(['VBP'])
    FUTURE_VB_PATTERN        = compile_future_tense_pattern()
    COMMA_PATTERN            = compile_POS_pattern([','])
    MULTICHAR_PUNC_PATTERN   = compile_token_pattern(['['+re.escape(punctuation)+']{2,}'])
    COMMON_NOUN_PATTERN      = compile_POS_pattern(['NN','NNS'])
    PROPER_NOUN_PATTERN      = compile_POS_pattern(['NNP','NNPS'])
    ADVERB_PATTERN           = compile_POS_pattern(['RB', 'RBR', 'RBS'])
    WH_WORD_PATTERN          = compile_POS_pattern(['WDT', 'WP', 'WP\$', 'WRB'])
    SLANG_ACROS_PATTERN      = compile_token_pattern(SLANG_ACROS)
    UPPERCASE_PATTERN        = compile_token_pattern([r'[A-Z]{3,}'])
    ANY_TOKEN_PATTERN        = compile_token_pattern([r'\S+'])
    NONPUNC_TOKEN_PATTERN    = compile_token_pattern([r'[\S]*[A-Za-z0-9]+[\S]*'])

def initialize_all_globals():
    ''' Initializes all the global variables i.e. compiling all reusable
    regex patterns and reading in the feats data files.
    '''
    global INITIALIZED
    INITIALIZED = True
    check_wordlists(wordlists_dir)
    compile_all_patterns()
    download_feats_data()

##### regex PATTERN COMPILATION FUNCTIONS #####
def compile_token_pattern(tokens):
    ''' Returns compiled regex pattern matching all occurrences of token
    in the comment.
    '''
    with open('part2_patterns.txt', 'a') as f:
        f.write(r'((?<=\s)|(?<=^))('+'|'.join(tokens)+r')\/\S+'+'\n')
    return re.compile(r'((?<=\s)|(?<=^))('+'|'.join(tokens)+r')\/\S+', flags=re.I)

def compile_future_tense_pattern():
    ''' Returns compiled regex pattern for future tense verbs 'll, will,
    gonna, and going+to+VB.
    '''
    p = r"((?<=\s)|(?<=^))(\'ll\/\w+|will\/\w+|gonna\/\w+|going\/\w+\s+to\/\w+)\s\S+\/VB\b"
    return re.compile(p, flags=re.I)

def compile_POS_pattern(pos_list):
    ''' Returns compiled regex pattern matching all occurrences of POS
    in the tagged comment.
    '''
    return re.compile(r'\S+\/(' + '|'.join(pos_list) + r')(?=\s)')

#### END regex PATTERN COMPLETION FUNCTIONS ####

######### FEATURE EXTRACTION FUNCTIONS #########
def count_pattern_matches(comment, pattern):
    ''' Returns the number of matches for the pattern in comment.

    Parameters:
        comment: string, the tagged comment
        pattern: re Pattern object, the compiled pattern to match

    Returns: int, number of occurrences of pattern
    '''
    return len(re.findall(pattern, comment))

def average_token_length(comment, tokens):
    ''' Returns the average token length for non-punctuation only tokens.

    Parameters:
        comment: string, the tagged comment
        pattern: re Pattern object, the compiled pattern to match

    Returns: float, the average length of a token in the comment
    '''

    total_length = sum([len(s) for s in tokens])
    if len(tokens) > 0:
        return total_length / len(tokens)
    return 0

def bristol_stats(tokens):
    ''' Computes and returns the Bristol et al. stats for features 18-23. Only
    considers the words in the comment that are in the norms file.

    Parameters:
        tokens: list-like object, contains the word tokens from the comment

    Returns:
        stats : numpy array, contains feature values 18 to 23 for the comment.
    '''
    aoa_values = []
    img_values = []
    fam_values = []
    stats = np.zeros((6,))
    for token in tokens:
        if token in BRISTOL_DATA.keys():
            aoa_values.append(BRISTOL_DATA[token]['aoa'])
            img_values.append(BRISTOL_DATA[token]['img'])
            fam_values.append(BRISTOL_DATA[token]['fam'])

    if len(aoa_values) != 0:
        stats = np.array([np.mean(aoa_values), np.mean(img_values),
                          np.mean(fam_values), np.std(aoa_values),
                          np.std(img_values), np.std(fam_values)])
    return stats

def warringer_stats(tokens):
    ''' Computes and returns the Warringer stats for features 24-29. Only
    considers the words in the comment that are in the norms file.

    Parameters:
        tokens: list-like object, contains the word tokens from the comment

    Returns:
        stats : numpy array, contains feature values 24 to 29 for the comment.
    '''
    vmean_values = []
    amean_values = []
    dmean_values = []
    stats = np.zeros((6,))
    for token in tokens:
        if token in BRISTOL_DATA.keys():
            vmean_values.append(WARRINGER_DATA[token]['vmean'])
            amean_values.append(WARRINGER_DATA[token]['amean'])
            dmean_values.append(WARRINGER_DATA[token]['dmean'])

    if len(vmean_values) != 0:
        stats = np.array([np.mean(vmean_values), np.mean(amean_values),
                          np.mean(dmean_values), np.std(vmean_values),
                          np.std(amean_values), np.std(dmean_values)])
    return stats

def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features
    '''
    feats = np.zeros((29,))
    if not INITIALIZED:
        print('Initializing')
        initialize_all_globals()

    feats[0]  = count_pattern_matches(comment, FIRST_PRONOUN_PATTERN)
    feats[1]  = count_pattern_matches(comment, SECOND_PRONOUN_PATTERN)
    feats[2]  = count_pattern_matches(comment, THIRD_PRONOUN_PATTERN)
    feats[3]  = count_pattern_matches(comment, COOR_CONJUNCTION_PATTERN)
    feats[4]  = count_pattern_matches(comment, PASTTENSE_VB_PATTERN)
    feats[5]  = count_pattern_matches(comment, FUTURE_VB_PATTERN)
    feats[6]  = count_pattern_matches(comment, COMMA_PATTERN)
    feats[7]  = count_pattern_matches(comment, MULTICHAR_PUNC_PATTERN)
    feats[8]  = count_pattern_matches(comment, COMMON_NOUN_PATTERN)
    feats[9]  = count_pattern_matches(comment, PROPER_NOUN_PATTERN)
    feats[10] = count_pattern_matches(comment, ADVERB_PATTERN)
    feats[11] = count_pattern_matches(comment, WH_WORD_PATTERN)
    feats[12] = count_pattern_matches(comment, SLANG_ACROS_PATTERN)
    feats[13] = count_pattern_matches(comment, UPPERCASE_PATTERN)
    num_sentences = count_pattern_matches(comment, "\\n") + 1 # +1 cause no \n after last sentence
    feats[14] = count_pattern_matches(comment, ANY_TOKEN_PATTERN) / num_sentences
    feats[15] = average_token_length(comment, re.findall(NONPUNC_TOKEN_PATTERN,
                                                         comment))
    feats[16] = count_pattern_matches(comment, "\\n")
    tokens = map(str.lower, re.findall(EXTRACT_TOKEN_PATTERN, comment))
    feats[17:23] = bristol_stats(tokens)
    feats[23:29] = warringer_stats(tokens)
    return feats

def main( args ):

    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 173+1))

    for i in range(len(data)):
        print("Extracting line:", i+1, "Cat: " + data[i]['cat'])
        feats[i][:29] = extract1(data[i]['body'])
        id_ = FEATURE_IDS[data[i]['cat']].index(data[i]['id'])
        feats[i][29:173] = LIWC_FEATURES[data[i]['cat']][id_]
        feats[i][173] = CLASS_DICT[data[i]['cat']]

    np.savez_compressed( args.output, feats)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()


    main(args)

