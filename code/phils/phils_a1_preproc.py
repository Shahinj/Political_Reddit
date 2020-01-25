import sys
import argparse
import os
import json
import html
import string
import spacy
import re
import time

# ----- GLOBAL VARIABLES DEFINED HERE ----- #
indir         = r'C:\Users\Shahin\Documents\School\Skule\Year 4\Winter\CSC401\A1\data'
stopworddir   = r'C:\Users\Shahin\Documents\School\Skule\Year 4\Winter\CSC401\A1\wordlists\StopWords'
abbrevdir     = r'C:\Users\Shahin\Documents\School\Skule\Year 4\Winter\CSC401\A1\wordlists\abbrev.english'
# indir         = '/u/cs401/A1/data'
# stopworddir   = '/u/cs401/Wordlists/StopWords'
# abbrevdir     = '/u/cs401/Wordlists/abbrev.english' 
stopwords     = []
abbreviations = []
nlp           = spacy.load('en_core_web_sm', disable=['parser','ner'])
URL_PATTERN   = re.compile(r'(http\S+?|www\S+?)(?=\.\s)|http\S+|www\S+', re.I)
STEP4_PATTERN = ''
STEP5_PATTERN = ''
STEP7_PATTERN = ''
STEP8_PATTERN = ''
STEP9_PATTERN = ''
INITIALIZED = False
# -------- END OF GLOBAL VARIABLES -------- #

def init_all_globals():
    """ Stores all the variables and patterns that will be reused for preproc1.
    """
    global stopwords, abbreviations
    global STEP4_PATTERN, STEP5_PATTERN, STEP7_PATTERN, STEP8_PATTERN, STEP9_PATTERN
    global INITIALIZED
    stopwords     = get_stopwords(stopworddir)
    abbreviations = get_abbreviations(abbrevdir)
    STEP4_PATTERN = get_step4_pattern()
    STEP5_PATTERN = get_step5_pattern()
    STEP7_PATTERN = get_step7_pattern()
    STEP8_PATTERN = get_step8_pattern()
    STEP9_PATTERN = get_step9_pattern()
    INITIALIZED   = True


def get_stopwords(fdir=r'C:\Users\Shahin\Documents\School\Skule\Year 4\Winter\CSC401\A1\wordlists\StopWords'):
    """ Gets the list of stopwords from the file.
    """
    stopwords = []
    with open(fdir, 'r') as f:
        stopwords = f.read().split()
    return stopwords


def get_abbreviations(fdir='/u/cs401/Wordlists/abbrev.english'):
    """ Gets the list of abbreviations from the file.
    """
    abbrev = []
    with open(fdir, 'r') as af:
        abbrev = af.read().split()
    other_abbrev = ['e.g.', 'E.g.', 'Lt.','i.e.']
    abbrev.extend(other_abbrev)
    return abbrev


def get_step4_pattern():
    """ Returns the compiled pattern that extracts punctuation from the comment
    by matching all number formats, abbreviations, punctuation, and words in
    separate groups.
    """
    punctuation = string.punctuation.replace("'",'')

    # Match all numbers | abbreviations | punctuation | words
    # EXPLANATION OF PATTERN re_numbers:
    # This expression matches numbers with various possible formatting
    # \b          # non-word boundary
    # (\-?        # optional negative sign
    #  \d+        # 1 or more digits
    #  (,\d+)*    # 0 or more commas followed by 1 or more digits
    #  (\.\d+     # period (decimal) followed by 1 or more digits (2)
    #  (e\d+      # 'e' exponential sign frollowed by 1 or more digits (1)
    #  )?         # optional above (1)
    #  )?         # optional above (2)
    # )\b         # non-word boundary
    re_numbers = r'\b(\-?\d+(,\d+)*(\.\d+(e\d+)?)?)\b'

    # EXPLANATION OF PATTERN re_abbrev:
    # This expression matches all abbreviations using the | operator
    re_abbrev = '('+r'|'.join(map(re.escape, abbreviations))+')'

    # EXPLANATION OF PATTERN re_quotes:
    # This expression matches single apostrophes that act as quotation marks
    # around a character sequence.
    re_quotes = r'(\'|\")\S+(\1)(?!\w+)'

    # EXPLANATION OF PATTERN re_punctuation:
    # This expression matches all punctuation excluding apostrophes except
    # for when they follow another punctuation.
    re_punctuation = '([' + re.escape(punctuation) + ']+\'*)'

    # EXPLANATION OF PATTERN re_words:
    # This expression matches all remaining words with apostrophes and hyphens
    # contained.
    # \b          # word boundary
    # ([          # begin character set containing
    #   \w\'      # word characters, apostrophes
    #  ]+         # match 1 or more of the character set
    # )\b         # word boundary
    re_words = r"([A-Za-z0-9\']+)"

    pattern = re.compile('|'.join([re_numbers,re_abbrev,
                                   re_punctuation,re_quotes,
                                   re_words]), re.I)
    return pattern


def get_step5_pattern():
    """ Returns the compiled pattern that matches all clitics for step 5.
    """
    clitics = ("'s","n't", "'m", "'re", "'ve", "'d", "'ll", "y'", "s'")
    return re.compile("|".join(clitics))


def get_step7_pattern():
    """ Returns the compiled pattern for step 7, which matches the spacy tagged
    stopwords.
    """
    # EXPLANATION OF PATTERN expr1:
    # (?:         # non capture group
    #  ^          # start of line
    # |(?<=       # or positive lookbehind: preceded by
    #   \s        # whitespace character
    # ))          # end of positive lookbehind and non capture group
    # (           # start of stopwords 
    expr1 = r"(?:^|(?<=\s))("

    # EXPLANATION OF PATTERN expr2:
    #  \/\S+      # forward slash followed by non-whitespace character
    # )           # end of stopwords
    # (?=         # positive lookahead: followed by
    #  \s         # whitespace character
    #  |$         # or end of line
    # )           # end of positive lookahead
    expr2 = r"\/\S+)(?=\s|$)"
    expr = [expr1 + s + expr2 for s in stopwords]
    return re.compile('|'.join(expr), flags=re.I)

def get_step8_pattern():
    """ Returns the compiled pattern for step 8/10, which gets the token and
    pos tag. Also matches optional presence of '\n' character after each tag.
    """
    # EXPLANATION OF STEP8_PATTERN:
    # (\S+)       # token group: 1 or more non-whitespace char
    # \/          # forward slash
    # (\S+        # start POS group: 1 or more non-whitespace char
    #  (?:        # non-capture group of
    #   \s\n      # whitespace followed by newline
    #  )?         # captures 1 or 0 or the preceding
    # )           # end of POS group
    return re.compile(r"(\S+)\/(\S+(?:\s\n)?)")

def get_step9_pattern():
    """ Returns the compiled pattern for step 9, which matches sentence endings
    .,!,? and .", !", ?".
    """
    # EXPLANATION OF STEP9_PATTERN
    # (\S+        # non-whitespace
    # [\.\!\?]    # sentence-final punctuation
    # [\'\"]      # quotation following sentence-final punctuation
    # \/\S+)      # rest of tag
    # |           # OR
    # (\S+\/\.)   # any token tagged with the sentence-final POS '.'
    return re.compile(r"(\S+[\.\!\?]+[\'\"]\/\S+|\S+\/\.)(?!$)")

def extract_first_group(M):
    """ Given a tuple of matches returned by re.findall, returns the first
    group match. Used for the preprocessing step 4.

    Parameters:
        M : list or tuple of strings

    Returns: first non-empty string
    """
    print(M)
    for m in M:
        if m != "":
            return m
    return ""


def extract_clitic(m):
    """ Takes in the re clitic match and returns the clitic string with
    whitespace inserted at the beginning or end depending on the clitic.
    Used for the preprocessing step 5.

    Parameters:
        m : re Match object

    Returns: string, the clitic with whitespace
    """
    if m.group(0) == "y'": # as in y'all -> y' all
        return m.group(0) + " "
    if m.group(0) == "s'": # as in dogs' -> dogs '
        return "s" + " '"
    return " " + m.group(0)


def split_tagged_tokens(modComm):
    """ Takes in the modified, assumed POS tagged comment and returns the
    tokens and POS in separate lists.

    Parameters:
        modComm: string, the modified comment from earlier steps

    Returns:
        tokens: list of strings, the list of tokens
        pos: list of strings, the list of POS corresponding to tokens
    """
    matches = re.findall(STEP8_PATTERN, modComm)
    tokens = [m[0] for m in matches]
    pos = [m[1] for m in matches]
    return tokens, pos


def sentence_newliner(m):
    """ Returns matched string with a new line character added at end.

    Parameters:
        m    : re Match object

    Returns:
        expr : string
    """
    return m.group(0) + " \n"

def preproc1( comment , steps=range(1,11)):
    ''' This function pre-processes a single commentQuickie


    Parameters:
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    modComm = comment

    if not INITIALIZED:
        print("not initialized")
        init_all_globals()

    if 1 in steps:
        # Remove all newline characters
        # print("___STEP 1___")
        # print("___OLD STR___")
        # print(comment)
        modComm = comment.replace('\n', '')
        # print("___NEW STR___")
        # print(modComm)

    if 2 in steps:
        # Replace HTML character codes (i.e., &...;) with their ASCII equivalent
        # print("___STEP 2___")
        # print("___OLD STR___")
        # print(modComm)
        modComm = html.unescape(modComm)
        # print("___NEW STR___")
        # print(modComm)

    if 3 in steps:
        # Remove all URLs (i.e., tokens beginning with http or www).
        # print("___STEP 3___")
        # print("___OLD STR___")
        # print(modComm)
        modComm = re.sub(URL_PATTERN, ' ', modComm)
        # print("___NEW STR___")
        # print(modComm)

    if 4 in steps:
        #  SPLIT EACH PUNCTUATION INTO ITS OWN TOKEN WITH WHITESPACE
        # print("___STEP 4___")
        # print("___OLD STR___")
        # print(modComm)
        matches = re.findall(STEP4_PATTERN, modComm)
        modComm = " ".join([extract_first_group(m) for m in matches])
        # print("___NEW STR___")
        # print(modComm)

    if 5 in steps:
        # SPLIT CLITICS
        # print("___STEP 5___")
        # print("___OLD STR___")
        # print(modComm)
        modComm = re.sub(STEP5_PATTERN, extract_clitic, modComm)
        # print("___NEW STR___")
        # print(modComm)

    if 6 in steps:
        # TAG EACH TOKEN WITH ITS POS WITH "/POS"
        # print("___STEP 6___")
        # print("___OLD STR___")
        # print(modComm)
        doc = spacy.tokens.Doc(nlp.vocab, words=modComm.split())
        doc = nlp.tagger(doc)
        modComm =  " ".join(["/".join((t.text, t.tag_)) for t in doc])
        # print("___NEW STR___")
        # print(modComm)

    if 7 in steps:
        # REMOVE STOPWORDS
        # print("___STEP 7___")
        # print("___OLD STR___")
        # print(modComm)
        modComm = re.sub(STEP7_PATTERN, '', modComm)
        modComm = " ".join(modComm.split()) # remove excess whitespace
        # print("___NEW STR___")
        # print(modComm)

    if 8 in steps:
        # REPLACE TOKENS WITH LEMMAS
        # print("___STEP 8____")
        # print("___OLD STR___")
        # print(modComm)
        tokens, pos = split_tagged_tokens(modComm)
        doc = spacy.tokens.Doc(nlp.vocab, words=tokens)
        doc = nlp.tagger(doc)
        lemmas = [token.lemma_ for token in doc]
        newComm = ""

        for lemma, token, part in zip(lemmas, tokens, pos):
            if lemma[0] == '-':
                newComm += "".join((token, "/", part)) + " "
            else:
                newComm += "".join((lemma, "/", part)) + " "

        modComm = newComm.rstrip()
        # print("___NEW STR___")
        # print(modComm)

    if 9 in steps:
        # ADD NEW LINE BETWEEN SENTENCES
        # print("___STEP 9____")
        # print("___OLD STR___")
        # print(modComm)
        modComm = re.sub(STEP9_PATTERN, sentence_newliner, modComm)
        # print("___NEW STR___")
        # print(modComm)

    if 10 in steps:
        # MAKE ALL TOKENS LOWERCASE
        # print("___STEP 10___")
        # print("___OLD STR___")
        # print(modComm)
        tokens, pos = split_tagged_tokens(modComm)
        tokens = map(str.lower, tokens)
        modComm = " ".join([token+"/"+part for token, part in zip(tokens,pos)])
        # print("___NEW STR___")
        # print(modComm)

    return modComm

def main( args ):
    start = time.time()
    allOutput = []
    outkeys = ['id','body','cat']
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)
            data = json.load(open(fullFile))
            start_idx = args.ID[0] % len(data)
            end_idx = start_idx + int(args.max)
            for i in range(start_idx, end_idx):
                print("______PROCESSING LINE_______", i%len(data), "FILE " + fullFile )
                line = data[i%len(data)] # circular indexing
                j = json.loads(line)
                j['cat'] = file
                # print("______BEFORE PREPROC________")
                # print(j['body'])
                j['body'] = preproc1(j['body'])
                # print("______AFTER  PREPROC________")
                # print(j['body'])
                allOutput.append({k: j[k] for k in outkeys})
    end = time.time()
    print("Total time for " + str(args.max) + " lines:", end-start)

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000, type=int)
    # args = parser.parse_args()
    
    ###
    args = parser.parse_args(['1002401634','-o',r'C:\Users\Shahin\Documents\School\Skule\Year 4\Winter\CSC401\A1\code\phils\preproc.json'])
    ###
    
    if (args.max > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)
    main(args)
