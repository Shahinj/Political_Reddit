import sys
import argparse
import os
import json
import re
import spacy
import time

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

###
import html
###

def preproc1(comment , steps=range(1, 5)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    modComm = comment
    if 1 in steps:  # replace newlines with spaces
        modComm = re.sub(r"\n{1,}", " ", modComm)
    if 2 in steps:  # unescape html
        modComm = html.unescape(modComm)            # TODO
    if 3 in steps:  # remove URLs
        modComm = re.sub(r"(http|http|www)\S+", "", modComm)
    if 4 in steps:  # remove duplicate spaces
        #strip initial and final spaces if exist
        modComm = modComm.strip()
        modComm = re.sub(r'\s+',' ',modComm)        # TODO

    # TODO: get Spacy document for modComm

    utt = nlp(modComm)
    cleaned = ''
    #loop through all sentences
    for sent in utt.sents:
        #loop through tokens
        for i,token in enumerate(sent):
            #get lemma and tag
            lemma = token.lemma_
            tag = token.tag_
            #use token if lemma starts with - and token does not start with -
            if(lemma[0] == '-' and token.text[0] != '-'):
                cleaned += token.text + '/' + tag
            else:
                #else use the lemma
                cleaned += lemma + '/' + tag

            #check if the lemma and tag were the last elements of the sentence, if not, add space, if yes, add new line character to the end.
            if (i != len(sent) - 1):
                cleaned += ' '
            else:
                cleaned += '\n'
    
    modComm = cleaned[:]

    return modComm


def main(args):
    start = time.time()
    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))
            counter = 0
            #read until the max specified
            while(counter != args.max):
                #get the sampling index using the formula in assignment sheet
                index = (args.ID[0] + counter) % len(data)
                #load the data of index
                post = json.loads(data[index])
                #create the required format and it to the allOutput
                to_add = {'id': post['id'], 'body': preproc1(post['body']) , 'cat': file}
                allOutput.append(to_add)
                counter += 1
            
    #save the results to the file specified
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()
    print('Took {0} seconds.'.format(time.time() - start))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    parser.add_argument("--a1_dir", help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.", default='/u/cs401/A1')
    
    args = parser.parse_args()
    
    ###
    #args = parser.parse_args(['1002401634','-o',r'C:\Users\Shahin\Documents\School\Skule\Year 4\Winter\CSC401\A1\preproc.json','--a1_dir',r'C:\Users\Shahin\Documents\School\Skule\Year 4\Winter\CSC401\A1'])
    ###


    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)
    
    indir = os.path.join(args.a1_dir, 'data')
    main(args)
