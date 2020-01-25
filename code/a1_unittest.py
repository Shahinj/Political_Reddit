import unittest
import os
#os.chdir(r'C:\Users\Shahin\Documents\School\Skule\Year 4\Winter\CSC401\A1\code')
#a1_dir = r'C:\Users\Shahin\Documents\School\Skule\Year 4\Winter\CSC401\A1'

#os.chdir(r'C:\Users\Shahin\Documents\School\Skule\Year 4\Winter\CSC401\A1\code')
a1_dir = r'/h/u1/cs401/A1'

from a1_preproc import *
from a1_extractFeatures import *

try:
    liwc
except:
    read_norms(a1_dir)


class TestStringMethods(unittest.TestCase):
    
    ###preprocessing
    def test_newline(self):
        comment = '\n'
        actual = preproc1(comment)
        expected = ''
        self.assertEqual(actual, expected)
        
        comment = '\n\n\n'
        actual = preproc1(comment)
        expected = ''
        self.assertEqual(actual, expected)
                
                
        comment = '\n \n'
        actual = preproc1(comment)
        expected = ''
        self.assertEqual(actual, expected)
        
        comment = '\n.'
        actual = preproc1(comment)
        expected = './.\n'
        self.assertEqual(actual, expected)
        
        comment = '\n.\n.\n.'
        actual = preproc1(comment)
        expected = './. ./. ./.\n'
        self.assertEqual(actual, expected)
        
        comment = 'Hi.\n Hi.\n'
        actual = preproc1(comment)
        expected = 'hi/UH ./.\nhi/UH ./.\n'
        self.assertEqual(actual, expected)
        
        comment = '\nHi.\n Hi.\n'
        actual = preproc1(comment)
        expected = 'hi/UH ./.\nhi/UH ./.\n'
        self.assertEqual(actual, expected)
        
        comment = '\r'
        actual = preproc1(comment)
        expected = ''
        self.assertEqual(actual, expected)

        comment = '\rHi.\r Hi.\r'
        actual = preproc1(comment)
        expected = 'hi/UH ./.\nhi/UH ./.\n'
        self.assertEqual(actual, expected)

    def test_unescape(self):
        
        comment = '&#36;'
        actual = preproc1(comment)
        expected = '$/$\n'
        self.assertEqual(actual, expected)
        
        comment = '&#36;&#36;&#36;'
        actual = preproc1(comment)
        expected = '$/$ $/$ $/$\n'
        self.assertEqual(actual, expected)
        
    def test_duplicate_spaces(self):
        
        comment = '    '
        actual = preproc1(comment)
        expected = ''
        self.assertEqual(actual, expected)
        
        comment = '    &#36;     .'
        actual = preproc1(comment)
        expected = '$/$ ./.\n'
        self.assertEqual(actual, expected)
        
        comment = '  Hi      &#36;     .'
        actual = preproc1(comment)
        expected = 'hi/UH $/$ ./.\n'
        self.assertEqual(actual, expected)
        
        
    def test_newline_sentence(self):
        
        comment = "Hi. I'm Shahin."
        actual = preproc1(comment)
        expected = 'hi/UH ./.\nI/PRP be/VBP Shahin/NNP ./.\n'
        self.assertEqual(actual, expected)

    ###features
    def test_uppercase(self):
        feature = 1
        comment = "Hi/STH SHAHIN/STH ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 1
        self.assertEqual(actual, expected)
        
        comment = "Hi/STH SHA/STH ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 1
        self.assertEqual(actual, expected)
        
        comment = "Hi/STH SH/STH ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 0
        self.assertEqual(actual, expected)
        
        
        comment = "Hi/STH SHaHIN/STH JAFARI/STH ALI/STH ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 2
        self.assertEqual(actual, expected)
    
    def test_fpp(self):
        feature = 2
        comment = "i/STH am/STH us/STH ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 2
        self.assertEqual(actual, expected)
        
        comment = "Hi/STH SHA/STH ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 0
        self.assertEqual(actual, expected)
        
    def test_spp(self):
        feature = 3
        comment = "u/STH are/STH you/STH ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 2
        self.assertEqual(actual, expected)
        
        comment = "Hi/STH SHA/STH ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 0
        self.assertEqual(actual, expected)
        
        
    def test_thpp(self):
        feature = 4
        comment = "hi/STH he/STH her/STH you/STH ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 2
        self.assertEqual(actual, expected)
        
        comment = "Hi/STH SHA/STH ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 0
        self.assertEqual(actual, expected)
        
    def test_coordinating_conj(self):
        feature = 5
        comment = "hi/STH he/CC her/CC you/STH ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 2
        self.assertEqual(actual, expected)
        
        comment = "Hi/STH SHA/STH ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 0
        self.assertEqual(actual, expected)
        
    def test_past_verbs(self):
        feature = 6
        comment = "hi/VBD he/VBD her/CC you/STH ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 2
        self.assertEqual(actual, expected)
        
        comment = "Hi/STH SHA/STH ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 0
        self.assertEqual(actual, expected)
        
    def test_future_verbs(self):
        feature = 7
        comment = preproc1("I will eat tomorrow")
        actual = extract1(comment)[0,feature-1]
        expected = 1
        self.assertEqual(actual, expected)
        
        comment = preproc1("I will not eat tomorrow")
        actual = extract1(comment)[0,feature-1]
        expected = 1
        self.assertEqual(actual, expected)
        
        comment = preproc1("I'll eat tomorrow")
        actual = extract1(comment)[0,feature-1]
        expected = 1
        self.assertEqual(actual, expected)
        
        comment = preproc1("I'm going to eat tomorrow")
        actual = extract1(comment)[0,feature-1]
        expected = 1
        self.assertEqual(actual, expected)
        
        comment = preproc1("I am going to eat tomorrow")
        actual = extract1(comment)[0,feature-1]
        expected = 1
        self.assertEqual(actual, expected)
        
        comment = preproc1("I'm gonna eat tomorrow and clean the house")
        actual = extract1(comment)[0,feature-1]
        expected = 1
        self.assertEqual(actual, expected)
        
        comment = "Hi/STH SHA/STH ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 0
        self.assertEqual(actual, expected)
        
        comment = preproc1("I'm gonna eat tomorrow and I'm not going to clean the house")
        actual = extract1(comment)[0,feature-1]
        expected = 2
        self.assertEqual(actual, expected)
        
    def test_commas(self):
        feature = 8
        comment = preproc1("I will eat tomorrow, not the day after, not yesterday.")
        actual = extract1(comment)[0,feature-1]
        expected = 2
        self.assertEqual(actual, expected)
        
        comment = preproc1(",,,,")
        actual = extract1(comment)[0,feature-1]
        expected = 4
        self.assertEqual(actual, expected)

    def test_multi_punctuation(self):
        feature = 9
        comment = preproc1("I will eat tomorrow!!!!")   #since it splits, should not be counted
        actual = extract1(comment)[0,feature-1]
        expected = 0
        self.assertEqual(actual, expected)
        
        comment = preproc1("Hi.....")
        actual = extract1(comment)[0,feature-1]
        expected = 1
        self.assertEqual(actual, expected)
        
        comment = preproc1("No ... Hi :))")
        actual = extract1(comment)[0,feature-1]
        expected = 2
        self.assertEqual(actual, expected)

        comment = preproc1("No ...... Hi :))")
        actual = extract1(comment)[0,feature-1]
        expected = 2
        self.assertEqual(actual, expected)
        
        
    def test_common_nouns(self):
        feature = 10
        comment = "hi/VBD he/NN her/PO you/NNI ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 1
        self.assertEqual(actual, expected)
        
        comment = "hi/VBD he/NI her/PO you/NNS ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 1
        self.assertEqual(actual, expected)
        
        
        comment = "hi/VBD he/NN her/NN you/NNS ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 3
        self.assertEqual(actual, expected)
        
        comment = "hi/VBD he/VBD her/CC you/STH ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 0
        self.assertEqual(actual, expected)
        
    def test_proper_nouns(self):
        feature = 11
        comment = "hi/VBD he/NNP her/PO you/NN ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 1
        self.assertEqual(actual, expected)
        
        comment = "hi/VBD he/NNPS her/PO you/NN ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 1
        self.assertEqual(actual, expected)
        
        comment = "hi/VBD he/VBD her/CC you/STH ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 0
        self.assertEqual(actual, expected)
        
        
    def test_adverbs(self):
        feature = 12
        comment = "hi/VBD he/RB her/STH you/NN ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 1
        self.assertEqual(actual, expected)
        
        comment = "hi/VBD he/RBR her/STH you/NN ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 1
        self.assertEqual(actual, expected)
        
        comment = "hi/VBD he/RBS her/STH you/NN ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 1
        self.assertEqual(actual, expected)
        
        comment = "hi/VBD he/RBR her/RBS you/RB ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 3
        self.assertEqual(actual, expected)
        
        comment = "hi/VBD he/STH her/STH you/NN ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 0
        self.assertEqual(actual, expected)
        
        
    def test_wh(self):
        feature = 13
        comment = "hi/VBD he/WDT her/STH you/NN ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 1
        self.assertEqual(actual, expected)
        
        comment = "hi/VBD he/WP her/STH you/NN ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 1
        self.assertEqual(actual, expected)
        
        comment = "hi/VBD he/WP$ her/STH you/NN ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 1
        self.assertEqual(actual, expected)
        
        comment = "hi/VBD he/WRB her/RBS you/RB ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 1
        self.assertEqual(actual, expected)
        
        comment = "hi/WDT he/WRB her/WP you/WP$ ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 4
        self.assertEqual(actual, expected)
        
        comment = "hi/VBD he/STH her/STH you/NN ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 0
        self.assertEqual(actual, expected)
            
            
    def test_slang(self):
        feature = 14
        for i in SLANG:
            comment = "{0}/STH he/WDT her/STH you/NN ./STH".format(i)
            actual = extract1(comment)[0,feature-1]
            expected = 1
            self.assertEqual(actual, expected)
            
        comment = "hi/VBD he/WP her/STH you/NN ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 0
        self.assertEqual(actual, expected)
        
    def test_slang(self):
        feature = 14
        for i in SLANG:
            comment = "{0}/STH he/WDT her/STH you/NN ./STH".format(i)
            actual = extract1(comment)[0,feature-1]
            expected = 1
            self.assertEqual(actual, expected)
            
        comment = "hi/VBD he/WP her/STH you/NN ./STH"
        actual = extract1(comment)[0,feature-1]
        expected = 0
        self.assertEqual(actual, expected)
    
    def test_avg_sent(self):
        feature = 15

        comment = preproc1("This is sentence. Hi.")
        actual = extract1(comment)[0,feature-1]
        expected = 3
        self.assertEqual(actual, expected)
            
        comment = preproc1("This is sentence.")
        actual = extract1(comment)[0,feature-1]
        expected = 4
        self.assertEqual(actual, expected)
        
        comment = preproc1("Hi. Hi.")
        actual = extract1(comment)[0,feature-1]
        expected = 2
        self.assertEqual(actual, expected)
        
    def test_avg_tokens(self):
        feature = 16

        comment = "hi/VBD he/WP her/STH you/NN ./."
        actual = extract1(comment)[0,feature-1]
        expected = 2.5
        self.assertEqual(actual, expected)
            
        comment = "hi/VBD he/WP her/STH"
        actual = extract1(comment)[0,feature-1]
        expected = 7/3.0
        self.assertEqual(actual, expected) 
        
    def test_num_sent(self):
        feature = 17

        comment = preproc1("this is sent1. this is sent2. this is sent3.")
        actual = extract1(comment)[0,feature-1]
        expected = 3
        self.assertEqual(actual, expected)
            
        comment = preproc1("hi.")
        actual = extract1(comment)[0,feature-1]
        expected = 1
        self.assertEqual(actual, expected)  
        
        #question about number of sentences on piazza
        comment = preproc1("hi")
        actual = extract1(comment)[0,feature-1]
        expected = 1
        self.assertEqual(actual, expected)  
        
    
    def test_preprocess_sample(self):
        input_file = '/h/u5/c0/01/jafaris9/Desktop/A1/sample_outputs/sample_in.json' 
        inputs = json.load(open(input_file))
        inputs = [json.loads(i) for i in inputs]

        output_file = '/h/u5/c0/01/jafaris9/Desktop/A1/sample_outputs/sample_out.json'
        outputs = json.load(open(output_file))
        #outputs = [json.loads(i) for i in outputs]
	
        results = [{'id':i['id'], 'body': preproc1(i['body'])} for i in inputs]
	
        for i in results:
                comment_id = i['id']
                actual_body = i['body']
                expected_body = [i['body'] for i in outputs if i['id'] == comment_id][0]
                self.assertEqual(actual_body, expected_body) 


    def test_feature_sample(self):
	#assuming the preprocessing was fine
        input_file = '/h/u5/c0/01/jafaris9/Desktop/A1/sample_outputs/sample_out.json'        
        data = json.load(open(input_file))
        feats = np.zeros((len(data), 173+1))
        classes = {'Alt': 3, 'Center': 1, 'Right': 2, 'Left': 0}

        expected_path = '/h/u5/c0/01/jafaris9/Desktop/A1/sample_outputs/sample.npz'
        with np.load(expected_path) as sample_data:
                keys = [i for i in sample_data.iterkeys()]
                features = sample_data[keys[0]]
        expected = features

        for index,post in enumerate(data):
                e1_feats = extract1(post['body'])
                feats[index,0:-1] = e1_feats
                feats[index,:-1] = extract2(feats[index,:-1], post['cat'], post['id'])
                feats[index,173] = classes[post['cat']]
        for i in range(0, feats.shape[0]):
                print('comparing comment {0}'.format(i))
                for j in range(0, 30):
                        actual = feats[i,j]
                        expected_value = expected[i,j]
                        print('feature: {2}   actual: {0}   expected: {1}'.format(actual, expected_value, j)) 
        np.testing.assert_array_almost_equal(feats, expected)

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main( exit = False)
    print('done')
