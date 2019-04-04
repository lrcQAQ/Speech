import os
import numpy as np
import string
import re

# dataDir = '/u/cs401/A3/data/'
dataDir = './subdata/'

def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """

def preprocess(sent, trans=False):
    puncs = list(string.punctuation)
    puncs.remove('[')
    puncs.remove(']')

    # lowercase and remove [i] [label]
    sent = sent.strip().lower().split()
    trans = sent[2:]
    trans = ' '.join(trans)

    # remove <> contents in transcripts
    if(trans):
        pattern = re.compile("<.*?>")
        trans = re.sub(pattern, '', trans)

    # remove punctuations
    for punc in puncs:
        trans = trans.replace(punc, '')
    
    return trans.split()

if __name__ == "__main__":
    google_wer = []
    kaldi_wer = []

    # discussion file
    with open("asrDiscussion.txt", "w+") as f:

        for subdir, dirs, files in os.walk(dataDir):
            for speaker in dirs:
                # f.write("Processing " + speaker)
                # read in transcript files for such speaker
                trans_path = os.path.join(dataDir, speaker, 'transcripts.txt')
                google_path = os.path.join(dataDir, speaker, 'transcripts.Google.txt')
                kaldi_path = os.path.join(dataDir, speaker, 'transcripts.Kaldi.txt')
                trans = open(trans_path, 'r').readlines()
                google = open(google_path, 'r').readlines()
                kaldi = open(kaldi_path, 'r').readlines()

                # for each line, we find its wer
                valid = len(trans) != 0 and (len(google) != 0 or len(kaldi) != 0)
                if(valid):
                    print("===========Processing " + speaker + "===========")
                    lines = min(len(trans), len(google), len(kaldi))
                    for i in range(lines):
                        curr_trans = preprocess(trans[i], True)
                        
                        # calculate result for google
                        if(len(google) != 0):
                            curr_google = preprocess(google[i])
                            # g_wer, g_sub, g_ins, g_del = Levenshtein(curr_trans, curr_google)
                            # google_wer.append(g_wer)
                            # g_res = "Google result: wer " + str(g_wer) + " sub " + str(g_sub) + " ins " + str(g_ins) + " del " + str(g_del)
                            # f.write(g_res)
                            # f.write('\n')
                            # print(g_res)

                        # # calculate result for kaldi
                        # if(len(kaldi) != 0):
                        #     curr_kaldi = preprocess(kaldi[i])
                        #     k_wer, k_sub, k_ins, k_del = Levenshtein(curr_trans, curr_kaldi)
                        #     google_wer.append(k_wer)
                        #     k_res = "Kaldi result: wer " + str(k_wer) + " sub " + str(k_sub) + " ins " + str(k_ins) + " del " + str(k_del)
                        #     f.write(k_res)
                        #     f.write('\n')
                        #     print(k_res)

                    f.write('\n')
                            
                else:
                    print("===========Empty transcript for " + speaker + "===========")

                f.write('\n')
        
        # # report summary of result
        # g_mean, g_std = np.mean(google_wer), np.std(google_wer)
        # k_mean, k_std = np.mean(kaldi_wer), np.std(kaldi_wer)

        # f.write("===========Summary===========")
        # g_sum = "Google: mean is " + str(g_mean) + ", std is " + str(g_std)
        # k_sum = "Kaldi: mean is " + str(k_mean) + ", std is " + str(k_std)
        # f.write(g_sum)
        # f.write(k_sum)

        f.close()

