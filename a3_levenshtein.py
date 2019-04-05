import os
import numpy as np
import string
import re

dataDir = '/u/cs401/A3/data/'
# dataDir = './subdata/'

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
    n = len(r)
    m = len(h)
    R = np.zeros((n + 1, m + 1)) # matrix of distances
    B = np.zeros((n + 1, m + 1)) # backtracing matrix

    # initialize R
    R[:, 0] = np.arange(n + 1)
    R[0, :] = np.arange(m + 1)

    # initialize backtrace, first row can only go left, first column can only go up
    B[1:, 0] = 1
    B[0, 1:] = 2

    # statr loop
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dele = R[i - 1, j] + 1
            sub = R[i - 1, j - 1] if r[i - 1] == h[j - 1] else R[i - 1, j - 1] + 1
            ins = R[i, j - 1] + 1
            R[i, j] = min(dele, sub, ins)

            if(R[i, j] == dele):
                B[i, j] = 1 # up
            elif(R[i, j] == ins):
                B[i, j] = 2 # left
            else:
                B[i, j] = 3 # up-left
            
    # get wer
    wer = R[n, m] / n

    # backtrace to get nS, nI, nD
    nS, nI, nD = 0, 0, 0
    i, j = n, m
    while i != 0 or j != 0:
        if(B[i, j] == 1): # up, delete
            nD += 1
            i -= 1
        elif(B[i, j] == 2): # left, insert
            nI += 1
            j -= 1
        else:
            # up-left substitute
            if(R[i, j] == R[i - 1, j - 1] + 1):
                nS += 1
            i -= 1
            j -= 1
    
    return wer, nS, nI, nD


def preprocess(sent):
    puncs = list(string.punctuation)
    puncs.remove('[')
    puncs.remove(']')

    # lowercase and ignore [i] [label]
    sent = sent.strip().lower().split()
    trans = sent[2:]
    trans = ' '.join(trans)

    # remove <> and [] contents in transcripts
    pattern = re.compile(r"<\w+>")
    trans = re.sub(pattern, '', trans)
    pattern = re.compile(r"\[\w+\]")
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
                # read in transcript files for such speaker
                trans_path = os.path.join(dataDir, speaker, 'transcripts.txt')
                google_path = os.path.join(dataDir, speaker, 'transcripts.Google.txt')
                kaldi_path = os.path.join(dataDir, speaker, 'transcripts.Kaldi.txt')
                trans = open(trans_path, 'r').readlines()
                google = open(google_path, 'r').readlines()
                kaldi = open(kaldi_path, 'r').readlines()

                # only process when transcript is nonempty and reference exist
                valid = len(trans) != 0 and (len(google) != 0 or len(kaldi) != 0)
                if(valid):

                    lines = min(len(trans), len(google), len(kaldi))
                    # for each paired lines, we find its wer
                    for i in range(lines):

                        curr_trans = preprocess(trans[i])

                        # calculate result for google
                        if(len(google) != 0):
                            curr_google = preprocess(google[i])
                            g_wer, g_sub, g_ins, g_del = Levenshtein(curr_trans, curr_google)
                            google_wer.append(g_wer)
                            g_res = speaker + " Google " + str(i) + " " + str(g_wer) + " S: " + str(g_sub) + " I: " + str(g_ins) + " D: " + str(g_del)
                            f.write(g_res)
                            f.write('\n')
                            print(g_res)

                        # calculate result for kaldi
                        if(len(kaldi) != 0):
                            curr_kaldi = preprocess(kaldi[i])
                            k_wer, k_sub, k_ins, k_del = Levenshtein(curr_trans, curr_kaldi)
                            kaldi_wer.append(k_wer)
                            k_res = speaker + " Kaldi " + str(i) + " " + str(k_wer) + " S: " + str(k_sub) + " I: " + str(k_ins) + " D: " + str(k_del)
                            f.write(k_res)
                            f.write('\n')
                            print(k_res)

                    f.write('\n')

                f.write('\n')
        
        # report summary of result
        g_mean, g_std = np.mean(google_wer), np.std(google_wer)
        k_mean, k_std = np.mean(kaldi_wer), np.std(kaldi_wer)

        g_sum = "Google: mean is " + str(g_mean) + ", std is " + str(g_std)
        k_sum = "Kaldi: mean is " + str(k_mean) + ", std is " + str(k_std)
        f.write(g_sum)
        f.write('\n')
        f.write(k_sum)
        print(g_sum)
        print(k_sum)

        f.close()

