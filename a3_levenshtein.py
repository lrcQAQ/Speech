import os
import numpy as np

dataDir = '/u/cs401/A3/data/'

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



if __name__ == "__main__":
    google_wer = []
    kaldi_wer = []

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print("Processing " + speaker)
            # read in transcript files for such speaker
            trans_path = os.path.join(dataDir, speaker, 'transcripts.txt')
            google_path = os.path.join(dataDir, speaker, 'transcripts.Google.txt')
            kaldi_path = os.path.join(dataDir, speaker, 'transcripts.Kaldi.txt')
            trans = open(trans_path, 'r').readlines()
            google = open(google_path, 'r').readlines()
            kaldi = open(kaldi_path, 'r').readlines()

            # for each line, we find its wer
            lines = min(len(trans), len(google), len(kaldi))
            for i in range(lines):
                curr_trans, curr_google, curr_kaldi = trans[i], google[i], kaldi[i]
                

