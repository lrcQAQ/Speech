'''
Run on pretrained model to process wav file is really time consuming, so I only tried on subdataset.

Selected result from S-11C:
i did okay and i thegh like a di o can all these bag scort me whetever i got good but im i'm not 
musical and new the lyrics andd was not very talented with the singing but um denighust kay um i took 
course in middle school and also played in the band anmiddle school im but for ma be ti yours total so 
thatwas about itum they enjoylissin to muic and they enjoy music but nobody's really musical i like stuff 
it's onheradio rock and rap r are inb um don't really appreciate instrumental or a that type of stuff ery 
much because i didn't play inswrurment that much so i don't ippreciate it gesis the truth so um no at 
church last weekend harder aka so ter boy

S-11C
S-11C Google 0 0.24598930481283424 S: 18 I: 2 D: 26
S-11C Kaldi 0 0.1497326203208556 S: 18 I: 6 D: 4
S-11C DeepSpeech 0 0.26737967914438504 S: 37 I: 2 D: 11
S-11C Google 1 0.36363636363636365 S: 19 I: 0 D: 33
S-11C Kaldi 1 0.21678321678321677 S: 13 I: 10 D: 8
S-11C DeepSpeech 1 0.38461538461538464 S: 33 I: 4 D: 18
S-11C Google 2 0.2430939226519337 S: 55 I: 0 D: 33
S-11C Kaldi 2 0.16574585635359115 S: 34 I: 7 D: 19
S-11C DeepSpeech 2 0.27071823204419887 S: 73 I: 3 D: 22
S-11C Google 3 0.2506527415143603 S: 37 I: 0 D: 59
S-11C Kaldi 3 0.18276762402088773 S: 35 I: 7 D: 28
S-11C DeepSpeech 3 0.2819843342036554 S: 69 I: 7 D: 32
S-11C Google 4 0.21708185053380782 S: 70 I: 3 D: 49
S-11C Kaldi 4 0.09074733096085409 S: 25 I: 3 D: 23
S-11C DeepSpeech 4 0.3309608540925267 S: 125 I: 12 D: 49
S-11C Google 5 0.21176470588235294 S: 55 I: 3 D: 50
S-11C Kaldi 5 0.09803921568627451 S: 22 I: 7 D: 21
S-11C DeepSpeech 5 0.2607843137254902 S: 95 I: 9 D: 29
S-11C Google 6 0.26570048309178745 S: 30 I: 1 D: 24
S-11C Kaldi 6 0.17391304347826086 S: 19 I: 5 D: 12
S-11C DeepSpeech 6 0.3671497584541063 S: 61 I: 1 D: 14
S-11C Google 7 0.24701195219123506 S: 36 I: 1 D: 25
S-11C Kaldi 7 0.11155378486055777 S: 15 I: 1 D: 12
S-11C DeepSpeech 7 0.3784860557768924 S: 71 I: 5 D: 19
S-11C Google 8 0.2549019607843137 S: 13 I: 1 D: 12
S-11C Kaldi 8 0.17647058823529413 S: 10 I: 5 D: 3
S-11C DeepSpeech 8 0.3333333333333333 S: 24 I: 2 D: 8

Google: mean is 0.25553703167766545, std is 0.04153513293733992
Kaldi: mean is 0.1517503645221992, std is 0.04043593529917004
DeepSpeech: mean is 0.3194902161544414, std is 0.04745023676606771

The std is small becasue I only use a 2 speakers as dataset, it took endless time to process wave file.

Discussion:
The result is not good, it contains lots of nonsense words, such as "di", "o", "enjoylissin", "itum".
And its mean of wer is larger than Google and Kaldi, which also indicates it performs worse.

To run the code, need "pip3 install deepspeech"
"wget https://github.com/mozilla/DeepSpeech/releases/download/v0.1.1/deepspeech-0.1.1-models.tar.gz"
"tar -xvzf deepspeech-0.1.1-models.tar.gz"

reference: https://progur.com/2018/02/how-to-use-mozilla-deepspeech-tutorial.html
reference: https://github.com/mozilla/DeepSpeech#using-the-python-package

'''

from a3_levenshtein import *
from deepspeech import Model
import scipy.io.wavfile as wav
import os, fnmatch


# dataDir = './data'
dataDir = '/u/cs401/A3/data/'

def generate_ds():
    '''
    A function that readin pretrained deepspeech model, and use such model to generate deepspeech
    transcript for each wave file, in ascending order (i.e. "0.wav", "1.wav" ... ). Result is wrote
    in a file in path "dataDir/speaker/transcript.ds.txt"
    '''
    model = Model('./models/output_graph.pb', 26, 9, './models/alphabet.txt', 500)
    
    # process for each speaker
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            
            # get all files of wav format in ascending order
            files = []
            i = 0
            path = os.path.join(dataDir, speaker, str(i)) + ".wav"
            while(os.path.isfile(path)):
                files.append(path)
                i += 1
                path = os.path.join(dataDir, speaker, str(i)) + ".wav"
                
            # process each wav file using model, to generate transcript
            res_speaker = []
            for file_path in files:
                fs, audio = wav.read(file_path)
                res_speaker.append(model.stt(audio, fs))
            
            # save such speaker's transcript
            with open(os.path.join(dataDir, speaker) + "transcript.ds.txt", 'w+') as f:
                f.write("\n".join(res_speaker))
                f.close()
    return
            
if __name__ == "__main__":

    google_wer = []
    kaldi_wer = []
    ds_wer = []

    # process all wave files
    generate_ds()
    
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            # read in transcript files for such speaker
            trans_path = os.path.join(dataDir, speaker, 'transcripts.txt')
            google_path = os.path.join(dataDir, speaker, 'transcripts.Google.txt')
            kaldi_path = os.path.join(dataDir, speaker, 'transcripts.Kaldi.txt')
            ds_path = os.path.join(dataDir, speaker, 'transcripts.ds.txt')
            trans = open(trans_path, 'r').readlines()
            google = open(google_path, 'r').readlines()
            kaldi = open(kaldi_path, 'r').readlines()
            ds = open(ds_path, 'r').readlines()

            # only process when transcript is nonempty and reference exist
            # and deepspeech transcript is nonempty
            valid = len(trans) != 0 and (len(google) != 0 or len(kaldi) != 0) and len(ds) != 0
            if(valid):

                lines = min(len(trans), len(google), len(kaldi), len(ds))
                # for each paired lines, we find its wer
                for i in range(lines):

                    curr_trans = preprocess(trans[i])

                    # calculate result for google
                    if(len(google) != 0):
                        curr_google = preprocess(google[i])
                        g_wer, g_sub, g_ins, g_del = Levenshtein(curr_trans, curr_google)
                        google_wer.append(g_wer)
                        g_res = speaker + " Google " + str(i) + " " + str(g_wer) + " S: " + str(g_sub) + " I: " + str(g_ins) + " D: " + str(g_del)
                        print(g_res)

                    # calculate result for kaldi
                    if(len(kaldi) != 0):
                        curr_kaldi = preprocess(kaldi[i])
                        k_wer, k_sub, k_ins, k_del = Levenshtein(curr_trans, curr_kaldi)
                        kaldi_wer.append(k_wer)
                        k_res = speaker + " Kaldi " + str(i) + " " + str(k_wer) + " S: " + str(k_sub) + " I: " + str(k_ins) + " D: " + str(k_del)
                        print(k_res)
                        
                    # calculate result for ds
                    if(len(ds) != 0):
                        # basic preprocess for ds
                        curr_ds = ds[i].strip().lower()
                        puncs = list(string.punctuation)
                        puncs.remove('[')
                        puncs.remove(']')
                        for punc in puncs:
                            curr_ds = curr_ds.replace(punc, '')
                        curr_ds = curr_ds.split()

                        d_wer, d_sub, d_ins, d_del = Levenshtein(curr_trans, curr_ds)
                        ds_wer.append(d_wer)
                        d_res = speaker + " DeepSpeech " + str(i) + " " + str(d_wer) + " S: " + str(d_sub) + " I: " + str(d_ins) + " D: " + str(d_del)
                        print(d_res)
        
        
        # report summary of result
        g_mean, g_std = np.mean(google_wer), np.std(google_wer)
        k_mean, k_std = np.mean(kaldi_wer), np.std(kaldi_wer)
        d_mean, d_std = np.mean(ds_wer), np.std(ds_wer)

        g_sum = "Google: mean is " + str(g_mean) + ", std is " + str(g_std)
        k_sum = "Kaldi: mean is " + str(k_mean) + ", std is " + str(k_std)
        d_sum = "DeepSpeech: mean is " + str(d_mean) + ", std is " + str(d_std)
        print(g_sum)
        print(k_sum)
        print(d_sum)

