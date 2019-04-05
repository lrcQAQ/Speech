# reference: https://progur.com/2018/02/how-to-use-mozilla-deepspeech-tutorial.html



from a3_levenshtein import *
from deepspeech import Model
import scipy.io.wavfile as wav
import os, fnmatch


dataDir = './data'

def generate_ds():
    model = Model('./models/output_graph.pb', 26, 9, './models/alphabet.txt', 500)
    
    # transcript for all speakers
    res = {}
    
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
            res[speaker] = res_speaker
    return res
            
if __name__ == "__main__":

    google_wer = []
    kaldi_wer = []
    ds_wer = []

    ds_transcript = generate_ds()
    
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            # read in transcript files for such speaker
            trans_path = os.path.join(dataDir, speaker, 'transcripts.txt')
            google_path = os.path.join(dataDir, speaker, 'transcripts.Google.txt')
            kaldi_path = os.path.join(dataDir, speaker, 'transcripts.Kaldi.txt')
            trans = open(trans_path, 'r').readlines()
            google = open(google_path, 'r').readlines()
            kaldi = open(kaldi_path, 'r').readlines()
            ds = ds_transcript[speaker]

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
                        curr_ds = preprocess(ds[i])
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

