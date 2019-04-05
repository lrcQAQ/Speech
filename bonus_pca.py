'''
result:
target dimension    Accuracy
    10              1.0
    8               1.0
    6               0.96875
    4               1.0
    2               0.84375
'''

from a3_gmm import *
from sklearn.decomposition import PCA

if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20

    # pca parameters [modify this parameter for different target dimension]
    dim = 2

    # gather all data for build the pca model
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:

            files = fnmatch.filter(os.listdir( os.path.join( dataDir, speaker ) ), '*npy')
            random.shuffle( files )
            
            testMFCC = np.load( os.path.join( dataDir, speaker, files.pop() ) )
            testMFCCs.append( testMFCC )

            X = np.empty((0,d))
            for file in files:
                myMFCC = np.load( os.path.join( dataDir, speaker, file ) )
                X = np.append( X, myMFCC, axis=0)

    # fit pca model
    pca = PCA(n_components=dim)
    pca.fit(X)

    testMFCCs = []

    # training using data transformed by pca model
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print( speaker )

            files = fnmatch.filter(os.listdir( os.path.join( dataDir, speaker ) ), '*npy')
            random.shuffle( files )
            
            testMFCC = np.load( os.path.join( dataDir, speaker, files.pop() ) )
            testMFCCs.append( testMFCC )

            X = np.empty((0,d))
            for file in files:
                myMFCC = np.load( os.path.join( dataDir, speaker, file ) )
                X = np.append( X, myMFCC, axis=0)

            # transform data using pca model
            X_trans = pca.transform(X)
            trainThetas.append( train(speaker, X_trans, M, epsilon, maxIter) )

    # evaluate 
    numCorrect = 0
    for i in range(0,len(testMFCCs)):
        # pca transform test data as well
        trans_x = pca.transform(testMFCCs[i])
        numCorrect += test( trans_x, i, trainThetas, k ) 
    accuracy = 1.0*numCorrect/len(testMFCCs)
    print("Accuracy: ", accuracy)
