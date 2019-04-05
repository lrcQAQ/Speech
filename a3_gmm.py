from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random

# packages
from scipy.special import logsumexp

dataDir = '/u/cs401/A3/data/'
# dataDir = './subdata/'


class theta:
    def __init__(self, name, M=8,d=13):
        self.name = name
        self.omega = np.zeros((M,1))
        self.mu = np.zeros((M,d))
        self.Sigma = np.zeros((M,d))


def log_b_m_x( m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM
    '''
    # extract shape and parameters
    D = myTheta.mu.shape[1]
    sigma = myTheta.Sigma[m]
    mu = myTheta.mu[m]

    # apply formula
    term1 = np.sum(np.divide(np.square(x - mu), sigma), axis=1)
    term2 = 0.5 * D * np.log(2 * np.pi)
    term3 = 0.5 * np.sum(np.log(np.square(sigma)))
    res = - term1 - term2 - term3

    return res

def log_b_m_X(m, X, myTheta):
    ''' Vectorized version of log_b_m_x.
    '''
    # extract shape and parameters
    D = X.shape[1]

    mu = myTheta.mu[m]
    sigma = myTheta.Sigma[m]

    # apply formula
    term1 = np.sum(np.divide(np.square(X - mu), sigma), axis=1)
    term2 = 0.5 * D * np.log(2 * np.pi)
    term3 = 0.5 * np.sum(np.log(np.square(np.prod(sigma))))
    res = - term1 - term2 - term3

    return res

def log_p_m_x( m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    # extract shape and parameters
    omega = myTheta.omega
    M = omega.shape[0]
    log_Bs = np.array([log_b_m_x(i, x, myTheta) for i in range(M)])

    # apply formula
    nume = np.log(omega[m, 0]) + log_Bs[m]
    deno = logsumexp(np.log(omega[:, 0]) + log_Bs)
    res = nume - deno
    return res

def log_ps(log_Bs, myTheta):
    # extract shape and parameters
    omega = myTheta.omega

    # apply formula
    nume = log_Bs + np.log(omega)
    deno = logsumexp(nume, axis=0)
    log_Ps = nume - deno
    return log_Ps

def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 

        See equation 3 of the handout
    '''
    # extract shape and parameters
    omega = myTheta.omega

    # apply formula
    res = np.sum(logsumexp(log_Bs + np.log(omega), axis=0))
    return res
  
def train( speaker, X, M=8, epsilon=0.0, maxIter=20 ):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    myTheta = theta( speaker, M, X.shape[1] )
    
    # initialize paramters
    T = X.shape[0]
    randX = np.random.choice(T, M, replace=False)
    myTheta.Sigma.fill(1)
    myTheta.omega[:, 0] = 1.0 / M
    for i in range(len(randX)):
        myTheta.mu[i] = X[randX[i]]
    
    # initialize loop
    i = 0
    prev_L = float('-inf')
    improvement = float('inf')
    # log_Bs = np.zeros((M, T))

    # start loop
    while i <= maxIter and improvement >= epsilon:
        # compute intermediate results
        log_Bs = np.zeros((M, T))
        for m in range(M):
            log_Bs[m] = log_b_m_X(m, X, myTheta)
        log_Ps = log_ps(log_Bs, myTheta)
        
        # compute likelihood
        L = logLik(log_Bs, myTheta)

        # update parameters
        for m in range(M):
            sum_Ps = np.sum(np.exp(log_Ps[m]))

            # omega
            myTheta.omega[m] = np.divide(sum_Ps, T)
            # mu
            myTheta.mu[m] = np.divide(np.dot(np.exp(log_Ps[m]), X), sum_Ps)
            # sigma
            sigma_term1_nume = np.dot(np.exp(log_Ps[m]), np.square(X))
            sigma_term1 = np.divide(sigma_term1_nume, sum_Ps)
            sigma_term2 = np.square(myTheta.mu[m])
            myTheta.Sigma[m] = np.subtract(sigma_term1, sigma_term2)

        # update loop
        improvement = L - prev_L
        prev_L = L
        i += 1

    return myTheta

def test( mfcc, correctID, models, k=5 ):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK] 

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    bestModel = -1
    all_likelihood = np.zeros((len(models)))
    best = float('-inf')

    # calculate all likelihood and save the best
    M, T = models[0].omega.shape[0], mfcc.shape[0]
    for i in range(len(models)):
        model = models[i]
        log_Bs = np.zeros((M, T))
        for m in range(M):
            log_Bs[m, :] = log_b_m_X(m, mfcc, model)
        likelihood = logLik(log_Bs, model)
        all_likelihood[i] = likelihood

        if(likelihood > best):
            bestModel = i
            best = likelihood
    
    # k best models
    top = np.argsort(-all_likelihood)

    # print to output, write to file
    with open('gmmLiks.txt', 'a') as f:
        title = models[correctID].name
        print(title)
        f.write(title)
        f.write('\n')
        for kk in range(k):
            idx = top[kk]
            output = str(models[idx].name) + ' ' + str(all_likelihood[idx])
            print(output)
            f.write(output)
            f.write('\n')
        f.write('\n')
        f.close()

    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
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

            trainThetas.append( train(speaker, X, M, epsilon, maxIter) )

    # evaluate 
    numCorrect = 0
    for i in range(0,len(testMFCCs)):
        numCorrect += test( testMFCCs[i], i, trainThetas, k ) 
    accuracy = 1.0*numCorrect/len(testMFCCs)
    print("Accuracy: ", accuracy)

