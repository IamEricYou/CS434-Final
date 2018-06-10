import sys
import time
from datetime import datetime, date, time, timedelta
import math
import decimal
import numpy
# import matplotlib.pyplot as plot
from matplotlib.dates import strpdate2num
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from keras.models import Sequential
from keras.layers import Dense, Dropout

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #For AVX2 FMA Error, which is causing from TensorFlow
numpy.random.seed(777)

def parse_args():
    if len( sys.argv ) < 2:
        return 0
    else:
        return int( sys.argv[1] )


####################################################################
 # Function: init_matrices
 # Description: Initialize the matricies
####################################################################

def init_matrices(filename):
    # convert_routine = { 0:strpdate2num(  ), 1:... }
    X = numpy.loadtxt( filename, numpy.datetime64, delimiter=",", usecols=( 0, ) )    # Load the timestamp data
    Y = numpy.loadtxt( filename, float, delimiter=",", usecols=range( 1, 10 ) )    # Load the data into a matrix using a integer datatype each item is sepearated by commas
    data = OrderedDict( zip( X, Y ) )
    return( data )


####################################################################
 # Function: condense_blocks
 # Description: Condenses the data into the means
####################################################################

def condense_blocks(X):

    data = []
    block = OrderedDict( )

    blockHasAttack = False
    expectedPeriod = numpy.timedelta64( 6, 'm' )

    for idx, ( key, value ) in enumerate( X.iteritems() ):
        foundGap = False
        if len( block ) > 0:
            period = key - block.keys()[ -1 ]
            if period > expectedPeriod:
                foundGap = True

            if idx % 6 == 0 or foundGap:
                means = numpy.matrix( block.values( ) ).mean( 0 ).A1
                if blockHasAttack:
                    means[ len( means ) - 1 ] = 1
                else:
                    means[ len( means ) - 1 ] = 0

                # print( means )
                data.append( means )
                block = OrderedDict( )
                blockHasAttack = False

        if value[ len( value ) - 1 ] == 1:
            blockHasAttack = True

        block.update( { key: value } )

    return( numpy.array( data ) )


####################################################################
 # Function: get_k_clusters
 # Description: Gets k random samples from a dataset
####################################################################

def get_k_clusters(X, k):
    # Return an matrix of k random samples from the dataset
    return X[ numpy.random.randint( X.shape[0], size=k ),: ]


####################################################################
 # Function: euclidian_classification
 # Description: Computes the euclidian diatances given x and Y
####################################################################

def euclidian_classification(x, Y):
    d = []  # Stores the computed distances
    # For each point in the training data compute the distance for the provided point
    for idx in range( 0, len( Y ) ):
        # Compute and append the current distance
        if len( x ) < len(  Y[idx, : ] ):
            d.append( numpy.sqrt( numpy.sum( numpy.square( x - Y[idx, : -1] ) ) ) )
        else:
            d.append( numpy.sqrt( numpy.sum( numpy.square( x[ : -1 ] - Y[idx, : -1] ) ) ) )

    return d.index( min( d ) )


####################################################################
 # Function: compute_sse
 # Description: Computes the SSE of a cluster
####################################################################

def compute_sse(X, y):
    # Compute the SSE of the cluster
    return numpy.sum( numpy.square( numpy.abs( numpy.subtract( X, y ) ) ) )


####################################################################
 # Function: KM
 # Description: Performs the K-Means algorithm
####################################################################

def KM(X, k, store=False):
    U = get_k_clusters( X, k )  # The base case of cluster seperations
    pSSE = 0    # Previous SSE
    cSSE = 0    # Current SSE
    SSEs = []   # Stored SSEs
    runs = []
    iterations = 0

    # Continue cluster analysis until convergence
    while( 1 ):
        pSSE = cSSE;    # Store the previous SSE
        cSSE = 0

        C = []  # Create a list to store the clusters
        for kdx in range( 0, k ):
            C.append( [] )  # Initialize a cluster

        # For each sample in the dataset split based on the cluster
        for x in X:
            classification = euclidian_classification( x, U )   # Find the classification of the point
            C[ classification ].append( x ) # Append the point to the cluster

        for idx in range( 0, k ):
            U[ idx ] = numpy.mean( C[ idx ], axis=0 )  # Get the mean of each feature in the cluster
            cSSE += compute_sse( C[ idx ], U[ idx ] )   # Add the cluster SSE to the total SSE

        if store:
            SSEs.append( cSSE ) # Store the current SSE for plotting
            runs.append( iterations )   # Store the current iteration for plotting

        # print ( "{}: Previous: {} - Current: {}".format( iterations, pSSE, cSSE ) )
        iterations += 1    # Increment the number of iterations

        if pSSE == cSSE:

            cs = 0
            count = 0
            for cluster in C:
                cs += 1
                subcount = 0
                for point in cluster:
                    if point[ len( point ) - 1 ] == 1:
                        count += 1
                        subcount += 1
                print( "[ CLUSTER {} ] Points: {}  -  Attacks: {}  -  Ratio: {}".format( cs, len( cluster ), subcount, ( ( subcount / float( len( cluster ) ) ) * 100 ) ) )
                means = numpy.matrix( cluster ).mean( 0 )
                for mean in means.tolist()[ 0 ]:
                    sys.stdout.write( str( mean ) )
                    sys.stdout.write(', ')
                    sys.stdout.flush()
                print( "\n" )

            print( "[ TOTAL ] Points: {}  -  Attacks: {}  -  Ratio: {}".format( len( X ), count, ( ( count / float( len( X ) ) ) * 100 ) ) )

            statuses = numpy.zeros( ( k, 1 ) )
            for idx, cluster in enumerate( U ):
                if cluster[ len( cluster ) - 1 ] > 0:
                    statuses[ idx ] = 1;

            return ( C, U, statuses )


####################################################################
 # Function: model_error
 # Description: Computes the total error given a model and it's
 # actual results
####################################################################

def model_error( M, Ye ):
    decimal.getcontext().prec = 4   # Set the decimal precision
    misses = 0
    missedNonAttacks = 0
    missedAttacks = 0
    # Check if the predicited result meets the actual for each point
    for idx in range( 0, M.shape[0] ):
        if M[ idx ][ -1 ] != Ye[ idx ]:
            if M[ idx ][ -1 ] == 0:
                missedNonAttacks += 1
            else:
                missedAttacks += 1
            misses += 1
    print( "\nFalse Positives: {}".format( missedNonAttacks ) )
    print( "False Negatives: {}".format( missedAttacks ) )
    error = ( ( decimal.Decimal( M.shape[0] ) - decimal.Decimal( misses ) ) / decimal.Decimal( M.shape[0] ) ) * decimal.Decimal( 100 )
    return( error )

def NN(training,testing):

    X = training[:,0:8]
    Y = training[:,8]
    real_testing = [ "./data/general_test_instances.csv" ]
    subj_testing1 = [ "./data/subject2_instances.csv" ]
    subj_testing2 = [ "./data/subject7_instances.csv" ]

    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu')) # input layer requires input_dim param
    model.add(Dense(15, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid')) # sigmoid instead of relu for final probability between 0 and 1
    
    model.compile(loss="binary_crossentropy", optimizer="SGD", metrics=['accuracy'])
    history = model.fit(X, Y, epochs = 300, batch_size=10, verbose=1)
    
    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    count = 0
    for idx, set in enumerate( testing ):
        test = numpy.loadtxt( set, float, delimiter=",", usecols=range( 1, 9 ) )
        test = numpy.matrix( test ).mean( 0 ).A1
        test = numpy.array([test])
        prediction = model.predict(test)
        print prediction[0]*100
        if prediction[0]*100 < 4.0:
            print(str(count) + " testset result: " + str(0))
            count = count + 1
        else:
            print(str(count) + " testset result: " + str(1))
            count = count + 1
            
    count = 0
    """
    for idx, set in enumerate( real_testing ):
        test = numpy.loadtxt( set, float, delimiter="," )
        for jdx, sample in enumerate( test ):
            sample = sample.reshape( ( 9, 7 ) )[ 1: ].T
            sample = numpy.matrix( sample ).mean( 0 ).A1
            sample = numpy.array([sample])
            prediction = model.predict(sample)
            print prediction[0]*100
    """


####################################################################
 # Function: main
####################################################################

def main(argv):

    # TODO: Test OneClassSVM by removing all instances of attacks then predicting whether the attack instances are outliers
    # TODO: If OneClassSVM Adds information combine it with Random Forest to get class probabilities and class
    # TODO: Save results in specified manner and run testing script

    trainingSets = [ "./data/subject-1.csv",
                     "./data/subject-2.csv",
                     "./data/subject-3.csv",
                     "./data/subject-4.csv" ]
    # testingSets = [ "./data/general-instances.csv" ]
    testingSets = [ "./sampleinstances/sampleinstance_1.csv",
                    "./sampleinstances/sampleinstance_2.csv",
                    "./sampleinstances/sampleinstance_3.csv",
                    "./sampleinstances/sampleinstance_4.csv",
                    "./sampleinstances/sampleinstance_5.csv" ]

    # trainingSets = [ "./data/individual-1.csv" ]
    # testingSets = [ "./data/individual-1-instances.csv" ]
    #
    # trainingSets = [ "./data/individual-2.csv" ]
    # testingSets = [ "./data/individual-2-instances.csv" ]

    k = parse_args( )
    numpy.set_printoptions( suppress=True )

    train = numpy.empty( ( 0, 9 ) )
    for set in trainingSets:
        data = init_matrices( set )
        data = condense_blocks( data )
        train = numpy.vstack( ( train, data ) )

    clusters, clusterModel, statuses = KM( train, k, store=False )
    forests = []
    SVMs = []
    for idx, cluster in enumerate( clusters ):
        SVMs.append(
            svm.OneClassSVM( nu=0.1, kernel="rbf", gamma=0.1 )
        )

        forests.append( RandomForestClassifier(
            bootstrap=False,
            class_weight=None,
            criterion='entropy',
            max_depth=None,
            max_features=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_impurity_split=None,
            min_samples_leaf=1,
            min_samples_split=2,
            min_weight_fraction_leaf=0.0,
            n_estimators=20,
            n_jobs=1,
            oob_score=False,
            random_state=0,
            verbose=0,
            warm_start=False )
        )

        T = numpy.matrix( cluster )
        Y = numpy.matrix( T.T[ T.shape[1] - 1 ] ).T   # Create the Y matrix by pulling the last column from T
        Y = numpy.ravel( Y )
        T = numpy.delete( T, T.shape[1] - 1, 1 )    # Delete the last column in T
        X = numpy.matrix( T, float )  # Apply the normilization to the features to get the vector for analysis

        forests[ idx ].fit( X, Y )
        # print( forests[ idx ].feature_importances_ )

        # removed = 0
        # svmX = numpy.copy( X )
        # for jdx, y in enumerate( Y ):
        #     if y == 1:
        #         svmX = numpy.delete( svmX, ( jdx - removed ), axis=0 )
        #         removed += 1
        svmY = numpy.copy( Y )
        svmY[ svmY == 1 ] = -1
        svmY[ svmY == 0 ] = 1
        SVMs[ idx ].fit( X, y=svmY )


    predictions = []
    for idx, sample in enumerate( train ):
        designation = euclidian_classification( sample, clusterModel )   # Find the classification of the test point
        if statuses[ designation ] > 0:
            prediction = forests[ designation ].predict( numpy.matrix( sample[ : -1 ] ) )[ 0 ]
            # prediction = SVMs[ designation ].predict( numpy.matrix( sample[ : -1 ] ) )[ 0 ]
            # if prediction == 1:
            #     prediction = 0
            # else:
            #     prediction = 1
            predictions.append( prediction )
        else:
            predictions.append( 0.0 )

    print( "Training Model Accuracy: {}%".format( model_error( train, predictions ) ) )

    print( " " )

    for idx, set in enumerate( testingSets ):
        hits = 0
        predictions = []
        test = numpy.loadtxt( set, float, delimiter="," )
        for jdx, sample in enumerate( test ):
            sample = sample.reshape( ( 9, 7 ) )[ 1: ].T
            sample = numpy.matrix( sample ).mean( 0 ).A1
            designation = euclidian_classification( sample, clusterModel )   # Find the classification of the test point
            if statuses[ designation ] > 0:
                prediction = forests[ designation ].predict( numpy.matrix( sample ) )[ 0 ]
                # prediction = SVMs[ designation ].predict( numpy.matrix( sample ) )[ 0 ]
                if prediction == 1.0:
                    hits += 1
                predictions.append( prediction )
            else:
                predictions.append( 0.0 )
        print( "Testing Model {} Attacks Predicted.".format( hits ) )
        print( "Testing Model Attack Ratio: {}%.".format( ( decimal.Decimal( hits ) / decimal.Decimal( len( test ) ) ) * decimal.Decimal( 100 ) ) )

        # test = numpy.matrix( test ).mean( 0 ).A1
        #
        # designation = euclidian_classification( test, clusterModel )   # Find the classification of the test point
        # print( "[ TEST {} ] Cluster Designation: {}.".format( idx, ( designation + 1 ) ) )
        # # print( test )
        #
        # if statuses[ designation ] > 0:
        #     prob =  forests[ designation ].predict_proba( numpy.matrix( test ) )
        #     clas = forests[ designation ].predict( numpy.matrix( test ) )
        #     print( "[ TEST {} ] Probability: {} Prediction: {}.\n".format( idx, prob, clas ) )
        #     print( numpy.matrix( test ) )
        # else:
        #     print( " " )

    #NN(train, testingSets) for NN
    print( " " )

if __name__ == "__main__":
    main( sys.argv )
