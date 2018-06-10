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
from sklearn.decomposition import PCA

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

def euclidian_classification(x, Y, display=False):
    d = []  # Stores the computed distances
    # For each point in the training data compute the distance for the provided point
    for idx in range( 0, len( Y ) ):
        # Compute and append the current distance
        if display:
            print( x )
            print( Y[ idx, : ].A1 )
        if len( x ) < len(  Y[ idx, : ].A1 ):
            d.append( numpy.sqrt( numpy.sum( numpy.square( x - Y[idx, : -1].A1 ) ) ) )
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
            classification = euclidian_classification( x.A1, U )   # Find the classification of the point
            C[ classification ].append( x.A1 ) # Append the point to the cluster

        for idx in range( 0, k ):
            U[ idx ] = numpy.mean( C[ idx ], axis=0 )  # Get the mean of each feature in the cluster
            cSSE += compute_sse( C[ idx ], U[ idx ] )   # Add the cluster SSE to the total SSE

        if store:
            SSEs.append( cSSE ) # Store the current SSE for plotting
            runs.append( iterations )   # Store the current iteration for plotting

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
                if cluster.A1[ len( cluster.A1 ) - 1 ] > 0:
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


def write_results(filename, predictions):
    file = open( filename, "a" )
    for pred in predictions:
        file.write( str( pred[ 0 ] ) + ", " + str( pred[ 1 ] ) + "\n" )
    file.close

####################################################################
 # Function: main
####################################################################

def main(argv):

    # TODO: Test OneClassSVM by removing all instances of attacks then predicting whether the attack instances are outliers
    # TODO: If OneClassSVM Adds information combine it with Random Forest to get class probabilities and class
    # TODO: Save results in specified manner and run testing script

    # trainingSets = [ "./data/subject-1.csv",
    #                  "./data/subject-2.csv",
    #                  "./data/subject-3.csv",
    #                  "./data/subject-4.csv" ]
    # testingSets = [ "./data/general-instances.csv" ]
    # resultsFile = "./results/general-pred-1.csv"

    # trainingSets = [ "./data/individual-1.csv" ]
    # testingSets = [ "./data/individual-1-instances.csv" ]
    # resultsFile = "./results/individual-1-pred-1.csv"

    trainingSets = [ "./data/individual-2.csv" ]
    testingSets = [ "./data/individual-2-instances.csv" ]
    resultsFile = "./results/individual-2-pred-1.csv"

    k = parse_args( )
    numpy.set_printoptions( suppress=True )

    train = numpy.empty( ( 0, 9 ) )
    for set in trainingSets:
        data = init_matrices( set )
        data = condense_blocks( data )
        train = numpy.vstack( ( train, data ) )

    tT = numpy.matrix( train )
    tY = numpy.matrix( tT.T[ tT.shape[1] - 1 ] ).T   # Create the Y matrix by pulling the last column from T
    tY = numpy.ravel( tY )
    tT = numpy.delete( tT, tT.shape[1] - 1, 1 )    # Delete the last column in T
    tX = numpy.matrix( tT, float )  # Apply the normilization to the features to get the vector for analysis

    pca = PCA( n_components=4 )
    fitted = pca.fit_transform( tX )

    clusters, clusterModel, statuses = KM( numpy.hstack( ( fitted, numpy.matrix( tY ).T ) ), k, store=False )

    forests = []

    for idx, cluster in enumerate( clusters ):
        forests.append( RandomForestClassifier(
            bootstrap=False,
            class_weight=None,
            criterion='gini',
            max_depth=7,
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

        forests[ idx ].fit( X, y=Y )
        # print( forests[ idx ].feature_importances_ )


    predictions = []
    for idx, sample in enumerate( train ):
        testSample = pca.transform( numpy.matrix( sample[ : -1 ] ) )
        designation = euclidian_classification( testSample[ 0 ], clusterModel, False )   # Find the classification of the test point
        if statuses[ designation ] > 0:
            prob = forests[ designation ].predict_proba( testSample )[ 0 ]
            if prob[ 1 ] > 0.15:
                predictions.append( 1.0 )
            else:
                predictions.append( 0.0 )
        else:
            predictions.append( 0.0 )

    print( "Training Model Accuracy: {}%".format( model_error( train, predictions ) ) )

    print( " " )

    for idx, set in enumerate( testingSets ):
        hits = 0
        predictions = []
        test = numpy.loadtxt( set, float, delimiter="," )
        # test = numpy.loadtxt( set, float, delimiter=",", usecols=range( 1, 9 ) )

        for jdx, sample in enumerate( test ):
            sample = sample.reshape( ( 9, 7 ) )[ 1: ].T
            sample = numpy.matrix( sample ).mean( 0 ).A1
            testSample = pca.transform( numpy.matrix( sample ) )
            designation = euclidian_classification( testSample[ 0 ], clusterModel )   # Find the classification of the test point
            if statuses[ designation ] > 0:
                prob = forests[ designation ].predict_proba( testSample )[ 0 ]
                if prob[ 1 ] > 0.15:
                    hits += 1
                    predictions.append( [ ( prob[ 1 ] * 100.0 ), 1.0 ] )
                else:
                    predictions.append( [ ( prob[ 1 ] * 100.0 ), 0.0 ] )
            else:
                predictions.append( [ 0.0 , 0.0 ] )
        print( "Testing Model {} Attacks Predicted.".format( hits ) )
        print( "Testing Model Attack Ratio: {}%.".format( ( decimal.Decimal( hits ) / decimal.Decimal( len( test ) ) ) * decimal.Decimal( 100 ) ) )

        write_results( resultsFile, predictions )


    print( " " )

if __name__ == "__main__":
    main( sys.argv )
