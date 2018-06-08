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
 # Function: main
####################################################################

def main(argv):

    trainingSets = [ "./data/subject-1.csv",
                     "./data/subject-2.csv",
                     "./data/subject-3.csv",
                     "./data/subject-4.csv" ]

    testingSets = [ "./sampleinstances/sampleinstance_1.csv",
                    "./sampleinstances/sampleinstance_2.csv",
                    "./sampleinstances/sampleinstance_3.csv",
                    "./sampleinstances/sampleinstance_4.csv",
                    "./sampleinstances/sampleinstance_5.csv" ]

    k = parse_args( )
    numpy.set_printoptions( suppress=True )

    train = numpy.empty( ( 0, 9 ) )
    for set in trainingSets:
        data = init_matrices( set )
        data = condense_blocks( data )
        train = numpy.vstack( ( train, data ) )

    clusters, clusterModel, statuses = KM( train, k, store=False )
    # print( "\nK Means SSE: {}\n".format( sse ) )

    print( " " )
    for idx, set in enumerate( testingSets ):
        test = numpy.loadtxt( set, float, delimiter=",", usecols=range( 1, 9 ) )
        test = numpy.matrix( test ).mean( 0 ).A1

        designation = euclidian_classification( test, clusterModel )   # Find the classification of the test point
        print( "[ TEST {} ] Cluster Designation: {}. ".format( idx, ( designation + 1 ) ) )

        T = numpy.matrix( clusters[ designation ] )
        Y = numpy.matrix( T.T[ T.shape[1] - 1 ] ).T   # Create the Y matrix by pulling the last column from T
        Y = numpy.ravel( Y )
        T = numpy.delete( T, T.shape[1] - 1, 1 )    # Delete the last column in T
        X = numpy.matrix( T, float )  # Apply the normilization to the features to get the vector for analysis

        if statuses[ designation ] > 0:
            forest = RandomForestClassifier( )
            forest.fit( X, Y )
            print( "[ TEST {} ] Prediction: {}.\n".format( idx, forest.predict( numpy.matrix( test ) )[ 0 ] ) )
        else:
            print( " " )

    print( " " )

if __name__ == "__main__":
    main( sys.argv )
