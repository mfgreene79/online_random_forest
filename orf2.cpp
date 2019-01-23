// -*- C++ -*-
/*
 * orf.cpp - Online Random Forests linked with Rcpp to R 
 * 
 * 
 * 
*/

//include the necessary RcppEigen library
#include <RcppEigen.h>
//[[Rcpp::depends(RcppEigen)]]

//include all libraries from Saffaris OMCBoost.cpp file
#include <cstdlib>
#include <iostream>
#include <string>
#include <string.h>
#include <libconfig.h++>

#include <fstream>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>


#include "OMCLPBoost/data.h"
#include "OMCLPBoost/classifier.h"
#include "OMCLPBoost/utilities.h"
#include "OMCLPBoost/hyperparameters.h"
#include "OMCLPBoost/experimenter.h"
#include "OMCLPBoost/online_rf.h"

using namespace std;
using namespace libconfig;
using namespace Rcpp;
using namespace Eigen;

/***********************************************************
 * 
 * Code to access Saffaris online random forest components
 * 
 ***********************************************************/

// defining a method for calling an Online Random Forest - fit with data, return the Forest Object
// Forest Object defined as an arma::field<arma::mat> where each component Matrix object is a Tree.  
// Each row in the tree represents a node.
// Node fields: 


/////create a DataSet item from input matrix and y
DataSet make_trainData(MatrixXd x, VectorXd y) {
  //creates a DataSet class from matrices x and y
  DataSet ds;
  ds.m_numFeatures = x.cols();
  ds.m_numSamples = x.rows();

  set<int> labels;  
  for (int nSamp = 0; nSamp < x.rows(); nSamp++) {
    Sample sample;
    sample.x = VectorXd(ds.m_numFeatures);
    sample.id = nSamp;
    sample.w = 1.0;
    sample.y = y(nSamp);
    labels.insert(sample.y);
    for (int nFeat = 0; nFeat < ds.m_numFeatures; nFeat++) {
      sample.x(nFeat) = x(nSamp, nFeat);
    } //loop nFeat
    ds.m_samples.push_back(sample); // push sample into dataset
  } //loop nSamp
  ds.m_numClasses = labels.size();
  ds.findFeatRange();
  
  return(ds);
}

//  OnlineRF(const Hyperparameters& hp, const int& numClasses, const int& numFeatures, const VectorXd& minFeatRange, const VectorXd& maxFeatRange);
// [[Rcpp::export]]
vector<MatrixXd> online_random_forest(MatrixXd x, VectorXd y,
                                      int numRandomTests, int counterThreshold, int maxDepth,
                                      int numTrees, int numEpochs,
                                      bool findTrainError=false,
                                      bool verbose=false) {
  //function uses OnlineRF class to construct a forest and return a field of trees
  // each tree is represented by a matrix.  each row in the matrix is a node in the tree

  //1. call the constructor

  //construct the hyper parameter class object
  Hyperparameters hp;
  hp.numRandomTests = numRandomTests;
  hp.counterThreshold = counterThreshold;
  hp.maxDepth = maxDepth;
  hp.numTrees = numTrees;
  hp.numEpochs = numEpochs;
  hp.findTrainError = findTrainError;
  hp.verbose = verbose;

  //convert data into DataSet class
  DataSet trainData = make_trainData(x, y);

//  Rcout << "trainData rows: " << trainData.m_numSamples << ", cols: " << trainData.m_numFeatures << ", classes:" << trainData.m_numClasses << std::endl ;
  
  
  //construct the forest
  Classifier* orf_ = NULL;
  orf_ = new OnlineRF(hp, trainData.m_numClasses, trainData.m_numFeatures,
                trainData.m_minFeatRange, trainData.m_maxFeatRange);

  //2. apply the training method - train will iterate over all rows
  train(orf_, trainData, hp);

  
  //3. extract forest information into the matrix
  //initialize object to return
  //Rcout << "1: got to here " << std::endl ;
  
   vector<MatrixXd> forest = orf_->getParms();
//    vector<MatrixXd> forest;

   //update the parameters in the forest for extraction through a pointer

  // // = getForestParms(orf_);

  //return
  return(forest);
}

