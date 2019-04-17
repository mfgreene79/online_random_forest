// -*- C++ -*-
/*
 * orf.cpp - Online Random Forests linked with Rcpp to R 

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Written (W) 2019 Michael Greene, mfgreene79@yahoo.com
 * Copyright (C) 2019 Michael Greene

*/

//include the necessary RcppEigen library
#include <Rcpp.h>
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
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>


#include "data.h"
#include "classifier.h"
#include "utilities.h"
#include "hyperparameters.h"
#include "experimenter.h"
#include "online_rf.h"


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
  for (int nSamp = 0; nSamp < x.rows(); ++nSamp) {
    Sample sample;
    sample.x = VectorXd(ds.m_numFeatures);
    sample.id = nSamp;
    sample.w = 1.0;
    sample.y = y(nSamp);
    labels.insert(sample.y);
    for (int nFeat = 0; nFeat < ds.m_numFeatures; ++nFeat) {
      sample.x(nFeat) = x(nSamp, nFeat);
    } //loop nFeat
    ds.m_samples.push_back(sample); // push sample into dataset
  } //loop nSamp
  ds.m_numClasses = labels.size();

//  cout << "numClasses: " << ds.m_numClasses << std::endl;
  ds.findFeatRange();

  return(ds);
}

/////version to apply when using a causal random forest - includes treatment indicators
DataSet make_trainData(MatrixXd x, VectorXd y, VectorXd treat) {
  //creates a DataSet class from matrices x and y
  DataSet ds;
  ds.m_numFeatures = x.cols();
  ds.m_numSamples = x.rows();

  set<int> labels;
  for (int nSamp = 0; nSamp < x.rows(); ++nSamp) {
    Sample sample;
    sample.x = VectorXd(ds.m_numFeatures);
    sample.id = nSamp;
    sample.w = 1.0;
    sample.y = y(nSamp);
    sample.treat = treat(nSamp);
    labels.insert(sample.y);
    for (int nFeat = 0; nFeat < ds.m_numFeatures; ++nFeat) {
      sample.x(nFeat) = x(nSamp, nFeat);
    } //loop nFeat
    ds.m_samples.push_back(sample); // push sample into dataset
  } //loop nSamp
  ds.m_numClasses = labels.size();

  ds.findFeatRange();

  return(ds);
}

DataSet make_testData(MatrixXd x, int numClasses) {
  //creates a DataSet class from matrix x
  DataSet ds;
  ds.m_numFeatures = x.cols();
  ds.m_numSamples = x.rows();
  
  set<int> labels;  
  for (int nSamp = 0; nSamp < x.rows(); ++nSamp) {
    Sample sample;
    sample.x = VectorXd(ds.m_numFeatures);
    sample.id = nSamp;
    sample.w = 1.0;
    for (int nFeat = 0; nFeat < ds.m_numFeatures; ++nFeat) {
      sample.x(nFeat) = x(nSamp, nFeat);
    } //loop nFeat
    ds.m_samples.push_back(sample); // push sample into dataset
  } //loop nSamp
  ds.m_numClasses = numClasses;
  //ds.findFeatRange();
  
  return(ds);
}

List hpToList(Hyperparameters hp) {
  List ret;

  ret["numRandomTests"] = hp.numRandomTests;
  ret["counterThreshold"] = hp.counterThreshold;
  ret["maxDepth"] = hp.maxDepth;
  ret["numTrees"] = hp.numTrees;
  ret["numEpochs"] = hp.numEpochs;
  ret["findTrainError"] = hp.findTrainError;
  ret["verbose"] = hp.verbose;
  ret["method"] = hp.method;
  ret["type"] = hp.type;
  ret["causal"] = hp.causal;
  
  return(ret);
}

// [[Rcpp::export]]
List causal_online_random_forest(MatrixXd x, VectorXd y, VectorXd treat,
				 int numRandomTests, int counterThreshold, int maxDepth,
				 int numTrees, int numEpochs,
				 string type="classification",
				 string method="gini",
				 bool causal=true,
				 bool findTrainError=false,
				 bool verbose=false, bool trainModel=true) {
  //function uses OnlineRF class to construct a forest and return a field of trees
  // each tree is represented by a matrix.  each row in the matrix is a node in the tree

  List ret;
  
  //construct the hyper parameter class object
  Hyperparameters hp;
  hp.numRandomTests = numRandomTests;
  hp.counterThreshold = counterThreshold;
  hp.maxDepth = maxDepth;
  hp.numTrees = numTrees;
  hp.numEpochs = numEpochs;
  hp.type = type;
  hp.method = method;
  hp.causal = causal;
  hp.findTrainError = findTrainError;
  hp.verbose = verbose;
  
  //convert data into DataSet class
  DataSet trainData;
  if(causal == true) {
    trainData = make_trainData(x, y, treat);
  } else {
    trainData = make_trainData(x, y);
  }
  
  //construct the forest
  Classifier* orf_ = NULL;
  orf_ = new OnlineRF(hp, trainData.m_numClasses, trainData.m_numFeatures,
                      trainData.m_minFeatRange, trainData.m_maxFeatRange);

  //apply the training method - train will iterate over all rows
  if(trainModel) {
    train(orf_, trainData, hp);
  }
  //extract forest information into the matrix
  vector<MatrixXd> forest = orf_->exportParms();

  //return a List object with some other basic information
  double oobe, counter; 
  oobe = orf_->getOOBE();
  counter = orf_->getCounter();
  List hp_list = hpToList(hp);

  ret["oobe"] = oobe;
  ret["n"] = counter;
  ret["hyperparameters"] = hp_list;
  ret["numClasses"] = trainData.m_numClasses;

  //Loop through all the trees, putting column names on the matrices
  List outForest;
  for(int numF=0; numF < forest.size(); ++numF) {
    //convert Eigen::MatrixXd to NumericMatrix to export to R
    NumericMatrix outForestMat = wrap(forest[numF]);
    //add column names
    CharacterVector matColNames;
    if(causal == false) {
      matColNames = CharacterVector::create("nodeNumber", "parentNodeNumber", 
					    "rightChildNodeNumber", "leftChildNodeNumber",
					    "depth", "isLeaf","label","counter",
					    "parentCounter","numClasses","numRandomTests");
      for(int i=0; i < trainData.m_numClasses; ++i) {
      	matColNames.push_back("labelStats_" + toString(i));
      }
      matColNames.push_back("bestTest_feature");
      matColNames.push_back("bestTest_threshold");
      for(int i=0; i < trainData.m_numClasses; ++i) {
	      matColNames.push_back("bestTest_trueStats" + toString(i));
      }
      for(int i=0; i < trainData.m_numClasses; ++i) {
	      matColNames.push_back("bestTest_falseStats" + toString(i));
      }
      for(int j=0; j < hp.numRandomTests; ++j) {
	      matColNames.push_back("randomTest" + toString(j) + "_feature");
	      matColNames.push_back("randomTest" + toString(j) + "_threshold");
	      for(int i=0; i < trainData.m_numClasses; ++i) {
	        matColNames.push_back("randomTest" + toString(j) + "_trueStats" + toString(i));
	      }
	      for(int i=0; i < trainData.m_numClasses; ++i) {
	        matColNames.push_back("randomTest" + toString(j) + "_falseStats" + toString(i));
	      }
      }
      colnames(outForestMat) = matColNames;
    } else {//causal==true
      //ADD COLUMN NAMES FOR CAUSAL CASE
      matColNames = CharacterVector::create("nodeNumber", "parentNodeNumber", 
					    "rightChildNodeNumber", "leftChildNodeNumber",
					    "depth", "isLeaf","label","counter");

      for(int i=0; i < trainData.m_numClasses; ++i) {
	      matColNames.push_back("ite_" + toString(i));
      }
      
      matColNames.push_back("treatCounter");
      matColNames.push_back("controlCounter");
      matColNames.push_back("parentCounter");
      matColNames.push_back("numClasses");
      matColNames.push_back("numRandomTests");

      for(int i=0; i < trainData.m_numClasses; ++i) {
	      matColNames.push_back("labelStats_" + toString(i));
      }
      for(int i=0; i < trainData.m_numClasses; ++i) {
	      matColNames.push_back("treatLabelStats_" + toString(i));
      }
      for(int i=0; i < trainData.m_numClasses; ++i) {
	      matColNames.push_back("controlLabelStats_" + toString(i));
      }

      matColNames.push_back("bestTest_feature");
      matColNames.push_back("bestTest_threshold");
      for(int i=0; i < trainData.m_numClasses; ++i) {
	      matColNames.push_back("bestTest_treatTrueStats" + toString(i));
      }
      for(int i=0; i < trainData.m_numClasses; ++i) {
	      matColNames.push_back("bestTest_treatFalseStats" + toString(i));
      }
      for(int i=0; i < trainData.m_numClasses; ++i) {
	      matColNames.push_back("bestTest_controlTrueStats" + toString(i));
      }
      for(int i=0; i < trainData.m_numClasses; ++i) {
	      matColNames.push_back("bestTest_controlFalseStats" + toString(i));
      }

      for(int j=0; j < hp.numRandomTests; ++j) {
	      matColNames.push_back("randomTest" + toString(j) + "_feature");
	      matColNames.push_back("randomTest" + toString(j) + "_threshold");
	      for(int i=0; i < trainData.m_numClasses; ++i) {
	        matColNames.push_back("randomTest" + toString(j) + "_treatTrueStats" + toString(i));
	      }
	      for(int i=0; i < trainData.m_numClasses; ++i) {
	        matColNames.push_back("randomTest" + toString(j) + "_treatFalseStats" + toString(i));
	      }
	      for(int i=0; i < trainData.m_numClasses; ++i) {
	        matColNames.push_back("randomTest" + toString(j) + "_controlTrueStats" + toString(i));
	      }
	      for(int i=0; i < trainData.m_numClasses; ++i) {
	        matColNames.push_back("randomTest" + toString(j) + "_controlFalseStats" + toString(i));
	      }
      }

      colnames(outForestMat) = matColNames;
    } //close causal if
    outForest["tree" + toString(numF)] = outForestMat;
  }

  ret["forest"] = outForest;

  List featList;
  pair<VectorXd, VectorXd> featRange = orf_->getFeatRange();
  featList["minFeatRange"] = featRange.first;
  featList["maxFeatRange"] = featRange.second;

  ret["featRange"] = featList;

  //clean up
  delete orf_;

  return(ret);
}

// [[Rcpp::export]]
List online_random_forest(MatrixXd x, VectorXd y,
			  int numRandomTests, int counterThreshold, int maxDepth,
			  int numTrees, int numEpochs,
			  string type="classification",
			  string method="gini",
			  bool findTrainError=false,
			  bool verbose=false, bool trainModel=true) {
  //makes call to causal random forest setting all treatment indicators to 0
  VectorXd treat = VectorXd::Zero(y.size());
  List corf;
  corf = causal_online_random_forest(x, y, treat,
				     numRandomTests, counterThreshold, 
				     maxDepth,
				     numTrees, numEpochs,
				     type,
				     method,
				     false, //calling with non-causal flag
				     findTrainError,
				     verbose, trainModel);
  return(corf);
}



// [[Rcpp::export]]
List orf(MatrixXd x, VectorXd y, List orfModel, bool trainModel=true) {
  //function uses OnlineRF class to construct a forest and return a field of trees
  // each tree is represented by a matrix.  each row in the matrix is a node in the tree
  // this version of the function will build the ORF from the parameters given

  //Version for NON causal RFs
  
  List ret;
  
  //construct the hyper parameter class object
  List hpList = orfModel["hyperparameters"];
  
  Hyperparameters hp;
  hp.numRandomTests = hpList["numRandomTests"];
  hp.counterThreshold = hpList["counterThreshold"];
  hp.maxDepth = hpList["maxDepth"];
  hp.numTrees = hpList["numTrees"];
  hp.numEpochs = hpList["numEpochs"];
  hp.findTrainError = hpList["findTrainError"];
  hp.verbose = hpList["verbose"];
  hp.method = Rcpp::as<std::string>(hpList["method"]);
  hp.type = Rcpp::as<std::string>(hpList["type"]);
  hp.causal = hpList["causal"]; // should be false

  
  //extract the feature list information that is needed
  List featList = orfModel["featRange"];
  VectorXd minFeatRange = featList["minFeatRange"];
  VectorXd maxFeatRange = featList["maxFeatRange"];
  
  double counter = orfModel["n"];
  double oobe = orfModel["oobe"];
  const int numClasses = orfModel["numClasses"];

  //convert data into DataSet class
  DataSet trainData = make_trainData(x, y);
  //need to fix num classes in the dataset
  trainData.m_numClasses = numClasses;

  //create the vector of matrices that have all the parms
  vector<MatrixXd> forestParms;
  List forestList = orfModel["forest"];
  for(int i=0; i<forestList.size(); ++i) {
    forestParms.push_back(forestList[i]);
  }


  //construct the forest
  Classifier* orf_ = NULL;
  orf_ = new OnlineRF(forestParms, hp, numClasses, oobe, counter, minFeatRange, maxFeatRange);
   
  //update the ORF with feature ranges from the new dataset
  orf_->updateFeatRange(trainData.m_minFeatRange, trainData.m_maxFeatRange);

  pair<VectorXd,VectorXd> featRange = orf_->getFeatRange();

  //apply the training method - train will iterate over all rows
  if(trainModel) {
    train(orf_, trainData, hp);
  }

  //extract forest information into the matrix
  vector<MatrixXd> forest = orf_->exportParms();

  //return a List object with some other basic information
  oobe = orf_->getOOBE();
  counter = orf_->getCounter();
  List hp_list = hpToList(hp);

  ret["numClasses"] = numClasses;
  ret["oobe"] = oobe;
  ret["n"] = counter;
  ret["hyperparameters"] = hp_list;


  //Loop through all the trees, putting column names on the matrices
  List outForest;
  for(int numF=0; numF < forest.size(); ++numF) {
    //convert Eigen::MatrixXd to NumericMatrix to export to R
    NumericMatrix outForestMat = wrap(forest[numF]);
    //add column names

    CharacterVector matColNames;
    matColNames = CharacterVector::create("nodeNumber", "parentNodeNumber", 
					  "rightChildNodeNumber", "leftChildNodeNumber",
					  "depth", "isLeaf","label","counter",
					  "parentCounter","numClasses","numRandomTests");
    for(int i=0; i < trainData.m_numClasses; ++i) {
      matColNames.push_back("labelStats_" + toString(i));
    }
    matColNames.push_back("bestTest_feature");
    matColNames.push_back("bestTest_threshold");
    for(int i=0; i < trainData.m_numClasses; ++i) {
      matColNames.push_back("bestTest_trueStats" + toString(i));
    }
    for(int i=0; i < trainData.m_numClasses; ++i) {
      matColNames.push_back("bestTest_falseStats" + toString(i));
    }
    for(int j=0; j < hp.numRandomTests; ++j) {
      matColNames.push_back("randomTest" + toString(j) + "_feature");
      matColNames.push_back("randomTest" + toString(j) + "_threshold");
      for(int i=0; i < trainData.m_numClasses; ++i) {
	matColNames.push_back("randomTest" + toString(j) + "_trueStats" + toString(i));
      }
      for(int i=0; i < trainData.m_numClasses; ++i) {
	matColNames.push_back("randomTest" + toString(j) + "_falseStats" + toString(i));
      }
    }
    colnames(outForestMat) = matColNames;
    outForest["tree" + toString(numF)] = outForestMat;
  }
  
  ret["forest"] = outForest;
  

  featRange = orf_->getFeatRange();
  featList["minFeatRange"] = featRange.first;
  featList["maxFeatRange"] = featRange.second;
  
  ret["featRange"] = featList;
  
  //clean up
  delete orf_;

  return(ret);
}


// [[Rcpp::export]]
List corf(MatrixXd x, VectorXd y, VectorXd treat, List orfModel, bool trainModel=true) {
  //function uses OnlineRF class to construct a forest and return a field of trees
  // each tree is represented by a matrix.  each row in the matrix is a node in the tree
  // this version of the function will build the ORF from the parameters given

  //Causal Random Forest version

  List ret;
  
  //construct the hyper parameter class object
  List hpList = orfModel["hyperparameters"];
  
  Hyperparameters hp;
  hp.numRandomTests = hpList["numRandomTests"];
  hp.counterThreshold = hpList["counterThreshold"];
  hp.maxDepth = hpList["maxDepth"];
  hp.numTrees = hpList["numTrees"];
  hp.numEpochs = hpList["numEpochs"];
  hp.findTrainError = hpList["findTrainError"];
  hp.verbose = hpList["verbose"];
  hp.method = Rcpp::as<std::string>(hpList["method"]);
  hp.type = Rcpp::as<std::string>(hpList["type"]);
  hp.causal = hpList["causal"]; //should be true

  
  //extract the feature list information that is needed
  List featList = orfModel["featRange"];
  VectorXd minFeatRange = featList["minFeatRange"];
  VectorXd maxFeatRange = featList["maxFeatRange"];
  
  double counter = orfModel["n"];
  double oobe = orfModel["oobe"];
  const int numClasses = orfModel["numClasses"];

  //convert data into DataSet class
  DataSet trainData = make_trainData(x, y, treat);
  //need to fix num classes in the dataset
  trainData.m_numClasses = numClasses;

  //create the vector of matrices that have all the parms
  vector<MatrixXd> forestParms;
  List forestList = orfModel["forest"];
  for(int i=0; i<forestList.size(); ++i) {
    forestParms.push_back(forestList[i]);
  }

  //construct the forest
  Classifier* orf_ = NULL;
  orf_ = new OnlineRF(forestParms, hp, numClasses, oobe, counter, minFeatRange, maxFeatRange);
   
  //update the ORF with feature ranges from the new dataset
  orf_->updateFeatRange(trainData.m_minFeatRange, trainData.m_maxFeatRange);

  pair<VectorXd,VectorXd> featRange = orf_->getFeatRange();

  //apply the training method - train will iterate over all rows
  if(trainModel) {
    train(orf_, trainData, hp);
  }

  //extract forest information into the matrix
  vector<MatrixXd> forest = orf_->exportParms();
  //vector<MatrixXd> forest;

  //return a List object with some other basic information
  oobe = orf_->getOOBE();
  counter = orf_->getCounter();
  List hp_list = hpToList(hp);

  ret["numClasses"] = numClasses;
  ret["oobe"] = oobe;
  ret["n"] = counter;
  ret["hyperparameters"] = hp_list;


  //Loop through all the trees, putting column names on the matrices
  List outForest;
  for(int numF=0; numF < forest.size(); ++numF) {
    //convert Eigen::MatrixXd to NumericMatrix to export to R
    NumericMatrix outForestMat = wrap(forest[numF]);
    //add column names

    CharacterVector matColNames;
    //ADD COLUMN NAMES FOR CAUSAL CASE
    matColNames = CharacterVector::create("nodeNumber", "parentNodeNumber", 
					  "rightChildNodeNumber", "leftChildNodeNumber",
					    "depth", "isLeaf","label","counter");

    for(int i=0; i < trainData.m_numClasses; ++i) {
      matColNames.push_back("ite_" + toString(i));
    }
      
    matColNames.push_back("treatCounter");
    matColNames.push_back("controlCounter");
    matColNames.push_back("parentCounter");
    matColNames.push_back("numClasses");
    matColNames.push_back("numRandomTests");
    
    for(int i=0; i < trainData.m_numClasses; ++i) {
      matColNames.push_back("labelStats_" + toString(i));
    }
    for(int i=0; i < trainData.m_numClasses; ++i) {
      matColNames.push_back("treatLabelStats_" + toString(i));
    }
    for(int i=0; i < trainData.m_numClasses; ++i) {
      matColNames.push_back("controlLabelStats_" + toString(i));
    }
    
    matColNames.push_back("bestTest_feature");
    matColNames.push_back("bestTest_threshold");
    for(int i=0; i < trainData.m_numClasses; ++i) {
      matColNames.push_back("bestTest_treatTrueStats" + toString(i));
    }
    for(int i=0; i < trainData.m_numClasses; ++i) {
      matColNames.push_back("bestTest_treatFalseStats" + toString(i));
    }
    for(int i=0; i < trainData.m_numClasses; ++i) {
      matColNames.push_back("bestTest_controlTrueStats" + toString(i));
    }
    for(int i=0; i < trainData.m_numClasses; ++i) {
      matColNames.push_back("bestTest_controlFalseStats" + toString(i));
    }
    
    for(int j=0; j < hp.numRandomTests; ++j) {
      matColNames.push_back("randomTest" + toString(j) + "_feature");
      matColNames.push_back("randomTest" + toString(j) + "_threshold");
      for(int i=0; i < trainData.m_numClasses; ++i) {
	matColNames.push_back("randomTest" + toString(j) + "_treatTrueStats" + toString(i));
      }
      for(int i=0; i < trainData.m_numClasses; ++i) {
	matColNames.push_back("randomTest" + toString(j) + "_treatFalseStats" + toString(i));
      }
      for(int i=0; i < trainData.m_numClasses; ++i) {
	matColNames.push_back("randomTest" + toString(j) + "_controlTrueStats" + toString(i));
      }
      for(int i=0; i < trainData.m_numClasses; ++i) {
	matColNames.push_back("randomTest" + toString(j) + "_controlFalseStats" + toString(i));
      }
    }

    colnames(outForestMat) = matColNames;
    
    outForest["tree" + toString(numF)] = outForestMat;
  }

  ret["forest"] = outForest;
  

  featRange = orf_->getFeatRange();
  featList["minFeatRange"] = featRange.first;
  featList["maxFeatRange"] = featRange.second;
  
  ret["featRange"] = featList;
  
  //clean up
  delete orf_;

  return(ret);
}


// [[Rcpp::export]]
List predictOrf(MatrixXd x, List orfModel, bool iteAll=false) {
  List ret;
  
   //construct the hyper parameter class object
   List hpList = orfModel["hyperparameters"];
  
  Hyperparameters hp;
  hp.numRandomTests = hpList["numRandomTests"];
  hp.counterThreshold = hpList["counterThreshold"];
  hp.maxDepth = hpList["maxDepth"];
  hp.numTrees = hpList["numTrees"];
  hp.numEpochs = hpList["numEpochs"];
  hp.findTrainError = hpList["findTrainError"];
  hp.verbose = hpList["verbose"];
  hp.method = Rcpp::as<std::string>(hpList["method"]);
  hp.type = Rcpp::as<std::string>(hpList["type"]);
  hp.causal = hpList["causal"];

  //extract the feature list information that is needed
  List featList = orfModel["featRange"];
  VectorXd minFeatRange = featList["minFeatRange"];
  VectorXd maxFeatRange = featList["maxFeatRange"];
  
  //convert data into DataSet class
  DataSet testData = make_testData(x, orfModel["numClasses"]);
 
  //create the vector of matrices that have all the parms
  vector<MatrixXd> forestParms;
  List forestList = orfModel["forest"];
  for(int i=0; i<forestList.size(); ++i) {
    forestParms.push_back(forestList[i]);
  }
  
  double counter = orfModel["n"];
  double oobe = orfModel["oobe"];
  const int numClasses = orfModel["numClasses"];
  
  //construct the forest
  Classifier* orf_ = NULL;
  orf_ = new OnlineRF(forestParms, hp, numClasses, oobe, counter, minFeatRange, maxFeatRange);
 
  //assign predictions to res vector
  vector<Result> res = test(orf_, testData, hp);

  //extract values for output
  MatrixXd resConf(x.rows(), numClasses);
  VectorXd resPred(x.rows());
  MatrixXd resIte(x.rows(), numClasses);

  for(int i=0; i < x.rows(); ++i) {
    //extract confidence and predictions
    resConf.row(i) = res[i].confidence;
    resPred(i) = res[i].prediction;
    
    //if causal extract ITE means and detail
    if(hp.causal == true) {
      resIte.row(i) = res[i].ite;
    }
  }

  if(iteAll == true) {
    if(hp.causal == true) {
      //if causal proceed through to extract ITEs from all trees and transform
      //List - one entry per class
      // each entry a matrix: one row per row of x, one col per tree
      List iteList; // one entry per class
    
      for(int nClass=0; nClass < numClasses; ++nClass) {
        CharacterVector iteNM_colnames;
        MatrixXd iteMat(x.rows(), hp.numTrees); //rows for trees, cols for classes
        for(int nTree=0; nTree < hp.numTrees; ++nTree) {
 	        for(int i=0; i < x.rows(); ++i) {
 	          Result treeRes = res[i];
 	          MatrixXd iteAllTrees = treeRes.iteAllTrees; //capture values for all trees - one col per class one row per tree
 	  //save
         	  iteMat(i, nTree) = iteAllTrees(nTree, nClass);
 	        }
	        iteNM_colnames.push_back("tree" + toString(nTree));
        }
       //save to output List
        NumericMatrix iteNM = wrap(iteMat);
        colnames(iteNM) = iteNM_colnames;
        iteList[toString(nClass)] = iteNM;
      }
      ret["ite_all"] = iteList;
    }
  } //close iteAll==true
  
  NumericMatrix resConfMat = wrap(resConf);
  NumericVector resPredVec = wrap(resPred);
  NumericVector resIteVec;

  CharacterVector resConfMatColNames;
  for(int i = 0; i < numClasses; ++i) {
    resConfMatColNames.push_back(toString(i));
  }
  colnames(resConfMat) = resConfMatColNames;

  //extract information relevant for causal trees
  if(hp.causal == true) {
    //ITE estimates
    NumericVector resIteVec = wrap(resIte);
    colnames(resIteVec) = resConfMatColNames;
    ret["ite"] = resIteVec;
  }
  
  ret["confidence"] = resConfMat;
  ret["prediction"] = resPredVec;
  
  //clean up
  delete orf_;

  return(ret);
}

  
// [[Rcpp::export]]
int roundup(double x) {
  int ret = static_cast<int>(x);
  if(x - static_cast<double>(ret) != 0)
    ++ret;
  
  return(ret);
}
  
// [[Rcpp::export]]
List causal_orf_cv(MatrixXd x, VectorXd y, VectorXd treat, 
	     int numClasses, int numRandomTests, int counterThreshold, 
	     int maxDepth, int numTrees, int numEpochs, int nfolds,
	     std::string type="classification", std::string method="gini") {
  
  //prepare items to return
  List ret;
  
  MatrixXd ret_probs(x.rows(), numClasses); //predicted probabilities
  MatrixXd ret_classes(x.rows(), 1); //predicted classes
  VectorXd ret_actuals(x.rows()); //actuals
  VectorXd ret_acc(x.rows()); //predictions accurate
  MatrixXd ret_tst_acc(nfolds, 3); //aggregate accuracies
  VectorXd ret_treat(x.rows()); // treatment indicators
  MatrixXd ret_ite(x.rows(), numClasses); //individual treatment effects

  //shuffle dataset and split into batches based on number of folds
  //create randomized index vector
  vector<int> randIndex;
  randPerm(y.size(), randIndex);
  
  MatrixXd x_new = x;
  VectorXd y_new = y;
  VectorXd treat_new = treat;
  
  //reorder according to randomized index vector
  for(int i=0; i < y.size(); ++i) {
    x_new.row(randIndex[i]) = x.row(i);
    y_new(randIndex[i]) = y(i);
    treat_new(randIndex[i]) = treat(i);
  }
  
  //partition data into cv-batches
  vector<MatrixXd> cv_batches_x;
  vector<VectorXd> cv_batches_y;
  vector<VectorXd> cv_batches_treat;
  int cv_batch_size = roundup(static_cast<double>(x.rows())/static_cast<double>(nfolds));

  //for each mini batch select appropriate number of rows or max if thats all there is
  for(int k=0;k<nfolds;++k) {
    int start = k * cv_batch_size;
    int stop = (k+1) * cv_batch_size - 1;
    if(stop >= x.rows()) {
      stop = x.rows() - 1;
    }
    int p = stop - start + 1;
    int q = x.cols();
    int i = start;
    int j = 0;
    
    MatrixXd sam_x = x_new.block(i,j,p,q);
    VectorXd sam_y = y_new.segment(i,p);
    VectorXd sam_treat = treat_new.segment(i,p);
    cv_batches_x.push_back(sam_x);
    cv_batches_y.push_back(sam_y);
    cv_batches_treat.push_back(sam_treat);
  } //loop k folds
  
  // for each cv batch - train model on the other batches and make prediction on this batch
  int tst_pos=0;
  for(int k=0;k < nfolds; ++k) {
    // prep data sets
    // test data is the k of interest
    MatrixXd tst_x = cv_batches_x[k];
    VectorXd tst_y = cv_batches_y[k];
    VectorXd tst_treat = cv_batches_treat[k];

    //training data is all the other data except in fold k
    MatrixXd tr_x(x.rows()-tst_x.rows(), x.cols());
    VectorXd tr_y(y.size()-tst_y.rows());
    VectorXd tr_treat(treat.size()-tst_treat.rows());
      
    int pos = 0;
    for(int kprime=0; kprime<nfolds;++kprime) {
      if(kprime != k) {
        MatrixXd cv_batch_x_kprime = cv_batches_x[kprime];
        VectorXd cv_batch_y_kprime = cv_batches_y[kprime];
	VectorXd cv_batch_treat_kprime = cv_batches_treat[kprime];
        for(int i=0; i < cv_batch_x_kprime.rows(); ++i) {
          tr_x.row(pos) = cv_batch_x_kprime.row(i);
          tr_y(pos) = cv_batch_y_kprime(i);
          tr_treat(pos) = cv_batch_treat_kprime(i);
          pos++;
        }
      }
    }
    //Train the models on training data
    List orfModel;
    orfModel = causal_online_random_forest(tr_x, tr_y, tr_treat,
					   numRandomTests, counterThreshold, 
					   maxDepth, numTrees, numEpochs, type, method);
  
    //Predict on the test data
    List preds = predictOrf(tst_x, orfModel);
    MatrixXd conf = preds["confidence"];
    VectorXd cls = preds["prediction"];
    MatrixXd ite = preds["ite"];

    //compare actual values to predictions
    //save predictions into matrices to return
    for(int i = 0; i < conf.rows(); ++i) {
      ret_probs.row(tst_pos) = conf.row(i);
      ret_classes(tst_pos) = cls(i);
      
      ret_actuals(tst_pos) = tst_y(i);
      if(tst_y(i) == cls(i)) {
        ret_acc(tst_pos) = 1;
      } else {
        ret_acc(tst_pos) = 0;
      }

      ret_treat(tst_pos) = tst_treat(i);
      ret_ite.row(tst_pos) = ite.row(i);
      
      ++tst_pos;
    }
  }

  ret["probs"] = ret_probs;
  ret["classes"] = ret_classes;
  ret["actuals"] = ret_actuals;
  ret["accurate"] = ret_acc;
  ret["treat"] = ret_treat;
  ret["ite"] = ret_ite;
  return(ret);
}

// [[Rcpp::export]]
List orf_cv(MatrixXd x, VectorXd y, int numClasses, int numRandomTests, 
	    int counterThreshold, int maxDepth, int numTrees, int numEpochs, int nfolds,
	    std::string type="classification", std::string method="gini") {
  
  //prepare items to return
  List ret;
  
  MatrixXd ret_probs(x.rows(), numClasses); //predicted probabilities
  MatrixXd ret_classes(x.rows(), 1); //predicted classes
  VectorXd ret_actuals(x.rows()); //actuals
  VectorXd ret_acc(x.rows()); //predictions accurate
  MatrixXd ret_tst_acc(nfolds, 3); //aggregate accuracies

  //shuffle dataset and split into batches based on number of folds
  //create randomized index vector
  vector<int> randIndex;
  randPerm(y.size(), randIndex);
  
  MatrixXd x_new = x;
  VectorXd y_new = y;
  
  //reorder according to randomized index vector
  for(int i=0; i < y.size(); ++i) {
    x_new.row(randIndex[i]) = x.row(i);
    y_new(randIndex[i]) = y(i);
  }
  
  //partition data into cv-batches
  vector<MatrixXd> cv_batches_x;
  vector<VectorXd> cv_batches_y;
  int cv_batch_size = roundup(static_cast<double>(x.rows())/static_cast<double>(nfolds));

  //for each mini batch select appropriate number of rows or max if thats all there is
  for(int k=0;k<nfolds;++k) {
    int start = k * cv_batch_size;
    int stop = (k+1) * cv_batch_size - 1;
    if(stop >= x.rows()) {
      stop = x.rows() - 1;
    }
    int p = stop - start + 1;
    int q = x.cols();
    int i = start;
    int j = 0;
    
    MatrixXd sam_x = x_new.block(i,j,p,q);
    VectorXd sam_y = y_new.segment(i,p);
    cv_batches_x.push_back(sam_x);
    cv_batches_y.push_back(sam_y);
  } //loop k folds
  
  // for each cv batch - train model on the other batches and make prediction on this batch
  int tst_pos=0;
  for(int k=0;k < nfolds; ++k) {
    // prep data sets
    // test data is the k of interest
    MatrixXd tst_x = cv_batches_x[k];
    VectorXd tst_y = cv_batches_y[k];

    //training data is all the other data except in fold k
    MatrixXd tr_x(x.rows()-tst_x.rows(), x.cols());
    VectorXd tr_y(y.size()-tst_y.rows());
      
    int pos = 0;
    for(int kprime=0; kprime<nfolds;++kprime) {
      if(kprime != k) {
        MatrixXd cv_batch_x_kprime = cv_batches_x[kprime];
        VectorXd cv_batch_y_kprime = cv_batches_y[kprime];
        for(int i=0; i < cv_batch_x_kprime.rows(); ++i) {
          tr_x.row(pos) = cv_batch_x_kprime.row(i);
          tr_y(pos) = cv_batch_y_kprime(i);
          pos++;
        }
      }
    }
    //Train the models on training data
    List orfModel;
    orfModel = online_random_forest(tr_x, tr_y,
				    numRandomTests, counterThreshold, 
				    maxDepth, numTrees, numEpochs, type, method);
  
    //Predict on the test data
    List preds = predictOrf(tst_x, orfModel);
    MatrixXd conf = preds["confidence"];
    VectorXd cls = preds["prediction"];

    //compare actual values to predictions
    //save predictions into matrices to return
    for(int i = 0; i < conf.rows(); ++i) {
      ret_probs.row(tst_pos) = conf.row(i);
      ret_classes(tst_pos) = cls(i);
      
      ret_actuals(tst_pos) = tst_y(i);
      if(tst_y(i) == cls(i)) {
        ret_acc(tst_pos) = 1;
      } else {
        ret_acc(tst_pos) = 0;
      }

      ++tst_pos;
    }
  }

  ret["probs"] = ret_probs;
  ret["classes"] = ret_classes;
  ret["actuals"] = ret_actuals;
  ret["accurate"] = ret_acc;
  return(ret);
}


// [[Rcpp::export]]
VectorXd getImps_(List orfModel) {
  
   //construct the hyper parameter class object
   List hpList = orfModel["hyperparameters"];
  
  Hyperparameters hp;
  hp.numRandomTests = hpList["numRandomTests"];
  hp.counterThreshold = hpList["counterThreshold"];
  hp.maxDepth = hpList["maxDepth"];
  hp.numTrees = hpList["numTrees"];
  hp.numEpochs = hpList["numEpochs"];
  hp.findTrainError = hpList["findTrainError"];
  hp.verbose = hpList["verbose"];
  hp.method = Rcpp::as<std::string>(hpList["method"]);
  hp.type = Rcpp::as<std::string>(hpList["type"]);
  hp.causal = hpList["causal"];

  //extract the feature list information that is needed
  List featList = orfModel["featRange"];
  VectorXd minFeatRange = featList["minFeatRange"];
  VectorXd maxFeatRange = featList["maxFeatRange"];
  
  //create the vector of matrices that have all the parms
  vector<MatrixXd> forestParms;
  List forestList = orfModel["forest"];
  for(int i=0; i<forestList.size(); ++i) {
    forestParms.push_back(forestList[i]);
  }
  
  double counter = orfModel["n"];
  double oobe = orfModel["oobe"];
  const int numClasses = orfModel["numClasses"];
  
  //construct the forest
  Classifier* orf_ = NULL;
  orf_ = new OnlineRF(forestParms, hp, numClasses, oobe, counter, minFeatRange, maxFeatRange);
  
  //get the feature importances, calculate weighted average
  MatrixXd featImp = orf_->getFeatureImportance();
  VectorXd featImpAvg = VectorXd::Zero(minFeatRange.size());
  for(int i=0; i < minFeatRange.size(); i++) {
    featImpAvg(i) = featImp(i,0) / featImp(i,1);
  }
  
  
  return(featImpAvg);
}
