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
 *  Written (W) 2019 Michael Greene, mfgreene79@yahoo.com
 *  Copyright (C) 2019 Michael Greene

*/

//include the necessary RcppEigen library
#include <Rcpp.h>
#include <RcppEigen.h>
//[[Rcpp::depends(RcppEigen)]]

#include <cstdlib>
#include <iostream>
#include <string>
#include <string.h>

#include <fstream>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <set>

#include "data.h"
#include "utilities.h"
#include "hyperparameters.h"
#include "online_rf.h"


using namespace std;
using namespace Rcpp;

Hyperparameters listToHP(List hpList) {
  //construct the hyper parameter class object
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

  if(hp.causal == true) {
    hp.numTreatments = hpList["numTreatments"];
  }

  return(hp);
}

List causal_online_random_forest_classification(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd treat,
						int numRandomTests, int counterThreshold, int maxDepth,
						int numTrees, int numEpochs,
						std::string method="gini",
						bool findTrainError=false,
						bool verbose=false, bool trainModel=true) {
  //function uses OnlineRF class to construct a forest and return a field of trees
  // each tree is represented by a matrix.  each row in the matrix is a node in the tree

  List ret;
  
  //construct the hyper parameter class object from the function arguments
  Hyperparameters hp;
 
  hp.numRandomTests = numRandomTests;
  hp.counterThreshold = counterThreshold;
  hp.maxDepth = maxDepth;
  hp.numTrees = numTrees;
  hp.numEpochs = numEpochs;
  hp.type = "classification";
  hp.method = method;
  hp.causal = true;
  hp.findTrainError = findTrainError;
  hp.verbose = verbose;
  
  //convert data into DataSet class
  //get numClasses and numTreatments from the training data
  DataSet trainData;
  trainData = DataSet(x, y, treat, hp.type);
  hp.numTreatments=trainData.m_numTreatments;

  //construct the forest
  OnlineRF* orf_ = NULL;
  orf_ = new OnlineRF(hp, trainData.m_numClasses, trainData.m_numFeatures,
                      trainData.m_minFeatRange, trainData.m_maxFeatRange);

  //apply the training method - train will iterate over all rows
  if(trainModel) {
    //    train(orf_, trainData, hp);
    orf_->train(trainData);
  }
  //extract forest information into the matrix
  vector<Eigen::MatrixXd> forest = orf_->exportParms();

  //return a List object with some other basic information
  double oobe, counter; 
  oobe = orf_->getOOBE();
  counter = orf_->getCounter();
  List hp_list = hp.hpToList();

  ret["oobe"] = oobe;
  ret["n"] = counter;
  ret["hyperparameters"] = hp_list;
  ret["numClasses"] = trainData.m_numClasses;

  //Loop through all the trees, putting column names on the matrices
  List outForest;
  for(int numF=0; numF < forest.size(); ++numF) {
    //convert Eigen::Eigen::MatrixXd to NumericMatrix to export to R
    NumericMatrix outForestMat = wrap(forest[numF]);
    //add column names
    CharacterVector matColNames;

    //ADD COLUMN NAMES FOR CAUSAL CASE
    matColNames = CharacterVector::create("nodeNumber", "parentNodeNumber", 
					  "rightChildNodeNumber", "leftChildNodeNumber",
					  "depth", "isLeaf","label","counter");
    
    for(int i=0; i < trainData.m_numClasses; ++i) {
      matColNames.push_back("tauHat_" + toString(i));
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

    //    cout << "outForestMat cols: " << outForestMat.cols() << "\n";
    //    cout << "matColNames length: " << matColNames.size() << "\n";
    colnames(outForestMat) = matColNames;
    
    outForest["tree" + toString(numF)] = outForestMat;
  }
  
  ret["forest"] = outForest;
  
  List featList;
  pair<Eigen::VectorXd, Eigen::VectorXd> featRange = orf_->getFeatRange();
  featList["minFeatRange"] = featRange.first;
  featList["maxFeatRange"] = featRange.second;
  
  ret["featRange"] = featList;
  
  //clean up
  delete orf_;
  
  ret.attr("class") = "orf";
  
  return(ret);
}


List causal_online_random_forest_regression(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd treat,
					    int numRandomTests, int counterThreshold, int maxDepth,
					    int numTrees, int numEpochs,
					    std::string method="gini",
					    bool findTrainError=false,
					    bool verbose=false, bool trainModel=true) {
  //function uses OnlineRF class to construct a forest and return a field of trees
  // each tree is represented by a matrix.  each row in the matrix is a node in the tree
  
  List ret;
  
  //construct the hyper parameter class object from the function arguments
  Hyperparameters hp;
 
  hp.numRandomTests = numRandomTests;
  hp.counterThreshold = counterThreshold;
  hp.maxDepth = maxDepth;
  hp.numTrees = numTrees;
  hp.numEpochs = numEpochs;
  hp.type = "regression";
  hp.method = method;
  hp.causal = true;
  hp.findTrainError = findTrainError;
  hp.verbose = verbose;
  
  //convert data into DataSet class
  //get numClasses and numTreatments from the training data
  DataSet trainData;
  trainData = DataSet(x, y, treat, hp.type);
  hp.numTreatments=trainData.m_numTreatments;
  
  //construct the forest
  OnlineRF* orf_ = NULL;
  orf_ = new OnlineRF(hp, trainData.m_numFeatures,
                      trainData.m_minFeatRange, trainData.m_maxFeatRange);
  
  //apply the training method - train will iterate over all rows
  if(trainModel) {
    orf_->train(trainData);
  }
  //extract forest information into the matrix
  vector<Eigen::MatrixXd> forest = orf_->exportParms();

  //return a List object with some other basic information
  double oobe, counter; 
  oobe = orf_->getOOBE();
  counter = orf_->getCounter();
  List hp_list = hp.hpToList();

  ret["oobe"] = oobe;
  ret["n"] = counter;
  ret["hyperparameters"] = hp_list;
  ret["numClasses"] = trainData.m_numClasses;

  //Loop through all the trees, putting column names on the matrices
  List outForest;
  for(int numF=0; numF < forest.size(); ++numF) {
    //convert Eigen::Eigen::MatrixXd to NumericMatrix to export to R
    NumericMatrix outForestMat = wrap(forest[numF]);
    //add column names
    CharacterVector matColNames;

    //ADD COLUMN NAMES FOR CAUSAL CASE
    matColNames = CharacterVector::create("nodeNumber", "parentNodeNumber", 
					  "rightChildNodeNumber", "leftChildNodeNumber",
					  "depth", "isLeaf","counter", "parentCounter",
					  "yMean","yVar","err");
    
    for(int i=0; i < hp.numTreatments; ++i) {
      matColNames.push_back("tauHat_" + toString(i));
    }
    for(int i=0; i < hp.numTreatments; ++i) {
      matColNames.push_back("tauVarHat_" + toString(i));
    }
    for(int i=0; i < hp.numTreatments; ++i) {
      matColNames.push_back("wCounts_" + toString(i));
    }
    for(int i=0; i < hp.numTreatments; ++i) {
      matColNames.push_back("yStats_" + toString(i));
    }
    for(int i=0; i < hp.numTreatments; ++i) {
      matColNames.push_back("yVarStats_" + toString(i));
    }
    
    matColNames.push_back("bestTest_feature");
    matColNames.push_back("bestTest_threshold");

    for(int j=0; j < hp.numRandomTests; ++j) {
      matColNames.push_back("randomTest" + toString(j) + "_feature");
      matColNames.push_back("randomTest" + toString(j) + "_threshold");
      matColNames.push_back("randomTest" + toString(j) + "_trueYMean");
      matColNames.push_back("randomTest" + toString(j) + "_trueYVar");
      matColNames.push_back("randomTest" + toString(j) + "_trueCount");
      matColNames.push_back("randomTest" + toString(j) + "_trueErr");
      matColNames.push_back("randomTest" + toString(j) + "_falseYMean");
      matColNames.push_back("randomTest" + toString(j) + "_falseYVar");
      matColNames.push_back("randomTest" + toString(j) + "_falseCount");
      matColNames.push_back("randomTest" + toString(j) + "_falseErr");

      for(int nTreat=0;  nTreat < hp.numTreatments; nTreat++) {
	matColNames.push_back("randomTest" + toString(j) + "_trueWCounts" + toString(nTreat));	
      }
      for(int nTreat=0;  nTreat < hp.numTreatments; nTreat++) {
	matColNames.push_back("randomTest" + toString(j) + "_falseWCounts" + toString(nTreat));	
      }
      for(int nTreat=0;  nTreat < hp.numTreatments; nTreat++) {
	matColNames.push_back("randomTest" + toString(j) + "_trueYStats" + toString(nTreat));	
      }
      for(int nTreat=0;  nTreat < hp.numTreatments; nTreat++) {
	matColNames.push_back("randomTest" + toString(j) + "_falseYStats" + toString(nTreat));	
      }
      for(int nTreat=0;  nTreat < hp.numTreatments; nTreat++) {
	matColNames.push_back("randomTest" + toString(j) + "_trueYVarStats" + toString(nTreat));	
      }
      for(int nTreat=0;  nTreat < hp.numTreatments; nTreat++) {
	matColNames.push_back("randomTest" + toString(j) + "_falseYVarStats" + toString(nTreat));	
      }

    }

    colnames(outForestMat) = matColNames;
    
    outForest["tree" + toString(numF)] = outForestMat;
  }
  
  ret["forest"] = outForest;
  
  List featList;
  pair<Eigen::VectorXd, Eigen::VectorXd> featRange = orf_->getFeatRange();
  featList["minFeatRange"] = featRange.first;
  featList["maxFeatRange"] = featRange.second;
  
  ret["featRange"] = featList;
  
  //clean up
  delete orf_;
  
  ret.attr("class") = "orf";
  
  return(ret);
}

// [[Rcpp::export]]
List causal_online_random_forest(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd treat,
				 int numRandomTests, int counterThreshold, int maxDepth,
				 int numTrees, int numEpochs,
				 std::string type="classification",
				 std::string method="gini",
				 bool findTrainError=false,
				 bool verbose=false, bool trainModel=true) {
  //function uses OnlineRF class to construct a forest and return a field of trees
  // each tree is represented by a matrix.  each row in the matrix is a node in the tree
  List ret;

  if(type == "classification") {
    ret = causal_online_random_forest_classification(x, y, treat,
						     numRandomTests, counterThreshold, maxDepth,
						     numTrees, numEpochs,
						     method,
						     findTrainError,
						     verbose, trainModel);
						     
  } else { //regression
    ret = causal_online_random_forest_regression(x, y, treat,
						 numRandomTests, counterThreshold, maxDepth,
						 numTrees, numEpochs,
						 method,
						 findTrainError,
						 verbose, trainModel);
  }

  return(ret);
}


List online_random_forest_classification(Eigen::MatrixXd x, Eigen::VectorXd y,
					 int numRandomTests, int counterThreshold, int maxDepth,
					 int numTrees, int numEpochs,
					 std::string method="gini",
					 bool findTrainError=false,
					 bool verbose=false, bool trainModel=true) {
  //function uses OnlineRF class to construct a forest and return a field of trees
  // each tree is represented by a matrix.  each row in the matrix is a node in the tree
  
  List ret;
  
  //construct the hyper parameter class object from the function arguments
  Hyperparameters hp;
 
  hp.numRandomTests = numRandomTests;
  hp.counterThreshold = counterThreshold;
  hp.maxDepth = maxDepth;
  hp.numTrees = numTrees;
  hp.numEpochs = numEpochs;
  hp.type = "classification";
  hp.method = method;
  hp.causal = false;
  hp.findTrainError = findTrainError;
  hp.verbose = verbose;
  
  //convert data into DataSet class
  //get numClasses and numTreatments from the training data
  DataSet trainData;
  trainData = DataSet(x, y, hp.type);
  
  //construct the forest
  OnlineRF* orf_ = NULL;
  orf_ = new OnlineRF(hp, trainData.m_numClasses, trainData.m_numFeatures,
                      trainData.m_minFeatRange, trainData.m_maxFeatRange);

  //apply the training method - train will iterate over all rows
  if(trainModel) {
    //    train(orf_, trainData, hp);
    orf_->train(trainData);
  }
  //extract forest information into the matrix
  vector<Eigen::MatrixXd> forest = orf_->exportParms();

  //return a List object with some other basic information
  double oobe, counter; 
  oobe = orf_->getOOBE();
  counter = orf_->getCounter();
  List hp_list = hp.hpToList();

  ret["oobe"] = oobe;
  ret["n"] = counter;
  ret["hyperparameters"] = hp_list;
  ret["numClasses"] = trainData.m_numClasses;

  //Loop through all the trees, putting column names on the matrices
  List outForest;
  for(int numF=0; numF < forest.size(); ++numF) {
    //convert Eigen::Eigen::MatrixXd to NumericMatrix to export to R
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
  
  List featList;
  pair<Eigen::VectorXd, Eigen::VectorXd> featRange = orf_->getFeatRange();
  featList["minFeatRange"] = featRange.first;
  featList["maxFeatRange"] = featRange.second;
  
  ret["featRange"] = featList;
  
  //clean up
  delete orf_;
  
  ret.attr("class") = "orf";
  
  return(ret);
}


List online_random_forest_regression(Eigen::MatrixXd x, Eigen::VectorXd y,
					 int numRandomTests, int counterThreshold, int maxDepth,
					 int numTrees, int numEpochs,
					 std::string method="gini",
					 bool findTrainError=false,
					 bool verbose=false, bool trainModel=true) {
  //function uses OnlineRF class to construct a forest and return a field of trees
  // each tree is represented by a matrix.  each row in the matrix is a node in the tree
  
  List ret;
  
  //construct the hyper parameter class object from the function arguments
  Hyperparameters hp;
 
  hp.numRandomTests = numRandomTests;
  hp.counterThreshold = counterThreshold;
  hp.maxDepth = maxDepth;
  hp.numTrees = numTrees;
  hp.numEpochs = numEpochs;
  hp.type = "regression";
  hp.method = method;
  hp.causal = false;
  hp.findTrainError = findTrainError;
  hp.verbose = verbose;
  
  //convert data into DataSet class
  //get numClasses and numTreatments from the training data
  DataSet trainData;
  trainData = DataSet(x, y, hp.type);
  
  //construct the forest
  OnlineRF* orf_ = NULL;
  orf_ = new OnlineRF(hp, trainData.m_numFeatures,
                      trainData.m_minFeatRange, trainData.m_maxFeatRange);

  //apply the training method - train will iterate over all rows
  if(trainModel) {
    //    train(orf_, trainData, hp);
    orf_->train(trainData);
  }
  //extract forest information into the matrix
  vector<Eigen::MatrixXd> forest = orf_->exportParms();

  //return a List object with some other basic information
  double oobe, counter; 
  oobe = orf_->getOOBE();
  counter = orf_->getCounter();
  List hp_list = hp.hpToList();

  ret["oobe"] = oobe;
  ret["n"] = counter;
  ret["hyperparameters"] = hp_list;

  //Loop through all the trees, putting column names on the matrices
  List outForest;
  for(int numF=0; numF < forest.size(); ++numF) {
    //convert Eigen::Eigen::MatrixXd to NumericMatrix to export to R
    NumericMatrix outForestMat = wrap(forest[numF]);

    //add column names
    CharacterVector matColNames;
    matColNames = CharacterVector::create("nodeNumber", "parentNodeNumber", 
					  "rightChildNodeNumber", "leftChildNodeNumber",
					  "depth", "isLeaf","counter", "parentCounter","yMean", "yVar",
					  "err");
    matColNames.push_back("bestTest_feature");
    matColNames.push_back("bestTest_threshold");
    for(int j=0; j < hp.numRandomTests; ++j) {
      matColNames.push_back("randomTest" + toString(j) + "_feature");
      matColNames.push_back("randomTest" + toString(j) + "_threshold");
      matColNames.push_back("randomTest" + toString(j) + "_trueYMean");
      matColNames.push_back("randomTest" + toString(j) + "_trueYVar");
      matColNames.push_back("randomTest" + toString(j) + "_trueYCount");
      matColNames.push_back("randomTest" + toString(j) + "_trueYErr");
      matColNames.push_back("randomTest" + toString(j) + "_falseYMean");
      matColNames.push_back("randomTest" + toString(j) + "_falseYVar");
      matColNames.push_back("randomTest" + toString(j) + "_falseYCount");
      matColNames.push_back("randomTest" + toString(j) + "_falseYErr");
    }
    colnames(outForestMat) = matColNames;
    outForest["tree" + toString(numF)] = outForestMat;
  }
  
  ret["forest"] = outForest;
  
  List featList;
  pair<Eigen::VectorXd, Eigen::VectorXd> featRange = orf_->getFeatRange();
  featList["minFeatRange"] = featRange.first;
  featList["maxFeatRange"] = featRange.second;
  
  ret["featRange"] = featList;
  
  //clean up
  delete orf_;
  
  ret.attr("class") = "orf";
  
  return(ret);
}

// [[Rcpp::export]]
List online_random_forest(Eigen::MatrixXd x, Eigen::VectorXd y,
			  int numRandomTests, int counterThreshold, int maxDepth,
			  int numTrees, int numEpochs,
			  std::string type="classification",
			  std::string method="gini",
			  bool findTrainError=false,
			  bool verbose=false, bool trainModel=true) {

  List ret;

  if(type == "classification") {
    ret = online_random_forest_classification(x, y, 
					      numRandomTests, counterThreshold, maxDepth,
					      numTrees, numEpochs,
					      method,
					      findTrainError,
					      verbose, trainModel);
						     
  } else { //regression
    ret = online_random_forest_regression(x, y, 
					  numRandomTests, counterThreshold, maxDepth,
					  numTrees, numEpochs,
					  method,
					  findTrainError,
					  verbose, trainModel);
  }

  return(ret);  
}




List orfClassification(Eigen::MatrixXd x, Eigen::VectorXd y, List orfModel, bool trainModel=true) {
  //function uses OnlineRF class to construct a forest and return a field of trees
  // each tree is represented by a matrix.  each row in the matrix is a node in the tree
  // this version of the function will build the ORF from the parameters given

  //Version for NON causal RFs
  
  List ret;
  
  //construct the hyper parameter class object
  List hpList = orfModel["hyperparameters"];
  Hyperparameters hp = listToHP(hpList);

  //extract the feature list information that is needed
  List featList = orfModel["featRange"];
  Eigen::VectorXd minFeatRange = featList["minFeatRange"];
  Eigen::VectorXd maxFeatRange = featList["maxFeatRange"];
  
  double counter = orfModel["n"];
  double oobe = orfModel["oobe"];
  const int numClasses = orfModel["numClasses"];

  //convert data into DataSet class
  DataSet trainData(x, y, hp.type);
  //need to fix num classes in the dataset
  trainData.m_numClasses = numClasses;

  //create the vector of matrices that have all the parms
  vector<Eigen::MatrixXd> forestParms;
  List forestList = orfModel["forest"];
  for(int i=0; i<forestList.size(); ++i) {
    forestParms.push_back(forestList[i]);
  }


  //construct the forest
  OnlineRF* orf_ = NULL;
  orf_ = new OnlineRF(forestParms, hp, numClasses, oobe, counter, minFeatRange, maxFeatRange);
   
  //update the ORF with feature ranges from the new dataset
  orf_->updateFeatRange(trainData.m_minFeatRange, trainData.m_maxFeatRange);

  pair<Eigen::VectorXd,Eigen::VectorXd> featRange = orf_->getFeatRange();

  //apply the training method - train will iterate over all rows
  if(trainModel) {
    //train(orf_, trainData, hp);
    orf_->train(trainData);
  }

  //extract forest information into the matrix
  vector<Eigen::MatrixXd> forest = orf_->exportParms();

  //return a List object with some other basic information
  oobe = orf_->getOOBE();
  counter = orf_->getCounter();
  List hp_list = hp.hpToList();

  ret["numClasses"] = numClasses;
  ret["oobe"] = oobe;
  ret["n"] = counter;
  ret["hyperparameters"] = hp_list;


  //Loop through all the trees, putting column names on the matrices
  List outForest;
  for(int numF=0; numF < forest.size(); ++numF) {
    //convert Eigen::Eigen::MatrixXd to NumericMatrix to export to R
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

  ret.attr("class") = "orf";
  return(ret);
}

List orfRegression(Eigen::MatrixXd x, Eigen::VectorXd y, List orfModel, bool trainModel=true) {
  //function uses OnlineRF class to construct a forest and return a field of trees
  // each tree is represented by a matrix.  each row in the matrix is a node in the tree
  // this version of the function will build the ORF from the parameters given

  //Version for NON causal RFs
  // Regression version
  
  List ret;
  
  //construct the hyper parameter class object
  List hpList = orfModel["hyperparameters"];
  Hyperparameters hp = listToHP(hpList);

  //extract the feature list information that is needed
  List featList = orfModel["featRange"];
  Eigen::VectorXd minFeatRange = featList["minFeatRange"];
  Eigen::VectorXd maxFeatRange = featList["maxFeatRange"];
  
  double counter = orfModel["n"];
  double oobe = orfModel["oobe"];

  //convert data into DataSet class
  DataSet trainData(x, y, hp.type);
  //need to fix num classes in the dataset
  trainData.m_numTreatments = hp.numTreatments;

  //create the vector of matrices that have all the parms
  vector<Eigen::MatrixXd> forestParms;
  List forestList = orfModel["forest"];
  for(int i=0; i<forestList.size(); ++i) {
    forestParms.push_back(forestList[i]);
  }


  //construct the forest
   OnlineRF* orf_ = NULL;
   orf_ = new OnlineRF(forestParms, hp, oobe, counter, minFeatRange, maxFeatRange);
   
   //update the ORF with feature ranges from the new dataset
   orf_->updateFeatRange(trainData.m_minFeatRange, trainData.m_maxFeatRange);

   pair<Eigen::VectorXd,Eigen::VectorXd> featRange = orf_->getFeatRange();

  //apply the training method - train will iterate over all rows
   if(trainModel) {
     orf_->train(trainData);
   }

  //extract forest information into the matrix
   vector<Eigen::MatrixXd> forest = orf_->exportParms();

  //return a List object with some other basic information
  oobe = orf_->getOOBE();
  counter = orf_->getCounter();
  List hp_list = hp.hpToList();

  ret["oobe"] = oobe;
  ret["n"] = counter;
  ret["hyperparameters"] = hp_list;

  //Loop through all the trees, putting column names on the matrices
  List outForest;
  for(int numF=0; numF < forest.size(); ++numF) {
    //convert Eigen::Eigen::MatrixXd to NumericMatrix to export to R
    NumericMatrix outForestMat = wrap(forest[numF]);
    //add column names

    CharacterVector matColNames;
    matColNames = CharacterVector::create("nodeNumber", "parentNodeNumber", 
					  "rightChildNodeNumber", "leftChildNodeNumber",
					  "depth", "isLeaf","counter", "parentCounter", "yMean","yVar","err");
    matColNames.push_back("bestTest_feature");
    matColNames.push_back("bestTest_threshold");
    for(int j=0; j < hp.numRandomTests; ++j) {
      matColNames.push_back("randomTest" + toString(j) + "_feature");
      matColNames.push_back("randomTest" + toString(j) + "_threshold");

      matColNames.push_back("randomTest" + toString(j) + "_trueYMean");
      matColNames.push_back("randomTest" + toString(j) + "_trueYVar");
      matColNames.push_back("randomTest" + toString(j) + "_trueCount");
      matColNames.push_back("randomTest" + toString(j) + "_trueErr");

      matColNames.push_back("randomTest" + toString(j) + "_falseYMean");
      matColNames.push_back("randomTest" + toString(j) + "_falseYVar");
      matColNames.push_back("randomTest" + toString(j) + "_falseCount");
      matColNames.push_back("randomTest" + toString(j) + "_falseErr");

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

  ret.attr("class") = "orf";
  return(ret);
}

// [[Rcpp::export]]
List orf(Eigen::MatrixXd x, Eigen::VectorXd y, List orfModel, bool trainModel=true) {
  //function uses OnlineRF class to construct a forest and return a field of trees
  // each tree is represented by a matrix.  each row in the matrix is a node in the tree
  // this version of the function will build the ORF from the parameters given
  List out;
  //construct the hyper parameter class object
  List hpList = orfModel["hyperparameters"];
  Hyperparameters hp = listToHP(hpList);

  if(hp.type == "classification") {
    out = orfClassification(x,y,orfModel,trainModel); 
  } else {
    out = orfRegression(x,y,orfModel,trainModel); 
  }
  return(out);
}

List corfClassification(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd treat, List orfModel, bool trainModel=true) {
  //Causal Random Forest version
  //Classification Version

  List ret;
  
  //construct the hyper parameter class object
  List hpList = orfModel["hyperparameters"];
  Hyperparameters hp = listToHP(hpList);
  
  //extract the feature list information that is needed
  List featList = orfModel["featRange"];
  Eigen::VectorXd minFeatRange = featList["minFeatRange"];
  Eigen::VectorXd maxFeatRange = featList["maxFeatRange"];
  
  double counter = orfModel["n"];
  double oobe = orfModel["oobe"];
  const int numClasses = orfModel["numClasses"];

  //convert data into DataSet class
  DataSet trainData(x, y, treat, hp.type);
  //need to fix num classes in the dataset
  trainData.m_numClasses = numClasses;

  //create the vector of matrices that have all the parms
  vector<Eigen::MatrixXd> forestParms;
  List forestList = orfModel["forest"];
  for(int i=0; i<forestList.size(); ++i) {
    forestParms.push_back(forestList[i]);
  }

  //construct the forest
  OnlineRF* orf_ = NULL;
  orf_ = new OnlineRF(forestParms, hp, numClasses, oobe, counter, minFeatRange, maxFeatRange);
   
  //update the ORF with feature ranges from the new dataset
  orf_->updateFeatRange(trainData.m_minFeatRange, trainData.m_maxFeatRange);

  pair<Eigen::VectorXd,Eigen::VectorXd> featRange = orf_->getFeatRange();

  //apply the training method - train will iterate over all rows
  if(trainModel) {
    orf_->train(trainData);
  }

  //extract forest information into the matrix
  vector<Eigen::MatrixXd> forest = orf_->exportParms();
  //vector<Eigen::MatrixXd> forest;

  //return a List object with some other basic information
  oobe = orf_->getOOBE();
  counter = orf_->getCounter();
  List hp_list = hp.hpToList();
  
  ret["numClasses"] = numClasses;
  ret["oobe"] = oobe;
  ret["n"] = counter;
  ret["hyperparameters"] = hp_list;


  //Loop through all the trees, putting column names on the matrices
  List outForest;
  for(int numF=0; numF < forest.size(); ++numF) {
    //convert Eigen::Eigen::MatrixXd to NumericMatrix to export to R
    NumericMatrix outForestMat = wrap(forest[numF]);
    //add column names

    CharacterVector matColNames;
    //ADD COLUMN NAMES FOR CAUSAL CASE
    matColNames = CharacterVector::create("nodeNumber", "parentNodeNumber", 
					  "rightChildNodeNumber", "leftChildNodeNumber",
					    "depth", "isLeaf","label","counter");

    for(int i=0; i < trainData.m_numClasses; ++i) {
      matColNames.push_back("tauHat_" + toString(i));
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

  ret.attr("class") = "orf";
  return(ret);
}

List corfRegression(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd treat, List orfModel, bool trainModel=true) {
  //Causal Random Forest version
  //Regression Version
  List ret;
  
  //construct the hyper parameter class object
  List hpList = orfModel["hyperparameters"];
  Hyperparameters hp = listToHP(hpList);
  
  //extract the feature list information that is needed
  List featList = orfModel["featRange"];
  Eigen::VectorXd minFeatRange = featList["minFeatRange"];
  Eigen::VectorXd maxFeatRange = featList["maxFeatRange"];
  
  double counter = orfModel["n"];
  double oobe = orfModel["oobe"];

  //convert data into DataSet class
  DataSet trainData(x, y, treat, hp.type);
  trainData.m_numTreatments = hp.numTreatments;
  
  //create the vector of matrices that have all the parms
  vector<Eigen::MatrixXd> forestParms;
  List forestList = orfModel["forest"];
  for(int i=0; i<forestList.size(); ++i) {
    forestParms.push_back(forestList[i]);
  }

  //construct the forest
  OnlineRF* orf_ = NULL;
  orf_ = new OnlineRF(forestParms, hp, oobe, counter, minFeatRange, maxFeatRange);
   
  //update the ORF with feature ranges from the new dataset
  orf_->updateFeatRange(trainData.m_minFeatRange, trainData.m_maxFeatRange);

  pair<Eigen::VectorXd,Eigen::VectorXd> featRange = orf_->getFeatRange();

  //apply the training method - train will iterate over all rows
  if(trainModel) {
    //train(orf_, trainData, hp);
    orf_->train(trainData);
  }

  //extract forest information into the matrix
  vector<Eigen::MatrixXd> forest = orf_->exportParms();

  //return a List object with some other basic information
  oobe = orf_->getOOBE();
  counter = orf_->getCounter();
  List hp_list = hp.hpToList();

  ret["oobe"] = oobe;
  ret["n"] = counter;
  ret["hyperparameters"] = hp_list;


  //Loop through all the trees, putting column names on the matrices
  List outForest;
  for(int numF=0; numF < forest.size(); ++numF) {
    //convert Eigen::Eigen::MatrixXd to NumericMatrix to export to R
    NumericMatrix outForestMat = wrap(forest[numF]);
    //add column names

    CharacterVector matColNames;
    //ADD COLUMN NAMES FOR CAUSAL CASE
    matColNames = CharacterVector::create("nodeNumber", "parentNodeNumber", 
					  "rightChildNodeNumber", "leftChildNodeNumber",
					  "depth", "isLeaf","counter", "parentCounter", "yMean","yVar",
					  "err");

    for(int i=0; i < hp.numTreatments; ++i) {
      matColNames.push_back("tauHat_" + toString(i));
    }
    for(int i=0; i < hp.numTreatments; ++i) {
      matColNames.push_back("tauVarHat_" + toString(i));
    }
    for(int i=0; i < hp.numTreatments; ++i) {
      matColNames.push_back("wCounts_" + toString(i));
    }
    for(int i=0; i < hp.numTreatments; ++i) {
      matColNames.push_back("yStats_" + toString(i));
    }
    for(int i=0; i < hp.numTreatments; ++i) {
      matColNames.push_back("yVarStats_" + toString(i));
    }

    matColNames.push_back("bestTest_feature");
    matColNames.push_back("bestTest_threshold");
    
    for(int j=0; j < hp.numRandomTests; ++j) {
      matColNames.push_back("randomTest" + toString(j) + "_feature");
      matColNames.push_back("randomTest" + toString(j) + "_threshold");

      matColNames.push_back("randomTest" + toString(j) + "_trueYMean");
      matColNames.push_back("randomTest" + toString(j) + "_trueYVar");
      matColNames.push_back("randomTest" + toString(j) + "_trueCount");
      matColNames.push_back("randomTest" + toString(j) + "_trueErr");
      matColNames.push_back("randomTest" + toString(j) + "_falseYMean");
      matColNames.push_back("randomTest" + toString(j) + "_falseYVar");
      matColNames.push_back("randomTest" + toString(j) + "_falseCount");
      matColNames.push_back("randomTest" + toString(j) + "_falseErr");

      for(int i=0; i < hp.numTreatments; ++i) {
	matColNames.push_back("randomTest" + toString(j) + "_trueCount" + toString(i));
      }
      for(int i=0; i < hp.numTreatments; ++i) {
	matColNames.push_back("randomTest" + toString(j) + "_falseCount" + toString(i));
      }
      for(int i=0; i < hp.numTreatments; ++i) {
	matColNames.push_back("randomTest" + toString(j) + "_trueYStats" + toString(i));
      }
      for(int i=0; i < hp.numTreatments; ++i) {
	matColNames.push_back("randomTest" + toString(j) + "_falseYStats" + toString(i));
      }
      for(int i=0; i < hp.numTreatments; ++i) {
	matColNames.push_back("randomTest" + toString(j) + "_trueYVarStats" + toString(i));
      }
      for(int i=0; i < hp.numTreatments; ++i) {
	matColNames.push_back("randomTest" + toString(j) + "_falseYVarStats" + toString(i));
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

  ret.attr("class") = "orf";
  return(ret);
}

// [[Rcpp::export]]
List corf(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd treat, List orfModel, bool trainModel=true) {
  //function uses OnlineRF class to construct a forest and return a field of trees
  // each tree is represented by a matrix.  each row in the matrix is a node in the tree
  // this version of the function will build the ORF from the parameters given
  List out;
  //construct the hyper parameter class object
  List hpList = orfModel["hyperparameters"];
  Hyperparameters hp = listToHP(hpList);

  if(hp.type == "classification") {
    out = corfClassification(x,y,treat,orfModel,trainModel); 
  } else {
    out = corfRegression(x,y,treat,orfModel,trainModel); 
  }
  return(out);
}

List predictOrfClassification(Eigen::MatrixXd x, List orfModel, bool allTrees=false) {
  List ret;
  
  //construct the hyper parameter class object
  List hpList = orfModel["hyperparameters"];
  Hyperparameters hp = listToHP(hpList);

  //extract the feature list information that is needed
  List featList = orfModel["featRange"];
  Eigen::VectorXd minFeatRange = featList["minFeatRange"];
  Eigen::VectorXd maxFeatRange = featList["maxFeatRange"];
  
  //convert data into DataSet class
  const int numClasses = orfModel["numClasses"];

  DataSet testData(x, numClasses);
 
  //create the vector of matrices that have all the parms
  vector<Eigen::MatrixXd> forestParms;
  List forestList = orfModel["forest"];
  for(int i=0; i<forestList.size(); ++i) {
    forestParms.push_back(forestList[i]);
  }
  
  double counter = orfModel["n"];
  double oobe = orfModel["oobe"];
  
  
  //construct the forest
  OnlineRF* orf_ = NULL;
  orf_ = new OnlineRF(forestParms, hp, numClasses, oobe, counter, minFeatRange, maxFeatRange);
 
  //assign predictions to res vector
  vector<Result> res = orf_->test(testData);

  //extract values for output
  Eigen::MatrixXd resConf(x.rows(), numClasses);
  Eigen::VectorXd resPred(x.rows());
  Eigen::MatrixXd resTauHat(x.rows(), numClasses);

  for(int i=0; i < x.rows(); ++i) {
    //extract confidence and predictions
    resConf.row(i) = res[i].confidence;
    resPred(i) = res[i].predictionClassification;
    
    //if causal extract ITE means and detail
    if(hp.causal == true) {
      resTauHat.row(i) = res[i].tauHat;
    }
  }

  if(allTrees == true) {
    if(hp.causal == true) {
      //if causal proceed through to extract ITEs from all trees and transform
      //List - one entry per class
      // each entry a matrix: one row per row of x, one col per tree
      List tauHatList; // one entry per class
    
      for(int nClass=0; nClass < numClasses; ++nClass) {
        CharacterVector tauHatNM_colnames;
        Eigen::MatrixXd tauHatMat(x.rows(), hp.numTrees); //rows for trees, cols for classes
        for(int nTree=0; nTree < hp.numTrees; ++nTree) {
 	        for(int i=0; i < x.rows(); ++i) {
 	          Result treeRes = res[i];
 	          Eigen::MatrixXd tauHatAllTrees = treeRes.tauHatAllTrees; //capture values for all trees - one col per class one row per tree
 	  //save
         	  tauHatMat(i, nTree) = tauHatAllTrees(nTree, nClass);
 	        }
	        tauHatNM_colnames.push_back("tree" + toString(nTree));
        }
       //save to output List
        NumericMatrix tauHatNM = wrap(tauHatMat);
        colnames(tauHatNM) = tauHatNM_colnames;
        tauHatList[toString(nClass)] = tauHatNM;
      }
      ret["tauHatAllTrees"] = tauHatList;
    }
  } //close allTrees==true
  
  NumericMatrix resConfMat = wrap(resConf);
  NumericVector resPredVec = wrap(resPred);
  NumericVector resTauHatVec;

  CharacterVector resConfMatColNames;
  for(int i = 0; i < numClasses; ++i) {
    resConfMatColNames.push_back(toString(i));
  }
  colnames(resConfMat) = resConfMatColNames;

  //extract information relevant for causal trees
  if(hp.causal == true) {
    //ITE estimates
    NumericVector resTauHatVec = wrap(resTauHat);
    colnames(resTauHatVec) = resConfMatColNames;
    ret["tauHat"] = resTauHatVec;
  }
  
  ret["confidence"] = resConfMat;
  ret["prediction"] = resPredVec;
  
  //clean up
  delete orf_;

  return(ret);
}

List predictOrfRegression(Eigen::MatrixXd x, List orfModel, bool allTrees=false) {
  List ret;
  
  //construct the hyper parameter class object
  List hpList = orfModel["hyperparameters"];
  Hyperparameters hp = listToHP(hpList);

  //extract the feature list information that is needed
  List featList = orfModel["featRange"];
  Eigen::VectorXd minFeatRange = featList["minFeatRange"];
  Eigen::VectorXd maxFeatRange = featList["maxFeatRange"];
  
  //convert data into DataSet class
  DataSet testData(x);
 
  //create the vector of matrices that have all the parms
  vector<Eigen::MatrixXd> forestParms;
  List forestList = orfModel["forest"];
  for(int i=0; i<forestList.size(); ++i) {
    forestParms.push_back(forestList[i]);
  }
  
  double counter = orfModel["n"];
  double oobe = orfModel["oobe"];

  //construct the forest
  OnlineRF* orf_ = NULL;
  orf_ = new OnlineRF(forestParms, hp, oobe, counter, minFeatRange, maxFeatRange);
 
  //assign predictions to res vector
  vector<Result> res = orf_->test(testData);

  //extract values for output
  Eigen::VectorXd resVar(x.rows());
  Eigen::VectorXd resPred(x.rows());
  Eigen::MatrixXd resTauHat(x.rows(), hp.numTreatments);
  Eigen::MatrixXd resTauVarHat(x.rows(), hp.numTreatments);

  for(int i=0; i < x.rows(); i++) {
    //extract prediction and variance estimate
    resPred(i) = res[i].predictionRegression;
    resVar(i) = res[i].predictionVarianceRegression;
    
    //if causal extract ITE means and detail
    if(hp.causal == true) {
      resTauHat.row(i) = res[i].tauHat;
      resTauVarHat.row(i) = res[i].tauVarHat;
    }
  }


  if(allTrees == true) {
    //copy yHat for all trees 
    Eigen::MatrixXd yHatAllTrees(x.rows(), hp.numTrees);
    for(int i=0; i < x.rows(); i++) {
      yHatAllTrees.row(i) = res[i].yHatAllTrees;
    }
    NumericMatrix resYHatAllMat = wrap(yHatAllTrees);

    //add column labels
    CharacterVector treeNames;
    for(int nTree=0; nTree < hp.numTrees; nTree++) {
      treeNames.push_back("tree" + toString(nTree));
    }
    colnames(resYHatAllMat) = treeNames;
    ret["yHatAllTrees"] = resYHatAllMat;

    if(hp.causal == true) {//if causal proceed through to extract ITEs from all trees and transform
      //List - one entry per treatment
      // each entry a matrix: one row per row of x, one col per tree
      //    each entry in the matrix is the difference between the treatment and control
      List tauHatList; // one entry per treatment
   
      for(int nTreat=0; nTreat < hp.numTreatments; nTreat++) {
        CharacterVector tauHatNM_colnames;
        Eigen::MatrixXd tauHatMat(x.rows(), hp.numTrees); //rows for data, cols for trees
	for(int i=0; i < x.rows(); i++) {
	  Result treeRes = res[i];
	  //treeRes.tauHatAllTrees has one row per tree and one col per treatment
	  Eigen::MatrixXd tauHatAllTrees = treeRes.tauHatAllTrees; //capture values for all trees - one col per treatment one row per tree
	  //save with one row per obs and one col per tree
	  tauHatMat.row(i) = tauHatAllTrees.col(nTreat).transpose();
	}

	//save to output List
        NumericMatrix tauHatNM = wrap(tauHatMat);
        colnames(tauHatNM) = treeNames;
        tauHatList["treatment"+toString(nTreat)] = tauHatNM;
      } //loop nTreat
      ret["tauHat_all"] = tauHatList;
    } //close causal==true
  } //close allTrees==true
  
  NumericVector resPredVec = wrap(resPred);
  NumericVector resVarVec = wrap(resVar);
  NumericMatrix resTauHatMat;
  NumericMatrix resTauVarHatMat;

  //extract information relevant for causal trees
  if(hp.causal == true) {
    //ITE estimates
     resTauHatMat = wrap(resTauHat);
     resTauVarHatMat = wrap(resTauVarHat);

    //add column names for tauhat and tauhatvar matrices
     CharacterVector tauHat_colnames;
     for(int nTreat=0; nTreat < hp.numTreatments; nTreat++) {
       tauHat_colnames.push_back("treatment" + toString(nTreat));
     }
     colnames(resTauHatMat) = tauHat_colnames;
     colnames(resTauVarHatMat) = tauHat_colnames;

     ret["tauHat"] = resTauHatMat;
     ret["tauVarHat"] = resTauVarHatMat;
  }
  
  ret["prediction"] = resPredVec;
  ret["variance"] = resVarVec;
  
  //clean up
  delete orf_;

  return(ret);
}

// [[Rcpp::export]]
List predictOrf(Eigen::MatrixXd x, List orfModel, bool allTrees=false) {
  List hpList = orfModel["hyperparameters"];
  Hyperparameters hp = listToHP(hpList);
  List ret;
  if(hp.type == "classification") {
    ret = predictOrfClassification(x, orfModel, allTrees);
  } else {
    ret = predictOrfRegression(x, orfModel, allTrees);
  }

  return(ret);
}
  
  
// [[Rcpp::export]]
List causal_orf_cv(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd treat, 
	     int numClasses, int numRandomTests, int counterThreshold, 
	     int maxDepth, int numTrees, int numEpochs, int nfolds,
	     std::string type="classification", std::string method="gini") {
  
  //prepare items to return
  List ret;
  
  Eigen::MatrixXd ret_probs(x.rows(), numClasses); //predicted probabilities
  Eigen::MatrixXd ret_classes(x.rows(), 1); //predicted classes
  Eigen::VectorXd ret_actuals(x.rows()); //actuals
  Eigen::VectorXd ret_acc(x.rows()); //predictions accurate
  Eigen::MatrixXd ret_tst_acc(nfolds, 3); //aggregate accuracies
  Eigen::VectorXd ret_treat(x.rows()); // treatment indicators
  Eigen::MatrixXd ret_tauHat(x.rows(), numClasses); //individual treatment effects

  //shuffle dataset and split into batches based on number of folds
  //create randomized index vector
  vector<int> randIndex;
  randPerm(y.size(), randIndex);
  
  Eigen::MatrixXd x_new = x;
  Eigen::VectorXd y_new = y;
  Eigen::VectorXd treat_new = treat;
  
  //reorder according to randomized index vector
  for(int i=0; i < y.size(); ++i) {
    x_new.row(randIndex[i]) = x.row(i);
    y_new(randIndex[i]) = y(i);
    treat_new(randIndex[i]) = treat(i);
  }
  
  //partition data into cv-batches
  vector<Eigen::MatrixXd> cv_batches_x;
  vector<Eigen::VectorXd> cv_batches_y;
  vector<Eigen::VectorXd> cv_batches_treat;
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
    
    Eigen::MatrixXd sam_x = x_new.block(i,j,p,q);
    Eigen::VectorXd sam_y = y_new.segment(i,p);
    Eigen::VectorXd sam_treat = treat_new.segment(i,p);
    cv_batches_x.push_back(sam_x);
    cv_batches_y.push_back(sam_y);
    cv_batches_treat.push_back(sam_treat);
  } //loop k folds
  
  // for each cv batch - train model on the other batches and make prediction on this batch
  int tst_pos=0;
  for(int k=0;k < nfolds; ++k) {
    // prep data sets
    // test data is the k of interest
    Eigen::MatrixXd tst_x = cv_batches_x[k];
    Eigen::VectorXd tst_y = cv_batches_y[k];
    Eigen::VectorXd tst_treat = cv_batches_treat[k];

    //training data is all the other data except in fold k
    Eigen::MatrixXd tr_x(x.rows()-tst_x.rows(), x.cols());
    Eigen::VectorXd tr_y(y.size()-tst_y.rows());
    Eigen::VectorXd tr_treat(treat.size()-tst_treat.rows());
      
    int pos = 0;
    for(int kprime=0; kprime<nfolds;++kprime) {
      if(kprime != k) {
        Eigen::MatrixXd cv_batch_x_kprime = cv_batches_x[kprime];
        Eigen::VectorXd cv_batch_y_kprime = cv_batches_y[kprime];
	Eigen::VectorXd cv_batch_treat_kprime = cv_batches_treat[kprime];
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
    Eigen::MatrixXd conf = preds["confidence"];
    Eigen::VectorXd cls = preds["prediction"];
    Eigen::MatrixXd tauHat = preds["tauHat"];

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
      ret_tauHat.row(tst_pos) = tauHat.row(i);
      
      ++tst_pos;
    }
  }

  ret["probs"] = ret_probs;
  ret["classes"] = ret_classes;
  ret["actuals"] = ret_actuals;
  ret["accurate"] = ret_acc;
  ret["treat"] = ret_treat;
  ret["tauHat"] = ret_tauHat;
  return(ret);
}

// [[Rcpp::export]]
List orf_cv(Eigen::MatrixXd x, Eigen::VectorXd y, int numClasses, int numRandomTests, 
	    int counterThreshold, int maxDepth, int numTrees, int numEpochs, int nfolds,
	    std::string type="classification", std::string method="gini") {
  
  //prepare items to return
  List ret;
  
  Eigen::MatrixXd ret_probs(x.rows(), numClasses); //predicted probabilities
  Eigen::MatrixXd ret_classes(x.rows(), 1); //predicted classes
  Eigen::VectorXd ret_actuals(x.rows()); //actuals
  Eigen::VectorXd ret_acc(x.rows()); //predictions accurate
  Eigen::MatrixXd ret_tst_acc(nfolds, 3); //aggregate accuracies

  //shuffle dataset and split into batches based on number of folds
  //create randomized index vector
  vector<int> randIndex;
  randPerm(y.size(), randIndex);
  
  Eigen::MatrixXd x_new = x;
  Eigen::VectorXd y_new = y;
  
  //reorder according to randomized index vector
  for(int i=0; i < y.size(); ++i) {
    x_new.row(randIndex[i]) = x.row(i);
    y_new(randIndex[i]) = y(i);
  }
  
  //partition data into cv-batches
  vector<Eigen::MatrixXd> cv_batches_x;
  vector<Eigen::VectorXd> cv_batches_y;
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
    
    Eigen::MatrixXd sam_x = x_new.block(i,j,p,q);
    Eigen::VectorXd sam_y = y_new.segment(i,p);
    cv_batches_x.push_back(sam_x);
    cv_batches_y.push_back(sam_y);
  } //loop k folds
  
  // for each cv batch - train model on the other batches and make prediction on this batch
  int tst_pos=0;
  for(int k=0;k < nfolds; ++k) {
    // prep data sets
    // test data is the k of interest
    Eigen::MatrixXd tst_x = cv_batches_x[k];
    Eigen::VectorXd tst_y = cv_batches_y[k];

    //training data is all the other data except in fold k
    Eigen::MatrixXd tr_x(x.rows()-tst_x.rows(), x.cols());
    Eigen::VectorXd tr_y(y.size()-tst_y.rows());
      
    int pos = 0;
    for(int kprime=0; kprime<nfolds;++kprime) {
      if(kprime != k) {
        Eigen::MatrixXd cv_batch_x_kprime = cv_batches_x[kprime];
        Eigen::VectorXd cv_batch_y_kprime = cv_batches_y[kprime];
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
    Eigen::MatrixXd conf = preds["confidence"];
    Eigen::VectorXd cls = preds["prediction"];

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
Eigen::VectorXd getImps_(List orfModel) {
  
  //construct the hyper parameter class object
  List hpList = orfModel["hyperparameters"];
  Hyperparameters hp = listToHP(hpList);

  //extract the feature list information that is needed
  List featList = orfModel["featRange"];
  Eigen::VectorXd minFeatRange = featList["minFeatRange"];
  Eigen::VectorXd maxFeatRange = featList["maxFeatRange"];
  
  //create the vector of matrices that have all the parms
  vector<Eigen::MatrixXd> forestParms;
  List forestList = orfModel["forest"];
  for(int i=0; i<forestList.size(); ++i) {
    forestParms.push_back(forestList[i]);
  }
  
  double counter = orfModel["n"];
  double oobe = orfModel["oobe"];
  int numClasses = 0;
  OnlineRF* orf_ = NULL;

  if(hp.type=="classification") {
    numClasses = orfModel["numClasses"];
    orf_ = new OnlineRF(forestParms, hp, numClasses, oobe, counter, minFeatRange, maxFeatRange);
  } else {
    orf_ = new OnlineRF(forestParms, hp, oobe, counter, minFeatRange, maxFeatRange);
  }
  
  //get the feature importances, weighted average calculated by the method
  Eigen::MatrixXd featImp = orf_->getFeatureImportance();
  
  return(featImp.col(0));
}
