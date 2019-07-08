// -*- C++ -*-
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Amir Saffari, amir@ymer.org
 * Copyright (C) 2010 Amir Saffari, 
 *                    Institute for Computer Graphics and Vision, 
 *                    Graz University of Technology, Austria
 * Modified 2019 Michael Greene, mfgreene79@yahoo.com
 *  added functionality and enabled ability to connect to R, 
 *  added causal forest functionality, regression forest functionality

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef ONLINERF_H_
#define ONLINERF_H_

#include "data.h"
#include "hyperparameters.h"
#include "utilities.h"

class RandomTest {
public:

  //////Classification Tree Versions
  //Version to initialize with randomization
  RandomTest(const Hyperparameters& hp, const int& numClasses, const int& numFeatures, 
	     const Eigen::VectorXd &minFeatRange, const Eigen::VectorXd &maxFeatRange,
	     const Eigen::VectorXd &rootLabelStats, const double &rootCounter);

  //Version to initialize from a known feature/threshold - not causal
  RandomTest(const Hyperparameters& hp, const int& numClasses, 
	     int feature, double threshold,
	     Eigen::VectorXd trueStats, Eigen::VectorXd falseStats,
	     const Eigen::VectorXd &rootLabelStats, const double &rootCounter);

  //Version to initialize from a known feature/threshold - causal 
  RandomTest(const Hyperparameters& hp, const int& numClasses, 
	     int feature, double threshold,
	     Eigen::VectorXd treatTrueStats, Eigen::VectorXd treatFalseStats,
	     Eigen::VectorXd controlTrueStats, Eigen::VectorXd controlFalseStats,
	     const Eigen::VectorXd &rootLabelStats, const double &rootCounter
	     );

  //////Regression Tree Versions
  //Version to initialize with randomization
  RandomTest(const Hyperparameters& hp, const int& numFeatures, 
	     const Eigen::VectorXd &minFeatRange, const Eigen::VectorXd &maxFeatRange,
	     const Eigen::VectorXd &rootYStats, const double &rootCounter);

  //Version to initialize from a known feature/threshold - not causal
  RandomTest(const Hyperparameters& hp, 
	     int feature, double threshold,
	     double trueYMean, double falseYMean,
	     double trueYVar, double falseYVar,
	     int trueCount, int falseCount,
	     double trueErr, double falseErr,
	     const Eigen::VectorXd &rootYStats, const double &rootCounter);

  //Version to initialize from a known feature/threshold - causal
  RandomTest(const Hyperparameters& hp, 
	     int feature, double threshold,
	     double trueYMean, double falseYMean,
	     double trueYVar, double falseYVar,
	     int trueCount, int falseCount,
	     double trueErr, double falseErr,
	     Eigen::VectorXd trueWCounts, Eigen::VectorXd falseWCounts,
	     Eigen::VectorXd trueYStats, Eigen::VectorXd falseYStats,
	     Eigen::VectorXd trueYVarStats, Eigen::VectorXd falseYVarStats,
	     const Eigen::VectorXd &rootYStats, const double &rootCounter);
  
  void update(const Sample& sample);
  
  bool eval(const Sample& sample) const;
  
  double score() const;
  double scoreClassification() const;
  double scoreRegression() const;
  
  
  //// Methods to fetch the statistics from the Random Test
  pair<int,double> getParms();
  pair<Eigen::VectorXd, Eigen::VectorXd > getStatsClassification(std::string type = "all") const;
  pair<int, int> getTotCounts();
  pair<double, double> getYMeans();
  pair<double, double> getYVars();
  pair<double, double> getErrs();
  pair<Eigen::VectorXd, Eigen::VectorXd> getWCounts();
  pair<Eigen::VectorXd, Eigen::VectorXd> getYStats();
  pair<Eigen::VectorXd, Eigen::VectorXd> getYVarStats();

//   void print();

    
 protected:
  const Hyperparameters* m_hp;
  const int* m_numClasses;
  const Eigen::VectorXd* m_rootLabelStats;
  const Eigen::VectorXd* m_rootYStats;
  const double* m_rootCounter;
  int m_feature;
  double m_threshold;

  //total counts and stats
  int m_trueCount;
  int m_falseCount;
  Eigen::VectorXd m_trueStats;
  Eigen::VectorXd m_falseStats;
    
  //treatment counts and stats
  int m_treatTrueCount;
  int m_treatFalseCount;
  Eigen::VectorXd m_treatTrueStats;
  Eigen::VectorXd m_treatFalseStats;

  //control counts and stats
  int m_controlTrueCount;
  int m_controlFalseCount;
  Eigen::VectorXd m_controlTrueStats;
  Eigen::VectorXd m_controlFalseStats;

  //Regression Tree versions
  double m_trueYMean; //mean of y for right side 
  double m_falseYMean; //mean of y for left side 
  double m_trueYVar; //mean of y for right side 
  double m_falseYVar; //mean of y for left side 
  double m_trueErr; //total error for right side
  double m_falseErr; //total error for left side
  Eigen::VectorXd m_trueWCounts; //vector of counts by treatment condition for right side. length numTreatments
  Eigen::VectorXd m_falseWCounts;//vector of counts by treatment condition for left side. length numTreatments
  Eigen::VectorXd m_trueYStats; //vector of means of y by treatment condition for right side. length numTreatments
  Eigen::VectorXd m_falseYStats;//vector of means of y by treatment condition for left side. length numTreatments
  Eigen::VectorXd m_trueYVarStats; //vector of variance of y by treatment condition for right side. length numTreatments
  Eigen::VectorXd m_falseYVarStats;//vector of variance of y by treatment condition for left side. length numTreatments

  //Methods for updating the statistics
  void updateStats(const Sample& sample, const bool& decision);
  void updateStatsClassification(const Sample& sample, const bool& decision);
  void updateStatsRegression(const Sample& sample, const bool& decision);
};

class OnlineNode {
public:
  //Classification Forest Constructors
  // version to initialize the root node
  OnlineNode(const Hyperparameters& hp, const int& numClasses, const int& numFeatures, 
	     const Eigen::VectorXd& minFeatRange, const Eigen::VectorXd& maxFeatRange, 
	     const int& depth, int& numNodes);

  //version to initialize versions below the root node - not causal
  OnlineNode(const Hyperparameters& hp, const int& numClasses, const int& numFeatures, 
	     const Eigen::VectorXd& minFeatRange, const Eigen::VectorXd& maxFeatRange, 
	     const int& depth, const Eigen::VectorXd& parentStats, 
	     int nodeNumber, int parentNodeNumber, int& numNodes,
	     const Eigen::VectorXd &rootLabelStats, const double &rootCounter);

  //version to initialize versions below the root node - causal
  OnlineNode(const Hyperparameters& hp, const int& numClasses, const int& numFeatures, 
	     const Eigen::VectorXd& minFeatRange, const Eigen::VectorXd& maxFeatRange, 
	     const int& depth, const Eigen::VectorXd& treatParentStats, 
	     const Eigen::VectorXd& controlParentStats, 
	     int nodeNumber, int parentNodeNumber, int& numNodes,
	     const Eigen::VectorXd &rootLabelStats, const double &rootCounter);
  
  //Version to initialize from a vector of information about the node - root node
  OnlineNode(const Eigen::VectorXd& nodeParms, const Hyperparameters& hp,
	     const int& numClasses, int& numNodes,
	     const Eigen::VectorXd& minFeatRange, const Eigen::VectorXd& maxFeatRange);

  //Version to initialize from a vector of information about the node - below the root
  OnlineNode(const Eigen::VectorXd& nodeParms, const Hyperparameters& hp,
	     const int& numClasses, int& numNodes,
	     const Eigen::VectorXd& minFeatRange, const Eigen::VectorXd& maxFeatRange,
	     const Eigen::VectorXd &rootLabelStats, const double &rootCounter);


  //Regression Forest Constructors
  // version to initialize the root node
  OnlineNode(const Hyperparameters& hp, const int& numFeatures, 
	     const Eigen::VectorXd& minFeatRange, const Eigen::VectorXd& maxFeatRange, 
	     const int& depth, int& numNodes);

  //version to initialize versions below the root node - not causal
  OnlineNode(const Hyperparameters& hp, const int& numFeatures, 
	     const Eigen::VectorXd& minFeatRange, 
	     const Eigen::VectorXd& maxFeatRange, 
	     const int& depth,
	     const double parentCounter,
	     const double parentYMean,
	     const double parentYVar,
	     const double parentErr,
	     int nodeNumber, int parentNodeNumber, int& numNodes,
	     const Eigen::VectorXd &rootYStats, const double &rootCounter);
  
  //version to initialize below the root - causal
  OnlineNode(const Hyperparameters& hp, const int& numFeatures, 
	     const Eigen::VectorXd& minFeatRange, 
	     const Eigen::VectorXd& maxFeatRange, 
	     const int& depth,
	     const double parentCounter,
	     const double parentYMean,
	     const double parentYVar,
	     const double parentErr,
	     const Eigen::VectorXd& parentWCounts,
	     const Eigen::VectorXd& parentYStats,
	     const Eigen::VectorXd& parentYVarStats,
	     int nodeNumber, int parentNodeNumber, int& numNodes,
	     const Eigen::VectorXd &rootYStats, const double &rootCounter);

  //Version to initialize from a vector of information about the node - root version
  OnlineNode(const Eigen::VectorXd& nodeParms, const Hyperparameters& hp,
	     int& numNodes, const Eigen::VectorXd& minFeatRange, 
	     const Eigen::VectorXd& maxFeatRange);

  //Version to initialize from a vector of information about the node - below root version
  OnlineNode(const Eigen::VectorXd& nodeParms, const Hyperparameters& hp,
	     int& numNodes, const Eigen::VectorXd& minFeatRange, 
	     const Eigen::VectorXd& maxFeatRange,
	     const Eigen::VectorXd &rootYStats, const double &rootCounter);

  ~OnlineNode();
    
  //update with data
  //  void update(const Sample& sample);
  void updateClassification(const Sample& sample);
  void updateRegression(const Sample& sample);
  void update(const Sample& sample);
  //evaluate based on a new data point
  void eval(const Sample& sample, Result& result);
  void evalRegression(const Sample& sample, Result& result);
  void evalClassification(const Sample& sample, Result& result);

  //version to grow the node recursively from a matrix of information

  void update(const Eigen::MatrixXd& treeParms);
  void updateClassification(const Eigen::MatrixXd& treeParms);
  void updateRegression(const Eigen::MatrixXd& treeParms);

  //set child node numbers if the split occurs
  void setChildNodeNumbers(int rightChildNodeNumber, int leftChildNodeNumber);

  //method to add nodeParms to the matrix of parms for the tree
  Eigen::VectorXd exportParms(); //export parms out to a vector
  Eigen::VectorXd exportParmsClassification(); //export parms out to a vector
  Eigen::VectorXd exportParmsClassificationCausal(); //export parms out to a vector
  Eigen::VectorXd exportParmsRegression(); //export parms out to a vector
  Eigen::VectorXd exportParmsRegressionCausal(); //export parms out to a vector
  
  //recursive function to add elements to the vector for each child node
  void exportChildParms(vector<Eigen::VectorXd> &treeParmsVector);

  //function to score node with labelstats
  double scoreClassification();
  double scoreRegression();
  double score();
  double getCount(); 

//   void printInfo();
//   void print();


  //recursive function to get feature importances
  Eigen::MatrixXd getFeatureImportance();


 private:
  int m_nodeNumber;
  int m_parentNodeNumber;
  int m_rightChildNodeNumber;
  int m_leftChildNodeNumber;
  const int* m_numClasses;
  int m_depth;
  bool m_isLeaf;
  const Hyperparameters* m_hp;
  int m_label;
  double m_counter;
  double m_treatCounter;
  double m_controlCounter;
  double m_parentCounter;
  Eigen::VectorXd m_labelStats;
  Eigen::VectorXd m_treatLabelStats;
  Eigen::VectorXd m_controlLabelStats;
  const Eigen::VectorXd* m_minFeatRange;
  const Eigen::VectorXd* m_maxFeatRange;
  
  OnlineNode* m_leftChildNode;
  OnlineNode* m_rightChildNode;
  
  vector<RandomTest*> m_onlineTests;
  RandomTest* m_bestTest;

  int* m_numNodes; //pointer to tree for number of nodes
  const Eigen::VectorXd* m_rootLabelStats;
  const Eigen::VectorXd* m_rootYStats;
  const double* m_rootCounter;

  //regression tree info
  //alternate specification - vectors of treatment conditions.  place 0 specifies control.
  Eigen::VectorXd m_wCounts; //counts of obs by treatment condition w.  vector of length numTreatments
  Eigen::VectorXd m_yStats; //means of outcome by treatment vector of length numTreatments
  Eigen::VectorXd m_yVarStats; //variance of outcome by treatment vector of length numTreatments
  
  double m_yMean; //mean of outcome for the node
  double m_yVar; //variance in y for the node
  double m_err; //total error for the node
  Eigen::VectorXd m_tauHat; //treatment effects for the node if causal.  length numTreatments
  Eigen::VectorXd m_tauVarHat; //treatment effect variance estimate if causal. length numTreatments

  bool shouldISplit() const;
  bool shouldISplitRegression() const;
  bool shouldISplitClassification() const;

};


class OnlineTree {
public:
  //Classification Tree Constructors
  //version to create with randomization
  OnlineTree(const Hyperparameters& hp, const int& numClasses, const int& numFeatures, 
	       const Eigen::VectorXd& minFeatRange, const Eigen::VectorXd& maxFeatRange);

  //version to create from a matrix of parameters
  OnlineTree(const Eigen::MatrixXd& treeParms, const Hyperparameters& hp,
	     const int& numClasses, double oobe, double counter,
	     const Eigen::VectorXd& minFeatRange, const Eigen::VectorXd& maxFeatRange);

  //Regression Tree Constructors
  //version to create with randomization
  OnlineTree(const Hyperparameters& hp, const int& numFeatures, 
	       const Eigen::VectorXd& minFeatRange, const Eigen::VectorXd& maxFeatRange);

  //version to create from a matrix of parameters
  OnlineTree(const Eigen::MatrixXd& treeParms, const Hyperparameters& hp,
	     double oobe, double counter,
	     const Eigen::VectorXd& minFeatRange, const Eigen::VectorXd& maxFeatRange);

  ~OnlineTree();

  //update the tree with a new data point
  void update(Sample& sample);
  void updateClassification(Sample& sample);
  void updateRegression(Sample& sample);

  //evaluate a new data point
  void eval(Sample& sample, Result& result);

  //export tree parameters
  vector<Eigen::MatrixXd> exportParms();  //using a vector as needs to be constant across the models class  

  //get info about the tree
  double getOOBE();
  double getCounter();

//   //print information about and of the tree
//   void printInfo();
//   void print();

  pair<Eigen::VectorXd,Eigen::VectorXd> getFeatRange();
  void updateFeatRange(Eigen::VectorXd minFeatRange, Eigen::VectorXd maxFeatRange);

  Eigen::MatrixXd getFeatureImportance();

  const string name() const {
    return m_name;
  }


private:
  int m_numNodes;
  double m_oobe;
  double m_counter;

  const int* m_numClasses; 
  const Hyperparameters* m_hp;
  OnlineNode* m_rootNode;

  const Eigen::VectorXd* m_minFeatRange;
  const Eigen::VectorXd* m_maxFeatRange;

  string m_name;

};


class OnlineRF {
public:
  // Classification Forest Creation
  //version to construct using randomization
  OnlineRF(const Hyperparameters& hp, const int& numClasses, const int& numFeatures, 
	   Eigen::VectorXd minFeatRange, Eigen::VectorXd maxFeatRange);

  //version to construct from a set of parameters
  OnlineRF(const vector<Eigen::MatrixXd> orfParms, const Hyperparameters& hp,
	   const int& numClasses, double oobe, double counter,
	   Eigen::VectorXd minFeatRange, Eigen::VectorXd maxFeatRange);

  //Regression Forest Constructors
  //version to construct using randomization
  OnlineRF(const Hyperparameters& hp, const int& numFeatures, 
	   Eigen::VectorXd minFeatRange, Eigen::VectorXd maxFeatRange);

  //version to construct from a set of parameters
  OnlineRF(const vector<Eigen::MatrixXd> orfParms, const Hyperparameters& hp,
	   double oobe, double counter,
	   Eigen::VectorXd minFeatRange, Eigen::VectorXd maxFeatRange);



  ~OnlineRF();
  
  //update with a new data point
  void update(Sample& sample);
  void updateClassification(Sample& sample);
  void updateRegression(Sample& sample);
  
  //evaluate a new data point
  void eval(Sample& sample, Result& result);
  void evalClassification(Sample& sample, Result& result);
  void evalRegression(Sample& sample, Result& result);
 
  //export forest parameters
  vector<Eigen::MatrixXd> exportParms(); 

  //get info about the tree
  double getOOBE();
  double getCounter();

//   //print information about and of the RF
//   void printInfo();
//   void print();

  pair<Eigen::VectorXd,Eigen::VectorXd> getFeatRange();
  void updateFeatRange(Eigen::VectorXd minFeatRange, Eigen::VectorXd maxFeatRange);

  Eigen::MatrixXd getFeatureImportance();

  const string name() const {
    return m_name;
  }

  //methods for training and testing with large amounts of data
  void train(DataSet& dataset);
  vector<Result> test(DataSet& dataset);

protected:
  double m_counter;
  double m_oobe;
    
  vector<OnlineTree*> m_trees;

  const int* m_numClasses;
  const Hyperparameters* m_hp;

  //store vectors of min and max feature ranges - to be checked and updated when more data loaded
  Eigen::VectorXd m_minFeatRange;
  Eigen::VectorXd m_maxFeatRange;

  string m_name;
};

#endif /* ONLINERF_H_ */
