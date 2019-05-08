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
 *  added functionality and enabled ability to connect to R

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
  
  void update(const Sample& sample);
  
  bool eval(const Sample& sample) const;
  
  double score() const;
  
  pair<int,double> getParms();
  
  pair<Eigen::VectorXd, Eigen::VectorXd > getStats(std::string type = "all") const;

  void print();

    
 protected:
  const Hyperparameters* m_hp;
  const int* m_numClasses;
  const Eigen::VectorXd* m_rootLabelStats;
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
  
  void updateStats(const Sample& sample, const bool& decision);
};

class OnlineNode {
public:
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

  ~OnlineNode();
    
  //update with data
  //  void update(const Sample& sample);
  void update(const Sample& sample);
  //evaluate based on a new data point
  void eval(const Sample& sample, Result& result);

  //version to grow the node recursively from a matrix of information

  void update(const Eigen::MatrixXd& treeParms);

  //set child node numbers if the split occurs
  void setChildNodeNumbers(int rightChildNodeNumber, int leftChildNodeNumber);

  //method to add nodeParms to the matrix of parms for the tree
  Eigen::VectorXd exportParms(); //export parms out to a vector
  
  //recursive function to add elements to the vector for each child node
  void exportChildParms(vector<Eigen::VectorXd> &treeParmsVector);

  //function to score node with labelstats
  double score();
  double getCount(); 

  void printInfo();
  void print();


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
  Eigen::VectorXd m_ite; //individual treatment effect - populated if causal tree, otherwise 0s
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
  const double* m_rootCounter;
    
  bool shouldISplit() const;

};


class OnlineTree {
public:
  //version to create with randomization
  OnlineTree(const Hyperparameters& hp, const int& numClasses, const int& numFeatures, 
	       const Eigen::VectorXd& minFeatRange, const Eigen::VectorXd& maxFeatRange);

  //version to create from a matrix of parameters
  OnlineTree(const Eigen::MatrixXd& treeParms, const Hyperparameters& hp,
	     const int& numClasses, double oobe, double counter,
	     const Eigen::VectorXd& minFeatRange, const Eigen::VectorXd& maxFeatRange);

  ~OnlineTree();

  //update the tree with a new data point
  void update(Sample& sample);

  //evaluate a new data point
  void eval(Sample& sample, Result& result);

  //export tree parameters
  vector<Eigen::MatrixXd> exportParms();  //using a vector as needs to be constant across the models class  

  //get info about the tree
  double getOOBE();
  double getCounter();

  //print information about and of the tree
  void printInfo();
  void print();

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
  //version to construct using randomization
  OnlineRF(const Hyperparameters& hp, const int& numClasses, const int& numFeatures, 
	   Eigen::VectorXd minFeatRange, Eigen::VectorXd maxFeatRange);

  //version to construct from a set of parameters
  OnlineRF(const vector<Eigen::MatrixXd> orfParms, const Hyperparameters& hp,
	   const int& numClasses, double oobe, double counter,
	   Eigen::VectorXd minFeatRange, Eigen::VectorXd maxFeatRange);

  ~OnlineRF();
  
  //udpate with a new data point
  void update(Sample& sample);
  
  //evaluate a new data point
  void eval(Sample& sample, Result& result);
 
  //export forest parameters
  vector<Eigen::MatrixXd> exportParms(); 

  //get info about the tree
  double getOOBE();
  double getCounter();

  //print information about and of the RF
  void printInfo();
  void print();

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
