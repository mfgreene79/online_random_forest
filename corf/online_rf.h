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
 */

#ifndef ONLINERF_H_
#define ONLINERF_H_

#include "classifier.h"
#include "data.h"
#include "hyperparameters.h"
#include "utilities.h"

class RandomTest {
public:
  //Version to initialize with randomization
  RandomTest(const Hyperparameters& hp, const int& numClasses, const int& numFeatures, 
	     const VectorXd &minFeatRange, const VectorXd &maxFeatRange);

  //Version to initialize from a known feature/threshold - not causal
  RandomTest(const Hyperparameters& hp, const int& numClasses, 
	     int feature, double threshold,
	     VectorXd trueStats, VectorXd falseStats);

  //Version to initialize from a known feature/threshold - causal 
  RandomTest(const Hyperparameters& hp, const int& numClasses, 
	     int feature, double threshold,
	     VectorXd treatTrueStats, VectorXd treatFalseStats,
	     VectorXd controlTrueStats, VectorXd controlFalseStats
	     );
  
  void update(const Sample& sample);
  
  bool eval(const Sample& sample) const;
  
  double score() const;
  
  pair<int,double> getParms();
  
  pair<VectorXd, VectorXd > getStats(string type = "all") const;

  void print();
  
    
 protected:
  const Hyperparameters* m_hp;
  const int* m_numClasses;
  int m_feature;
  double m_threshold;

  //total counts and stats
  int m_trueCount;
  int m_falseCount;
  VectorXd m_trueStats;
  VectorXd m_falseStats;
    
  //treatment counts and stats
  int m_treatTrueCount;
  int m_treatFalseCount;
  VectorXd m_treatTrueStats;
  VectorXd m_treatFalseStats;

  //control counts and stats
  int m_controlTrueCount;
  int m_controlFalseCount;
  VectorXd m_controlTrueStats;
  VectorXd m_controlFalseStats;

  
  void updateStats(const Sample& sample, const bool& decision);
};

class OnlineNode {
public:
  // version to initialize the root node
  OnlineNode(const Hyperparameters& hp, const int& numClasses, const int& numFeatures, 
	     const VectorXd& minFeatRange, const VectorXd& maxFeatRange, 
	     const int& depth, int& numNodes);
  //version to initialize versions below the root node - not causal
  OnlineNode(const Hyperparameters& hp, const int& numClasses, const int& numFeatures, 
	     const VectorXd& minFeatRange, const VectorXd& maxFeatRange, 
	     const int& depth, const VectorXd& parentStats, 
	     int nodeNumber, int parentNodeNumber, int& numNodes);

  //version to initialize versions below the root node - causal
  OnlineNode(const Hyperparameters& hp, const int& numClasses, const int& numFeatures, 
	     const VectorXd& minFeatRange, const VectorXd& maxFeatRange, 
	     const int& depth, const VectorXd& treatParentStats, 
	     const VectorXd& controlParentStats, 
	     int nodeNumber, int parentNodeNumber, int& numNodes);
  
  //Version to initialize from a vector of information about the node
  OnlineNode(const VectorXd& nodeParms, const Hyperparameters& hp,
	     const int& numClasses, int& numNodes,
	     const VectorXd& minFeatRange, const VectorXd& maxFeatRange);

  ~OnlineNode();
    
  //update with data
  //  void update(const Sample& sample);
  void update(const Sample& sample);
  //evaluate based on a new data point
  void eval(const Sample& sample, Result& result);

  //version to grow the node recursively from a matrix of information

  void update(const MatrixXd& treeParms);

  //set child node numbers if the split occurs
  void setChildNodeNumbers(int rightChildNodeNumber, int leftChildNodeNumber);

  //method to add nodeParms to the matrix of parms for the tree
  VectorXd exportParms(); //export parms out to a vector
  
  //recursive function to add elements to the vector for each child node
  void exportChildParms(vector<VectorXd> &treeParmsVector);

  void printInfo();
  void print();


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
  VectorXd m_ite; //individual treatment effect - populated if causal tree, otherwise 0s
  double m_counter;
  double m_treatCounter;
  double m_controlCounter;
  double m_parentCounter;
  VectorXd m_labelStats;
  VectorXd m_treatLabelStats;
  VectorXd m_controlLabelStats;
  const VectorXd* m_minFeatRange;
  const VectorXd* m_maxFeatRange;
  
  OnlineNode* m_leftChildNode;
  OnlineNode* m_rightChildNode;
  
  vector<RandomTest*> m_onlineTests;
  RandomTest* m_bestTest;

  int* m_numNodes; //pointer to tree for number of nodes
    
  bool shouldISplit() const;
};


class OnlineTree: public Classifier {
public:
  //version to create with randomization
  OnlineTree(const Hyperparameters& hp, const int& numClasses, const int& numFeatures, 
	       const VectorXd& minFeatRange, const VectorXd& maxFeatRange);

  //version to create from a matrix of parameters
  OnlineTree(const MatrixXd& treeParms, const Hyperparameters& hp,
	     const int& numClasses, double oobe, double counter,
	     const VectorXd& minFeatRange, const VectorXd& maxFeatRange);

  ~OnlineTree();

  //update the tree with a new data point
  virtual void update(Sample& sample);

  //evaluate a new data point
  virtual void eval(Sample& sample, Result& result);

  //export tree parameters
  virtual vector<MatrixXd> exportParms();  //using a vector as needs to be constant across the classifier class  

  //get info about the tree
  virtual double getOOBE();
  virtual double getCounter();

  //print information about and of the tree
  virtual void printInfo();
  virtual void print();

  virtual pair<VectorXd,VectorXd> getFeatRange();
  virtual void updateFeatRange(VectorXd minFeatRange, VectorXd maxFeatRange);

private:
  int m_numNodes;
  double m_oobe;
  double m_counter;

  const int* m_numClasses; 
  const Hyperparameters* m_hp;
  OnlineNode* m_rootNode;

  const VectorXd* m_minFeatRange;
  const VectorXd* m_maxFeatRange;

};


class OnlineRF: public Classifier {
public:
  //version to construct using randomization
  OnlineRF(const Hyperparameters& hp, const int& numClasses, const int& numFeatures, 
	   VectorXd minFeatRange, VectorXd maxFeatRange);

  //version to construct from a set of parameters
  OnlineRF(const vector<MatrixXd> orfParms, const Hyperparameters& hp,
	   const int& numClasses, double oobe, double counter,
	   VectorXd minFeatRange, VectorXd maxFeatRange);

  ~OnlineRF();
  
  //udpate with a new data point
  virtual void update(Sample& sample);
  
  //evaluate a new data point
  virtual void eval(Sample& sample, Result& result);
 
  //export forest parameters
  virtual vector<MatrixXd> exportParms(); 

  //get info about the tree
  virtual double getOOBE();
  virtual double getCounter();

  //print information about and of the RF
  virtual void printInfo();
  virtual void print();

  virtual pair<VectorXd,VectorXd> getFeatRange();
  virtual void updateFeatRange(VectorXd minFeatRange, VectorXd maxFeatRange);

protected:
  double m_counter;
  double m_oobe;
    
  vector<OnlineTree*> m_trees;

  const int* m_numClasses;
  const Hyperparameters* m_hp;

  //store vectors of min and max feature ranges - to be checked and updated when more data loaded
  VectorXd m_minFeatRange;
  VectorXd m_maxFeatRange;
};

#endif /* ONLINERF_H_ */
