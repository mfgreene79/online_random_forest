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
 * 
 *
 * Updated 2019 Michael Greene mfgreene79@yahoo.com
 *  added functionality to incorporate into R package
 *  changed to Causal Online Random Forest 
 */

#include "online_rf.h"


/****************************************************************************************
 *
 *  RANDOM TEST CONSTRUCTORS AND METHODS 
 *
 ******************************************************************************************/

//version to construct with randomization
RandomTest::RandomTest(const Hyperparameters& hp,
		       const int& numClasses, const int& numFeatures, 
		       const VectorXd &minFeatRange, const VectorXd &maxFeatRange) :
  m_numClasses(&numClasses), m_hp(&hp),
  m_trueCount(0.0), m_falseCount(0.0),
  m_trueStats(VectorXd::Zero(numClasses)), m_falseStats(VectorXd::Zero(numClasses)),
  m_treatTrueCount(0.0), m_treatFalseCount(0.0),
  m_treatTrueStats(VectorXd::Zero(numClasses)), m_treatFalseStats(VectorXd::Zero(numClasses)),
  m_controlTrueCount(0.0), m_controlFalseCount(0.0),
  m_controlTrueStats(VectorXd::Zero(numClasses)), m_controlFalseStats(VectorXd::Zero(numClasses))
 {
  m_feature = floor(randDouble(0, numFeatures));
  m_threshold = randDouble(minFeatRange(m_feature), maxFeatRange(m_feature));
}

//version to construct from parameters - not causal
RandomTest::RandomTest(const Hyperparameters& hp, const int& numClasses, 
		       int feature, double threshold,
		       VectorXd trueStats, VectorXd falseStats) : 
  m_feature(feature), m_threshold(threshold), m_numClasses(&numClasses), m_hp(&hp),
  m_trueCount(0.0), m_falseCount(0.0),
  m_trueStats(trueStats), m_falseStats(falseStats),
  m_treatTrueCount(0.0), m_treatFalseCount(0.0),
  m_treatTrueStats(VectorXd::Zero(numClasses)), m_treatFalseStats(VectorXd::Zero(numClasses)),
  m_controlTrueCount(0.0), m_controlFalseCount(0.0),
  m_controlTrueStats(VectorXd::Zero(numClasses)), m_controlFalseStats(VectorXd::Zero(numClasses))
{
  m_trueCount = m_trueStats.sum();
  m_falseCount = m_falseStats.sum();
}

//version to construct from parameters - causal
RandomTest::RandomTest(const Hyperparameters& hp, const int& numClasses, 
		       int feature, double threshold,
		       VectorXd treatTrueStats, VectorXd treatFalseStats,
		       VectorXd controlTrueStats, VectorXd controlFalseStats) : 
  m_feature(feature), m_threshold(threshold), m_numClasses(&numClasses), m_hp(&hp),
  m_treatTrueStats(treatTrueStats), m_treatFalseStats(treatFalseStats),
  m_controlTrueStats(controlTrueStats), m_controlFalseStats(controlFalseStats),
  m_trueStats(VectorXd::Zero(numClasses)), m_falseStats(VectorXd::Zero(numClasses)),
  m_trueCount(0), m_falseCount(0)
{

  for(int i=0; i < numClasses; ++i) {
    //total stats
    m_trueStats(i) += treatTrueStats(i);
    m_falseStats(i) += treatFalseStats(i);
    m_trueStats(i) += controlTrueStats(i);
    m_falseStats(i) += treatFalseStats(i);

    //total counts
    m_trueCount += m_trueStats(i);
    m_falseCount += m_falseStats(i);
    m_treatTrueCount += treatTrueStats(i);
    m_treatFalseCount += treatFalseStats(i);
    m_controlTrueCount += controlTrueStats(i);
    m_controlFalseCount += controlFalseStats(i);
  }
}

void RandomTest::update(const Sample& sample) {
    updateStats(sample, eval(sample));
}
    
bool RandomTest::eval(const Sample& sample) const {
    return (sample.x(m_feature) > m_threshold) ? true : false;
}

double splitScore(VectorXd trueStats, VectorXd falseStats, int trueCount, int falseCount, 
	     string method="gini") { 
  double out;
  int numClasses = trueStats.size();
  if(method == "gini") {
    double trueScore = 0.0, falseScore = 0.0, p;
    if (trueCount) {
      for (int nClass = 0; nClass < numClasses; ++nClass) {
	p = trueStats[nClass] / trueCount;
	trueScore += p * (1 - p);
      }
    }      
    if (falseCount) {
      for (int nClass = 0; nClass < numClasses; ++nClass) {
	p = falseStats[nClass] / falseCount;
	falseScore += p * (1 - p);
      }
    }      
    out = (trueCount * trueScore + falseCount * falseScore) / (trueCount + falseCount + 1e-16);
  } else if(method == "entropy") {
    double trueScore = 0.0, falseScore = 0.0, p;
    if (trueCount) {
      for (int nClass = 0; nClass < numClasses; ++nClass) {
	p = trueStats[nClass] / trueCount;
	trueScore += p * log2(p);
      }
    }      
    if (falseCount) {
      for (int nClass = 0; nClass < numClasses; ++nClass) {
	p = falseStats[nClass] / falseCount;
	falseScore += p * log2(p);
      }
    }      
    out = (trueCount * trueScore + falseCount * falseScore) / (trueCount + falseCount + 1e-16);
  }
  return(out);
}

    
double RandomTest::score() const {
  double theta; //value to minimize

  if(m_hp->causal == true) {
    double treat_score, control_score;

    //score the treatment and control counts
    treat_score = splitScore(m_treatTrueStats, m_treatFalseStats, 
			m_treatTrueCount, m_treatFalseCount, 
			m_hp->method);
    control_score = splitScore(m_controlTrueStats, m_controlFalseStats, 
			m_controlTrueCount, m_controlFalseCount, 
			m_hp->method);

    //minimizing the difference between -abs(treatment and control)
    //equiv to maximizing the diff between treatment and control
    theta = - abs(treat_score - control_score);
  } else { //not looking at causal tree
    //minimizing the score directly
    theta = splitScore(m_trueStats, m_falseStats, 
		  m_trueCount, m_falseCount, 
		  m_hp->method);
  }
  return(theta);
}
    
pair<VectorXd, VectorXd > RandomTest::getStats(string type) const {
  
  //  VectorXd trueStats, falseStats;
  pair<VectorXd, VectorXd> outStats;

  if(type == "all") {
    outStats = pair<VectorXd, VectorXd> (m_trueStats, m_falseStats);
  } else if(type == "treat") {
    outStats = pair<VectorXd, VectorXd> (m_treatTrueStats, m_treatFalseStats);
  } else if(type == "control") {
   outStats = pair<VectorXd, VectorXd> (m_controlTrueStats, m_controlFalseStats);
  }
  
  return outStats;
}

void RandomTest::updateStats(const Sample& sample, const bool& decision) {
  if (decision) {
    m_trueCount += sample.w;
    m_trueStats(sample.y) += sample.w;
    if(sample.treat == true) {
      m_treatTrueCount += sample.w;
      m_treatTrueStats(sample.y) += sample.w;
    } else {
      m_controlTrueCount += sample.w;
      m_controlTrueStats(sample.y) += sample.w;
    }
  } else {
    m_falseCount += sample.w;
    m_falseStats(sample.y) += sample.w;
    if(sample.treat == true) {
      m_treatFalseCount += sample.w;
      m_treatFalseStats(sample.y) += sample.w;
    } else {
      m_controlFalseCount += sample.w;
      m_controlFalseStats(sample.y) += sample.w;
    }
  }
}

pair<int, double> RandomTest::getParms() {
  //fetch the parms for the RandomTest as a vector
  return pair<int, double> (m_feature, m_threshold);
}

void RandomTest::print() {
  cout << "m_feature: " << m_feature << ", threshold: " << m_threshold << std::endl;
}

/****************************************************************************************
 *
 *  ONLINE NODE CONSTRUCTORS AND METHODS 
 *
 ******************************************************************************************/

//version for the root node
OnlineNode::OnlineNode(const Hyperparameters& hp, const int& numClasses, 
		       const int& numFeatures, const VectorXd& minFeatRange, 
		       const VectorXd& maxFeatRange, 
                       const int& depth, int& numNodes) :
  m_numClasses(&numClasses), m_depth(depth), m_isLeaf(true), m_hp(&hp), m_label(-1),
  m_counter(0.0), m_parentCounter(0.0), m_labelStats(VectorXd::Zero(numClasses)),
  m_treatLabelStats(VectorXd::Zero(numClasses)), 
  m_controlLabelStats(VectorXd::Zero(numClasses)),
  m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange), m_nodeNumber(0),
  m_numNodes(&numNodes), m_treatCounter(0.0), m_controlCounter(0.0),
  m_ite(VectorXd::Zero(numClasses))
{
  // Creating random tests
  for (int nTest = 0; nTest < hp.numRandomTests; ++nTest) {
    m_onlineTests.push_back(new RandomTest(hp, numClasses, numFeatures, 
					   minFeatRange, maxFeatRange));
  }  
  setChildNodeNumbers(-1, -1);
  ++numNodes;
}

    
//version for those below the root node - not causal
OnlineNode::OnlineNode(const Hyperparameters& hp, const int& numClasses, 
		       const int& numFeatures, const VectorXd& minFeatRange, 
		       const VectorXd& maxFeatRange, 
                       const int& depth, const VectorXd& parentStats, 
		       int nodeNumber, int parentNodeNumber, int& numNodes) :
  m_numClasses(&numClasses), m_depth(depth), m_isLeaf(true), m_hp(&hp), m_label(-1),
  m_counter(0.0), m_parentCounter(parentStats.sum()), m_labelStats(parentStats),
  m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange), m_nodeNumber(nodeNumber),
  m_parentNodeNumber(parentNodeNumber), m_numNodes(&numNodes), 
  m_treatCounter(0.0), m_controlCounter(0.0),
  m_ite(VectorXd::Zero(numClasses)) {
  //calculate the label
  m_labelStats.maxCoeff(&m_label);
  
  // Creating random tests
  for (int nTest = 0; nTest < hp.numRandomTests; ++nTest) {
    m_onlineTests.push_back(new RandomTest(hp, numClasses, numFeatures, minFeatRange, maxFeatRange));
  }
  setChildNodeNumbers(-1, -1);
  ++numNodes;
}

//version below the root node - causal tree
OnlineNode::OnlineNode(const Hyperparameters& hp, const int& numClasses, 
		       const int& numFeatures, const VectorXd& minFeatRange, 
		       const VectorXd& maxFeatRange, 
                       const int& depth, 
		       const VectorXd& treatParentStats, 
		       const VectorXd& controlParentStats, 
		       int nodeNumber, int parentNodeNumber, int& numNodes) :
  m_numClasses(&numClasses), m_depth(depth), m_isLeaf(true), m_hp(&hp), m_label(-1),
  m_counter(0.0), m_parentCounter(0.0), m_treatCounter(treatParentStats.sum()), 
  m_controlCounter(controlParentStats.sum()),
  m_labelStats(VectorXd::Zero(numClasses)),
  m_treatLabelStats(treatParentStats), m_controlLabelStats(controlParentStats),
  m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange), m_nodeNumber(nodeNumber),
  m_parentNodeNumber(parentNodeNumber), m_numNodes(&numNodes), 
  m_ite(VectorXd::Zero(numClasses)) {


  m_treatCounter = treatParentStats.sum();
  m_controlCounter = controlParentStats.sum();
  m_parentCounter = m_treatCounter + m_controlCounter;
  
  for(int nClass=0; nClass < *m_numClasses; ++nClass) {
    m_labelStats(nClass) = treatParentStats(nClass) + controlParentStats(nClass);
  }

  //calculate the label
  m_labelStats.maxCoeff(&m_label);

  //update the ite
  //set ite to be difference from control
  for(int i=0; i < *m_numClasses; ++i) {
    if(m_treatCounter > 0 & m_controlCounter > 0) { 
      m_ite(i) = (m_treatLabelStats(i)/m_treatCounter) - (m_controlLabelStats(i)/m_controlCounter);
    } else {
      m_ite(i) = 0;
    }
  }
  
  // Creating random tests
  for (int nTest = 0; nTest < hp.numRandomTests; ++nTest) {
    m_onlineTests.push_back(new RandomTest(hp, numClasses, numFeatures, minFeatRange, maxFeatRange));
  }
  setChildNodeNumbers(-1, -1);
  ++numNodes;
}

//version to create from parameters
OnlineNode::OnlineNode(const VectorXd& nodeParms, const Hyperparameters& hp,
		       const int& numClasses, int& numNodes,
		       const VectorXd& minFeatRange, const VectorXd& maxFeatRange) : 
  m_hp(&hp), m_numNodes(&numNodes), m_numClasses(&numClasses),
  m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange),
  m_labelStats(VectorXd::Zero(numClasses)),
  m_treatLabelStats(VectorXd::Zero(numClasses)),
  m_controlLabelStats(VectorXd::Zero(numClasses)),
  m_ite(VectorXd::Zero(numClasses))
{

  //extract information about the node from the vector
  //common information whether causal or not
  m_nodeNumber = static_cast<int>(nodeParms(0));
  m_parentNodeNumber = static_cast<int>(nodeParms(1));
  m_rightChildNodeNumber = static_cast<int>(nodeParms(2));
  m_leftChildNodeNumber = static_cast<int>(nodeParms(3));
  m_depth = static_cast<int>(nodeParms(4));
  m_isLeaf = static_cast<bool>(nodeParms(5));
  m_label = static_cast<int>(nodeParms(6));
  m_counter = nodeParms(7);
  //insert treatCounter and controlCounter  
  
  //if causal need to extract treatment and control separately
  if(m_hp->causal == true) {
     int pos=8;
     //ite
     for(int l=0; l < numClasses; ++l) {
       m_ite(l) = static_cast<double>(nodeParms(pos+l));
     }
    pos=8+numClasses;

    m_treatCounter = nodeParms(pos);
    m_controlCounter = nodeParms(pos+1);
    m_parentCounter = nodeParms(pos+2);
      
    //copy in information for labelStats
    pos = 13+numClasses; //skip two positions for stats collected otherwise

    for(int c=0; c < *m_numClasses; ++c) {
      m_labelStats(c) = nodeParms(pos + c);
      m_treatLabelStats(c) = nodeParms(pos + c + *m_numClasses);
      m_controlLabelStats(c) = nodeParms(pos + c + 2 * *m_numClasses);
    }
    
    pos = 13 + 4 * *m_numClasses;

    //set up random tests and best random test
    //advance by 2 - where the bestTest would be stored
    VectorXd treatTrueStats = VectorXd::Zero(numClasses);
    VectorXd treatFalseStats = VectorXd::Zero(numClasses);
    VectorXd controlTrueStats = VectorXd::Zero(numClasses);
    VectorXd controlFalseStats = VectorXd::Zero(numClasses);
    
    if(m_isLeaf == false) { //when not a leaf create the best test
      RandomTest* bt;
      int feature = nodeParms(pos);
      double threshold = nodeParms(pos + 1);

      for(int i=0; i<numClasses; ++i) {
	treatTrueStats(i) = nodeParms(pos + 2 + i);
	treatFalseStats(i) = nodeParms(pos + 2 + i + numClasses);
	controlTrueStats(i) = nodeParms(pos + 2 + i + 2*numClasses);
	controlFalseStats(i) = nodeParms(pos + 2 + i + 3*numClasses);
      }
      
      bt = new RandomTest(hp, numClasses, feature, threshold, 
			  treatTrueStats, treatFalseStats,
			  controlTrueStats, controlFalseStats);
      m_bestTest = bt;
    } //close isLeaf

    //for all nodes (leaf or not) create the randomtests
    pos = 15 + 8 * numClasses;
    RandomTest* rt;
    for(int i=0; i < m_hp->numRandomTests; ++i) {
      int feature = nodeParms(pos + i * (2 * (1 + 2 * numClasses)));
      double threshold = nodeParms(pos + 1 + i * (2 * (1 + 2 * numClasses)));
      treatTrueStats = VectorXd::Zero(numClasses);
      treatFalseStats = VectorXd::Zero(numClasses);
      controlTrueStats = VectorXd::Zero(numClasses);
      controlFalseStats = VectorXd::Zero(numClasses);
      for(int j=0; j<numClasses; ++j) {
	treatTrueStats(j) = nodeParms(pos + 2 + j + i * (2 * (1 + 2 * numClasses)));
	treatFalseStats(j) = nodeParms(pos + 2 + j + numClasses + i * (2 * (1 + 2 * numClasses)));
	controlTrueStats(j) = nodeParms(pos + 2 + j + 2 * numClasses + i * (2 * (1 + 2 * numClasses)));
	controlFalseStats(j) = nodeParms(pos + 2 + j + 3 * numClasses + i * (2 * (1 + 2 * numClasses)));
      }
      rt = new RandomTest(hp, numClasses, feature, threshold, 
			  treatTrueStats, treatFalseStats,
			  controlTrueStats, controlFalseStats);
      m_onlineTests.push_back(rt);
    } //close i loop
    
  } else { //causal==false

    m_parentCounter = nodeParms(8);
  
    //copy in information for labelStats
    VectorXd labelStats(*m_numClasses);


    int pos = 11; //starts at 11 - numClasses and numRandomTests already captured (9 and 10)
    for(int c=0; c < *m_numClasses; ++c) {
      labelStats(c) = nodeParms(pos + c);
    }
    m_labelStats = labelStats;
    
    pos = 11 + *m_numClasses;

    //set up random tests and best random test
    //advance by 2 - where the bestTest would be stored
    VectorXd trueStats = VectorXd::Zero(numClasses);
    VectorXd falseStats = VectorXd::Zero(numClasses);
    
    if(m_isLeaf == false) { //when not a leaf create the best test
      RandomTest* bt;
      int feature = nodeParms(pos);
      double threshold = nodeParms(pos + 1);
      VectorXd trueStats(numClasses);
      VectorXd falseStats(numClasses);
      for(int i=0; i<numClasses; ++i) {
	trueStats(i) = nodeParms(pos + 2 + i);
	falseStats(i) = nodeParms(pos + 2 + i + numClasses);
      }
      
      bt = new RandomTest(hp, numClasses, feature, threshold, trueStats, falseStats);
      m_bestTest = bt;
    } //close isLeaf

    //for all nodes (leaf or not) create the randomtests
    pos = 13 + 3 * numClasses;
    RandomTest* rt;
    for(int i=0; i < m_hp->numRandomTests; ++i) {
      int feature = nodeParms(pos + i * (2 * (1 + numClasses)));
      double threshold = nodeParms(pos + 1 + i * (2 * (1 + numClasses)));
      trueStats = VectorXd::Zero(numClasses);
      falseStats = VectorXd::Zero(numClasses);
      for(int j=0; j<numClasses; ++j) {
	trueStats(j) = nodeParms(pos + 2 + j + i * (2 * (1 + numClasses)));
	falseStats(j) = nodeParms(pos + 2 + j + numClasses + i * (2 * (1 + numClasses)));
      }
      rt = new RandomTest(hp, numClasses, feature, threshold, trueStats, falseStats);
      m_onlineTests.push_back(rt);
    } //close i loop
  } //close causal==false
} //close method
    
OnlineNode::~OnlineNode() {
  if (m_isLeaf == false) {
    delete m_leftChildNode;
    delete m_rightChildNode;
    delete m_bestTest;
  } else {
    for (int nTest = 0; nTest < m_hp->numRandomTests; ++nTest) {
      delete m_onlineTests[nTest];
    }
  }
  --m_numNodes;
}

//set the child node numbers if needed 
void OnlineNode::setChildNodeNumbers(int rightChildNodeNumber, int leftChildNodeNumber) {
  m_rightChildNodeNumber = rightChildNodeNumber;
  m_leftChildNodeNumber = leftChildNodeNumber;
}
    
//void OnlineNode::update(const Sample& sample) {
void OnlineNode::update(const Sample& sample) {
  m_counter += sample.w;
  m_labelStats(sample.y) += sample.w;

  //increment treatment and control stats if a causal tree
  if(m_hp->causal == true) {
    if(sample.treat == true) {
      m_treatCounter += sample.w;
      m_treatLabelStats(sample.y) += sample.w;
    } else {
      m_controlCounter += sample.w;
      m_controlLabelStats(sample.y) += sample.w;
    }
  }

   if (m_isLeaf == true) {
    // Update online tests
    for (vector<RandomTest*>::iterator itr = m_onlineTests.begin(); 
	 itr != m_onlineTests.end(); ++itr) {
      (*itr)->update(sample);
    }
    
    // Update the label
    m_labelStats.maxCoeff(&m_label);

    //update the ite    
    if(m_hp->causal == true) {
      //set ite to be difference from control
      for(int i=0; i < *m_numClasses; ++i) {
	if(m_treatCounter > 0 & m_controlCounter > 0) {
	  m_ite(i) = (m_treatLabelStats(i)/m_treatCounter) - (m_controlLabelStats(i)/m_controlCounter); 
	} else {
	  m_ite(i) = 0;
	}
      }
    }
    
    // Decide for split
    if (shouldISplit()) {
      m_isLeaf = false;
      
      // Find the best online test
      int nTest = 0, minIndex = 0;
      double minScore = 1, score;
      for (vector<RandomTest*>::const_iterator itr(m_onlineTests.begin()); 
	   itr != m_onlineTests.end(); ++itr, nTest++) {
	score = (*itr)->score();
	if (score < minScore) {
	  minScore = score;
	  minIndex = nTest;
	}
      }
      m_bestTest = m_onlineTests[minIndex];

      //commented out delete - keep track of what happened instead at expense of memory
//       for (int nTest = 0; nTest < m_hp->numRandomTests; ++nTest) {
//       	If (minIndex != nTest) {
//       	  delete m_onlineTests[nTest];
//       	}
//       }
      
      //Figure out the next available nodeNumber - then increment the one for the tree
      int newNodeNumber = *m_numNodes;

      // Split - initializing with versions beyond the root node
      if(m_hp->causal == false) {
	pair<VectorXd, VectorXd> parentStats = m_bestTest->getStats("all");      

	m_rightChildNode = new OnlineNode(*m_hp, *m_numClasses,
					  m_minFeatRange->rows(), *m_minFeatRange, 
					  *m_maxFeatRange, m_depth + 1, 
					  parentStats.first, newNodeNumber, 
					  m_nodeNumber, *m_numNodes);
	m_leftChildNode = new OnlineNode(*m_hp, *m_numClasses, m_minFeatRange->rows(),
					 *m_minFeatRange, *m_maxFeatRange, m_depth + 1,
					 parentStats.second, newNodeNumber + 1, 
					 m_nodeNumber, *m_numNodes);
      } else { // causal==true
	pair<VectorXd, VectorXd> treatParentStats = m_bestTest->getStats("treat");      
	pair<VectorXd, VectorXd> controlParentStats = m_bestTest->getStats("control");      

	m_rightChildNode = new OnlineNode(*m_hp, *m_numClasses,
					  m_minFeatRange->rows(), *m_minFeatRange, 
					  *m_maxFeatRange, m_depth + 1, 
					  treatParentStats.first, 
					  controlParentStats.first, 
					  newNodeNumber, 
					  m_nodeNumber, *m_numNodes);
	m_leftChildNode = new OnlineNode(*m_hp, *m_numClasses, m_minFeatRange->rows(),
					 *m_minFeatRange, *m_maxFeatRange, m_depth + 1,
					 treatParentStats.second,
					 controlParentStats.second,
					 newNodeNumber + 1, 
					 m_nodeNumber, *m_numNodes);

      }
      
      //set the child node numbers now that nodes have been created
      setChildNodeNumbers(newNodeNumber, newNodeNumber + 1);
    }
   } else {
    if (m_bestTest->eval(sample)) {
      m_rightChildNode->update(sample);
    } else {
      m_leftChildNode->update(sample);
    }
   }
}

void OnlineNode::eval(const Sample& sample, Result& result) {
  if (m_isLeaf == true) {
    if (m_counter + m_parentCounter) {
      result.confidence = m_labelStats / (m_counter + m_parentCounter);
      result.prediction = m_label;
      if(m_hp->causal == true) {
	result.ite = m_ite;
      } else {
	result.ite = VectorXd::Zero(*m_numClasses);
      }
    } else {
      result.confidence = VectorXd::Constant(m_labelStats.rows(), 1.0 / *m_numClasses);
      result.prediction = 0;
      result.ite = VectorXd::Zero(*m_numClasses);
    }
  } else {
    if (m_bestTest->eval(sample)) {
      m_rightChildNode->eval(sample, result);
    } else {
      m_leftChildNode->eval(sample, result);
    }
  }
}

//version of update to grow from a set of parameters
void OnlineNode::update(const MatrixXd& treeParms) {
  if(m_isLeaf == false) { // if its a leaf then theres no splitting to do
    
    //search through matrix of parms to find the correct rows and make node parms
    int found=0;
    for(int i=0; i < treeParms.rows(); ++i) {
      VectorXd nodeParmsVec = treeParms.row(i);
      int npv_nodeNumber = static_cast<int>(nodeParmsVec(0));
      if(npv_nodeNumber == m_rightChildNodeNumber) {
	m_rightChildNode = new OnlineNode(nodeParmsVec, *m_hp, *m_numClasses, *m_numNodes,
					  *m_minFeatRange, *m_maxFeatRange);
	found++;
      } else if(npv_nodeNumber == m_leftChildNodeNumber) {
	m_leftChildNode = new OnlineNode(nodeParmsVec, *m_hp, *m_numClasses, *m_numNodes,
					 *m_minFeatRange, *m_maxFeatRange);
	found++;
      }
      //once the two nodes have been located stop the looping
      if(found > 1) {
	break;
      }
    }
    //update the new nodes - making this recursive
    m_rightChildNode->update(treeParms);
    m_leftChildNode->update(treeParms);
  }
}

bool OnlineNode::shouldISplit() const {
  bool isPure = false;
  for (int nClass = 0; nClass < *m_numClasses; ++nClass) {
    if (m_labelStats(nClass) == m_counter + m_parentCounter) {
      isPure = true;
      break;
    }
  }
  if ((isPure) || (m_depth >= m_hp->maxDepth) || (m_counter < m_hp->counterThreshold)) {
    return false;
  } else {
    return true;
  }
}

VectorXd OnlineNode::exportParms() {
  //create vector to export
  
  //see layout spreadsheet for accounting of length
  int vec_size;
  if(m_hp->causal == true) {
    vec_size = 15 + 8 * *m_numClasses + 2 * m_hp->numRandomTests * (1 + 2 * *m_numClasses);
  } else {
    vec_size = 13 + 3 * *m_numClasses + 2 * m_hp->numRandomTests * (1 + *m_numClasses);
  }
  VectorXd nodeParms = VectorXd::Zero(vec_size);  //initialize the vector with zeros
  
  //fetch the private parameters and save into the Node parms object
  int pos = 0;
  nodeParms(0) = static_cast<double>(m_nodeNumber);
  nodeParms(1) = static_cast<double>(m_parentNodeNumber);
  nodeParms(2) = static_cast<double>(m_rightChildNodeNumber);
  nodeParms(3) = static_cast<double>(m_leftChildNodeNumber);
  nodeParms(4) = static_cast<double>(m_depth);
  nodeParms(5) = static_cast<double>(m_isLeaf);
  nodeParms(6) = static_cast<double>(m_label);
  nodeParms(7) = static_cast<double>(m_counter);
  
  if(m_hp->causal == true) {
    //put in the ite estimates
    pos=8;
    for(int l=0; l < *m_numClasses; ++l) {
      nodeParms(pos + l) = m_ite(l);
    }
    pos = 8 + *m_numClasses;

    //layout for causal tree has 4 vectors for each random test
    nodeParms(pos) = static_cast<double>(m_treatCounter);
    nodeParms(pos+1) = static_cast<double>(m_controlCounter);
    nodeParms(pos+2) = static_cast<double>(m_parentCounter);
    nodeParms(pos+3) = static_cast<double>(*m_numClasses);
    nodeParms(pos+4) = static_cast<double>(m_hp->numRandomTests);
    pos = 13 + *m_numClasses;
    

    //copy in the label stats
     for(int l=0; l < *m_numClasses;++l) {
       nodeParms(pos + l) = static_cast<double>(m_labelStats(l));
       nodeParms(pos + l + *m_numClasses) = static_cast<double>(m_treatLabelStats(l));
       nodeParms(pos + l + 2 * *m_numClasses) = static_cast<double>(m_controlLabelStats(l));
     } 
    pos = 13 + 4 * *m_numClasses;

    pair<int, double> bt_parms;
    pair<VectorXd, VectorXd> bt_treatStats;
    pair<VectorXd, VectorXd> bt_controlStats;
    if(m_isLeaf == false) { //if NOT a leaf then we dont have a best test but do have randomtests
      bt_parms = m_bestTest->getParms();
      bt_treatStats = m_bestTest->getStats("treat");
      bt_controlStats = m_bestTest->getStats("control");
    } else { //otherwise use zeros (and -1 for the feature)
      int bt1 = -1;
      double bt2 = 0;
      VectorXd bt3 = VectorXd::Zero(*m_numClasses);

      bt_parms = pair<int, double> (bt1, bt2);
      bt_treatStats=pair<VectorXd, VectorXd> (bt3, bt3);
      bt_controlStats=pair<VectorXd, VectorXd> (bt3, bt3);
    }
    //write bt information to the vector
    nodeParms(pos) = bt_parms.first;
    nodeParms(pos + 1) = bt_parms.second;
    
    //copy the information from trueStats and falseStats into the parms
    //m_numClass columns for m_trueStats and m_numClass cols for m_falseStats
    pos = 15 + 4 * *m_numClasses;
    VectorXd treatTrueStats = bt_treatStats.first;
    VectorXd treatFalseStats = bt_treatStats.second;
    VectorXd controlTrueStats = bt_controlStats.first;
    VectorXd controlFalseStats = bt_controlStats.second;
    
    for(int i=0; i < *m_numClasses; ++i) {
      nodeParms(pos+i) = treatTrueStats(i);
      nodeParms(pos + *m_numClasses + i) = treatFalseStats(i);
      nodeParms(pos + *m_numClasses * 2 + i) = controlTrueStats(i);
      nodeParms(pos + *m_numClasses * 3 + i) = controlFalseStats(i);
    }
    
    pos = 15 + 8 * *m_numClasses;
    
    //copy in the random test information
    for(int i=0; i <  m_hp->numRandomTests; ++i) {
      pos = 15 + 8 * *m_numClasses + i * (2 * (1 + 2 * *m_numClasses)); //for each random test
      
      RandomTest rt = *m_onlineTests[i];
      pair<int, double> rt_parms = rt.getParms();
      pair<VectorXd, VectorXd> rt_treatStats = rt.getStats("treat");
      pair<VectorXd, VectorXd> rt_controlStats = rt.getStats("control");
      //feature
      nodeParms(pos) = static_cast<double>(rt_parms.first);
      //threshold
      nodeParms(pos + 1) = static_cast<double>(rt_parms.second);
      //copy in the true and false stats
      VectorXd treatTrueStats = rt_treatStats.first;
      VectorXd treatFalseStats = rt_treatStats.second;
      VectorXd controlTrueStats = rt_controlStats.first;
      VectorXd controlFalseStats = rt_controlStats.second;
      for(int j=0; j < *m_numClasses; ++j) {
	nodeParms(pos + 2 + j) = treatTrueStats(j);
	nodeParms(pos + 2 + j + *m_numClasses) = treatFalseStats(j);
	nodeParms(pos + 2 + j + *m_numClasses * 2) = controlTrueStats(j);
	nodeParms(pos + 2 + j + *m_numClasses * 3) = controlFalseStats(j);
      } //loop j
    } //loop i
  } else { //causal == false
    nodeParms(8) = static_cast<double>(m_parentCounter);
    nodeParms(9) = static_cast<double>(*m_numClasses);
    nodeParms(10) = static_cast<double>(m_hp->numRandomTests);
    pos = 11;
  
    //copy in the label stats
    for(int l=0; l < *m_numClasses;++l) {
      nodeParms(pos + l) = static_cast<double>(m_labelStats(l));
    } 
    pos = 11 + *m_numClasses;

    //layout for causal tree has 4 vectors for each random test
    pair<int, double> bt_parms;
    pair<VectorXd, VectorXd> bt_stats;
    if(m_isLeaf == false) { //if NOT a leaf then we dont have a best test but do have randomtests
      bt_parms = m_bestTest->getParms();
      bt_stats = m_bestTest->getStats();
    } else { //otherwise use zeros (and -1 for the feature)
      int bt1 = -1;
      double bt2 = 0;
      VectorXd bt3 = VectorXd::Zero(*m_numClasses);;
      VectorXd bt4 = VectorXd::Zero(*m_numClasses);

      bt_parms = pair<int, double> (bt1, bt2);
      bt_stats=pair<VectorXd, VectorXd> (bt3, bt4);
    }
    //write bt information to the vector
    nodeParms(pos) = bt_parms.first;
    nodeParms(pos + 1) = bt_parms.second;
    
    //copy the information from trueStats and falseStats into the parms
    //m_numClass columns for m_trueStats and m_numClass cols for m_falseStats
    pos = 13 + *m_numClasses;
    VectorXd trueStats = bt_stats.first;
    VectorXd falseStats = bt_stats.second;
    
    for(int i=0; i < *m_numClasses; ++i) {
      nodeParms(pos+i) = trueStats(i);
      nodeParms(pos+*m_numClasses+i) = falseStats(i);
    }
    
    pos = 13 + 3 * *m_numClasses;
    
    //copy in the random test information
    for(int i=0; i <  m_hp->numRandomTests; ++i) {
      pos = 13 + 3 * *m_numClasses + i * (2 * (1 + *m_numClasses)); //for each random test
      
      RandomTest rt = *m_onlineTests[i];
      pair<int, double> rt_parms = rt.getParms();
      pair<VectorXd, VectorXd> rt_stats = rt.getStats();
      //feature
      nodeParms(pos) = static_cast<double>(rt_parms.first);
      //threshold
      nodeParms(pos + 1) = static_cast<double>(rt_parms.second);
      //copy in the true and false stats
      VectorXd trueStats = rt_stats.first;
      VectorXd falseStats = rt_stats.second;
      for(int j=0; j < *m_numClasses; ++j) {
	nodeParms(pos + 2 + j) = trueStats(j);
	nodeParms(pos + 2 + j + *m_numClasses) = falseStats(j);
      } //loop j
    } //loop i
  } //causal condition
  return(nodeParms);
}

//method to recursively return information - updating matrix at the tree level
void OnlineNode::exportChildParms(vector<VectorXd> &treeParmsVector) {
  //add the right and left child parms to parms for the tree
  if(m_isLeaf == false) {
    //collect and export the parms if this is not a leaf
    VectorXd rightParms = m_rightChildNode->exportParms();
    treeParmsVector.push_back(rightParms);

    VectorXd leftParms = m_leftChildNode->exportParms();
    treeParmsVector.push_back(leftParms);

    //recurse to the next level if NOT a leaf
    m_rightChildNode->exportChildParms(treeParmsVector);
    m_leftChildNode->exportChildParms(treeParmsVector);
  }
}



void OnlineNode::printInfo() {
  cout << "Node Information about Node " << m_nodeNumber << std::endl;
  cout << "\tisLeaf: " << m_isLeaf << ", rightChildNodeNumber: " << m_rightChildNodeNumber << ", leftChildNodeNumber: " << m_leftChildNodeNumber << std::endl;
}

void OnlineNode::print() {
  cout << "Node details: " << m_nodeNumber << std::endl;
  cout << exportParms() << std::endl;
}


/****************************************************************************************
 *
 *  ONLINE TREE CONSTRUCTORS AND METHODS 
 *
 ******************************************************************************************/

//version to construct with randomization
OnlineTree::OnlineTree(const Hyperparameters& hp, const int& numClasses, 
		       const int& numFeatures, 
                       const VectorXd& minFeatRange, const VectorXd& maxFeatRange) :
  Classifier(hp, numClasses), m_numClasses(&numClasses), m_hp(&hp),
  m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange) {
  
  //initialize here - will get updated later in the program during update
  m_oobe = 0.0;
  m_counter = 0.0;
  m_numNodes = 0;
  //initialize with root node version
  m_rootNode = new OnlineNode(hp, numClasses, numFeatures, minFeatRange, maxFeatRange, 
  			      0, m_numNodes);
    
  m_name = "OnlineTree";
}

//version to construct from a set of parameters
OnlineTree::OnlineTree(const MatrixXd& treeParms, const Hyperparameters& hp, 
		       const int& numClasses, double oobe, double counter,
		       const VectorXd& minFeatRange, const VectorXd& maxFeatRange) :
  Classifier(hp, treeParms(0,9)), m_hp(&hp),
  m_oobe(oobe), m_counter(counter), m_numClasses(&numClasses), m_numNodes(0),
  m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange) {

  //find the max node number from the treeParms matrix - position 0
  m_numNodes = treeParms.rows();
  
  //cout << "nodeParms size: " << treeParms.row(0).size() << "\n";

  //initialize with the version that takes parameters
  m_rootNode = new OnlineNode(treeParms.row(0), hp, numClasses, m_numNodes, 
  			      minFeatRange, maxFeatRange);

  //grow the tree based on matrix of parameters - recursive
  m_rootNode->update(treeParms);

  m_name = "OnlineTree";
  
}

OnlineTree::~OnlineTree() {
    delete m_rootNode;
}
    
void OnlineTree::update(Sample& sample) {
  //increment counter for obs passing through
  m_counter += sample.w;

  //make a prediction about this obs before update
  Result treeResult;
  eval(sample, treeResult);
  
  if (treeResult.prediction != sample.y) {
    m_oobe += sample.w;
  }

  //update tree parms using this obs
  m_rootNode->update(sample);
}

void OnlineTree::eval(Sample& sample, Result& result) {
    m_rootNode->eval(sample, result);
}

vector<MatrixXd> OnlineTree::exportParms() {
  vector<MatrixXd> ret;

  if(m_numNodes > 0) {
    VectorXd nodeParms;
    nodeParms = m_rootNode->exportParms(); // parms for the root node
    MatrixXd treeParms(m_numNodes, nodeParms.size()); // matrix to collect everything
  
    //initialize the collector of tree information
    vector<VectorXd> treeParmsVector;
  
    //add information from the root node to the vector
    treeParmsVector.push_back(nodeParms);

    //proceed recursively through tree adding info vector for each node
     if(m_numNodes > 1) {
       m_rootNode->exportChildParms(treeParmsVector);
     }

    //combine information from the vector back into the MatrixXd
    for(int i=0; i < treeParmsVector.size(); ++i) {
      treeParms.row(i) = treeParmsVector[i];
    }
  
    //return a vector since classifier requires
    ret.push_back(treeParms);
  }
  return(ret);
}

void OnlineTree::printInfo() {
  cout << "Tree Info: ";
  cout << "m_numNodes: " << m_numNodes << std::endl;
}

void OnlineTree::print() {
  cout << "Tree details: " << std::endl;
  vector<MatrixXd> treeParms = exportParms();
  if(treeParms.size() > 0) {
    cout << treeParms[0] << std::endl;
  }
}

double OnlineTree::getOOBE() {
  return(m_oobe);
}

double OnlineTree::getCounter() {
  return(m_counter);
}

pair<VectorXd,VectorXd> OnlineTree::getFeatRange() {
  return(pair<VectorXd, VectorXd> (*m_minFeatRange, *m_maxFeatRange));
}

void OnlineTree::updateFeatRange(VectorXd minFeatRange, VectorXd maxFeatRange) {
  //update the min and max feature range to extend if needed
  VectorXd newMinFeatRange = *m_minFeatRange;
  VectorXd newMaxFeatRange = *m_maxFeatRange;

  for(int i=0; i < minFeatRange.size(); ++i) {
    if(minFeatRange(i) < newMinFeatRange(i)) {
      newMinFeatRange(i) = minFeatRange(i);
    }
    if(maxFeatRange(i) > newMaxFeatRange(i)) {
      newMaxFeatRange(i) = maxFeatRange(i);
    }
  }

  m_minFeatRange = &newMinFeatRange;
  m_maxFeatRange = &newMaxFeatRange;
}


/****************************************************************************************
 *
 *  ONLINE RF CONSTRUCTORS AND METHODS 
 *
 ******************************************************************************************/

//version to construct using randomization
OnlineRF::OnlineRF(const Hyperparameters& hp, const int& numClasses, const int& numFeatures,
		   VectorXd minFeatRange, VectorXd maxFeatRange) :
  Classifier(hp, numClasses), m_counter(0.0), m_oobe(0.0), m_numClasses(&numClasses), 
  m_hp(&hp), m_minFeatRange(minFeatRange), m_maxFeatRange(maxFeatRange) {
  OnlineTree *tree;
  for (int nTree = 0; nTree < hp.numTrees; ++nTree) {
    tree = new OnlineTree(hp, numClasses, numFeatures, m_minFeatRange, m_maxFeatRange);
    m_trees.push_back(tree);
  }
  m_name = "OnlineRF";
}

//version to construction from a set of parameters
OnlineRF::OnlineRF(const vector<MatrixXd> orfParms, const Hyperparameters& hp,
		   const int& numClasses, double oobe, double counter,
		   VectorXd minFeatRange, VectorXd maxFeatRange) :
  Classifier(hp, numClasses), m_counter(counter), m_oobe(oobe),
  m_hp(&hp), m_numClasses(&numClasses),
  m_minFeatRange(minFeatRange), m_maxFeatRange(maxFeatRange) {
  OnlineTree *tree;
  for (int nTree = 0; nTree < m_hp->numTrees; ++nTree) {
    //create the trees using method to construct from parameters
    //initializing oobe and counter to 0 until i can figure that out
    tree = new OnlineTree(orfParms[nTree], hp, numClasses, 0, 0.0, 
    			  m_minFeatRange, m_maxFeatRange);
    m_trees.push_back(tree);

    //    cout << "treeParms size: " << orfParms[nTree].rows() << ", " << orfParms[nTree].cols() << "\n";

  }

  m_name = "OnlineRF";
}

OnlineRF::~OnlineRF() {
  for (int nTree = 0; nTree < m_hp->numTrees; ++nTree) {
    delete m_trees[nTree];
  }
}

void OnlineRF::update(Sample& sample) {
  m_counter += sample.w;
  Result result(*m_numClasses), treeResult;
  int numTries;
  for (int nTree = 0; nTree < m_hp->numTrees; ++nTree) {
    numTries = poisson(1.0);
    if (numTries) {
      for (int nTry = 0; nTry < numTries; ++nTry) {
	m_trees[nTree]->update(sample);	
      }
    } else {
      m_trees[nTree]->eval(sample, treeResult);
      result.confidence += treeResult.confidence;
    }
  }
  
  int pre;
  result.confidence.maxCoeff(&pre);
  if (pre != sample.y) {
    m_oobe += sample.w;
  }
}

void OnlineRF::eval(Sample& sample, Result& result) {
  Result treeResult;
  MatrixXd iteAll(m_hp->numTrees, *m_numClasses);

  for (int nTree = 0; nTree < m_hp->numTrees; ++nTree) {
    //calculate the prediction for the tree
    m_trees[nTree]->eval(sample, treeResult);
    //calculate the aggregate confidences and ITEs
    result.confidence += treeResult.confidence;
    if(m_hp->causal == true) {
      result.ite += treeResult.ite;

      //copy all ITE estimates from the tree into the matrix
      iteAll.row(nTree) = treeResult.ite;
    }
  }

  //average confidence
  result.confidence /= m_hp->numTrees;

  //prediction is associated with the max confidence
  result.confidence.maxCoeff(&result.prediction);

  if(m_hp->causal == true) {
    //mean ITE estimate
    result.ite /= m_hp->numTrees;

    //all ITE estimates
    result.iteAllTrees = iteAll;
  }
}

//return the parameters updated by the update method
vector<MatrixXd> OnlineRF::exportParms() {
  vector<MatrixXd> out;
  for(int nTree=0; nTree < m_trees.size(); ++nTree) {
    vector<MatrixXd> treeParmsVec = m_trees[nTree]->exportParms();
    out.push_back(treeParmsVec[0]);
  }
  return(out);
}

double OnlineRF::getOOBE() {
  return(m_oobe);
}

double OnlineRF::getCounter() {
  return(m_counter);
}

void OnlineRF::printInfo() {
  cout << "RF Info: ";
  cout << "Number of trees: " << m_trees.size() << std::endl;
}

void OnlineRF::print() {
  cout << "RF details: " << std::endl;  
  vector<MatrixXd> rfParms = exportParms();
  for(int nTree=0; nTree < rfParms.size(); ++nTree) {
    cout << "\tTree: " << nTree << std::endl;
    cout << "\t\t";
    cout << rfParms[nTree] << std::endl;
  }
}

pair<VectorXd,VectorXd> OnlineRF::getFeatRange() {
  return(pair<VectorXd, VectorXd> (m_minFeatRange, m_maxFeatRange));
}

void OnlineRF::updateFeatRange(VectorXd minFeatRange, VectorXd maxFeatRange) {
  //update the min and max feature range to extend if needed
  for(int i=0; i < minFeatRange.size(); ++i) {
    if(minFeatRange(i) < m_minFeatRange(i)) {
      m_minFeatRange(i) = minFeatRange(i);
    }
    if(maxFeatRange(i) > m_maxFeatRange(i)) {
      m_maxFeatRange(i) = maxFeatRange(i);
    }
  }
}
