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
 * Modified 2019 Michael Greene, mfgreene79@yahoo.com
 *  added functionality and enabled ability to connect to R
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
		       const Eigen::VectorXd &minFeatRange, const Eigen::VectorXd &maxFeatRange,
		       const Eigen::VectorXd &rootLabelStats, const double &rootCounter) :
  m_numClasses(&numClasses), m_hp(&hp),
  m_trueCount(0), m_falseCount(0),
  m_trueStats(Eigen::VectorXd::Zero(numClasses)), m_falseStats(Eigen::VectorXd::Zero(numClasses)),
  m_treatTrueCount(0), m_treatFalseCount(0),
  m_treatTrueStats(Eigen::VectorXd::Zero(numClasses)), m_treatFalseStats(Eigen::VectorXd::Zero(numClasses)),
  m_controlTrueCount(0), m_controlFalseCount(0),
  m_controlTrueStats(Eigen::VectorXd::Zero(numClasses)), m_controlFalseStats(Eigen::VectorXd::Zero(numClasses)),
  m_rootLabelStats(&rootLabelStats), m_rootCounter(&rootCounter)
 {
  m_feature = floor(randDouble(0, numFeatures));
  m_threshold = randDouble(minFeatRange(m_feature), maxFeatRange(m_feature));
}

//version to construct from parameters - not causal
RandomTest::RandomTest(const Hyperparameters& hp, const int& numClasses, 
		       int feature, double threshold,
		       Eigen::VectorXd trueStats, Eigen::VectorXd falseStats,
		       const Eigen::VectorXd &rootLabelStats, const double &rootCounter) : 
  m_feature(feature), m_threshold(threshold), m_numClasses(&numClasses), m_hp(&hp),
  m_trueCount(0.0), m_falseCount(0.0),
  m_trueStats(trueStats), m_falseStats(falseStats),
  m_treatTrueCount(0.0), m_treatFalseCount(0.0),
  m_treatTrueStats(Eigen::VectorXd::Zero(numClasses)), m_treatFalseStats(Eigen::VectorXd::Zero(numClasses)),
  m_controlTrueCount(0.0), m_controlFalseCount(0.0),
  m_controlTrueStats(Eigen::VectorXd::Zero(numClasses)), m_controlFalseStats(Eigen::VectorXd::Zero(numClasses)),
  m_rootLabelStats(&rootLabelStats), m_rootCounter(&rootCounter)
{
  m_trueCount = m_trueStats.sum();
  m_falseCount = m_falseStats.sum();
}

//version to construct from parameters - causal
RandomTest::RandomTest(const Hyperparameters& hp, const int& numClasses, 
		       int feature, double threshold,
		       Eigen::VectorXd treatTrueStats, Eigen::VectorXd treatFalseStats,
		       Eigen::VectorXd controlTrueStats, Eigen::VectorXd controlFalseStats,
		       const Eigen::VectorXd &rootLabelStats, const double &rootCounter) : 
  m_feature(feature), m_threshold(threshold), m_numClasses(&numClasses), m_hp(&hp),
  m_treatTrueStats(treatTrueStats), m_treatFalseStats(treatFalseStats),
  m_controlTrueStats(controlTrueStats), m_controlFalseStats(controlFalseStats),
  m_trueStats(Eigen::VectorXd::Zero(numClasses)), m_falseStats(Eigen::VectorXd::Zero(numClasses)),
  m_trueCount(0), m_falseCount(0), m_treatTrueCount(0), m_treatFalseCount(0),
  m_controlTrueCount(0), m_controlFalseCount(0),
  m_rootLabelStats(&rootLabelStats), m_rootCounter(&rootCounter)
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


    
double RandomTest::score() const {
  double theta; //value to minimize

  if(m_hp->causal == true) {
    //score the treatment and control counts
    //causal version - 
    // squared difference between ratios of treatment and control 
    // summed over classes
    // weighted average for left and right sides
    
    double theta=0.0;
    double trueScore = 0.0, falseScore = 0.0, treatP=0.0, controlP=0.0; 
    //total sum of square difference on the left side
    if (m_treatTrueCount > 0 & m_controlTrueCount > 0) {
      for (int nClass = 0; nClass < *m_numClasses; nClass++) {
	treatP = m_treatTrueStats[nClass] / m_treatTrueCount;
	controlP = m_controlTrueStats[nClass] / m_controlTrueCount;
	trueScore += pow(treatP - controlP, 2);
      }
    }
    //total sum of square difference on the right side
    if (m_treatFalseCount > 0 & m_controlFalseCount > 0) {
      for (int nClass = 0; nClass < *m_numClasses; nClass++) {
	treatP = m_treatFalseStats[nClass] / m_treatFalseCount;
	controlP = m_controlFalseStats[nClass] / m_controlFalseCount;
	falseScore += pow(treatP - controlP, 2);
      }
    }      

    theta = (m_trueCount * trueScore + m_falseCount * falseScore) / (m_trueCount + m_falseCount + 1e-16);
    //searching for minimum.  but goal is to maximize sum of squares
    //so looking to minimize  -SS
    theta = -theta;

  } else { //not causal tree
    //minimizing the score directly
    //non causal version allowing different methods
    double theta=0.0;
    if(m_hp->method == "gini") {
      double trueScore = 0.0, falseScore = 0.0, p;
      if (m_trueCount) {
	for (int nClass = 0; nClass < *m_numClasses; ++nClass) {
	  p = m_trueStats[nClass] / m_trueCount;
	  trueScore += p * (1 - p);
	}
      }      
      if (m_falseCount) {
	for (int nClass = 0; nClass < *m_numClasses; ++nClass) {
	  p = m_falseStats[nClass] / m_falseCount;
	  falseScore += p * (1 - p);
	}
      }      
      theta = (m_trueCount * trueScore + m_falseCount * falseScore) / (m_trueCount + m_falseCount + 1e-16);
    } else if(m_hp->method == "entropy") {
      double trueScore = 0.0, falseScore = 0.0, p;
      if (m_trueCount) {
	for (int nClass = 0; nClass < *m_numClasses; ++nClass) {
	  p = m_trueStats[nClass] / m_trueCount;
	  if(p > 0) {
	    trueScore += p * log2(p);
	  }
	}
      }      
      if (m_falseCount) {
	for (int nClass = 0; nClass < *m_numClasses; ++nClass) {
	  p = m_falseStats[nClass] / m_falseCount;
	  if(p > 0) {
	    falseScore += p * log2(p);
	  }
	}
      }      
      theta = (m_trueCount * trueScore + m_falseCount * falseScore) / (m_trueCount + m_falseCount + 1e-16);
    } else if(m_hp->method == "hellinger") {
      double trueScore = 0.0, falseScore = 0.0, p, p_root;
      Eigen::VectorXd rootLabelStats = *m_rootLabelStats;
      if(m_trueCount) {
	for (int nClass = 0; nClass < *m_numClasses; ++nClass) {
	  p_root = rootLabelStats(nClass) / *m_rootCounter;
	  p = m_trueStats[nClass] / m_trueCount;
	  trueScore += pow(sqrt(p) - sqrt(p_root),2);
	}
      }
      if(m_falseCount) {
	for (int nClass = 0; nClass < *m_numClasses; ++nClass) {
	  p_root = rootLabelStats(nClass) / *m_rootCounter;
	  p = m_falseStats[nClass] / m_falseCount;
	  falseScore += pow(sqrt(p) - sqrt(p_root),2);
	}
      }
      theta = sqrt((m_trueCount * trueScore + m_falseCount * falseScore) / (m_trueCount + m_falseCount + 1e-16));     
    } //close method == hellinger
  } //if not causal
  return(theta);
}
    
pair<Eigen::VectorXd, Eigen::VectorXd > RandomTest::getStats(std::string type) const {
  
  //  Eigen::VectorXd trueStats, falseStats;
  pair<Eigen::VectorXd, Eigen::VectorXd> outStats;

  if(type == "all") {
    outStats = pair<Eigen::VectorXd, Eigen::VectorXd> (m_trueStats, m_falseStats);
  } else if(type == "treat") {
    outStats = pair<Eigen::VectorXd, Eigen::VectorXd> (m_treatTrueStats, m_treatFalseStats);
  } else if(type == "control") {
   outStats = pair<Eigen::VectorXd, Eigen::VectorXd> (m_controlTrueStats, m_controlFalseStats);
  }
  
  return outStats;
}

void RandomTest::updateStats(const Sample& sample, const bool& decision) {
  if (decision) {
    m_trueCount += sample.w;
    m_trueStats(sample.y) += sample.w;
    if(sample.W) {
      m_treatTrueCount += sample.w;
      m_treatTrueStats(sample.y) += sample.w;
    } else {
      m_controlTrueCount += sample.w;
      m_controlTrueStats(sample.y) += sample.w;
    }
  } else {
    m_falseCount += sample.w;
    m_falseStats(sample.y) += sample.w;
    if(sample.W) {
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
		       const int& numFeatures, const Eigen::VectorXd& minFeatRange, 
		       const Eigen::VectorXd& maxFeatRange, 
                       const int& depth, int& numNodes) :
  m_numClasses(&numClasses), m_depth(depth), m_isLeaf(true), m_hp(&hp), m_label(-1),
  m_counter(0.0), m_parentCounter(0.0), m_labelStats(Eigen::VectorXd::Zero(numClasses)),
  m_treatLabelStats(Eigen::VectorXd::Zero(numClasses)), 
  m_controlLabelStats(Eigen::VectorXd::Zero(numClasses)),
  m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange), m_nodeNumber(0),
  m_numNodes(&numNodes), m_treatCounter(0.0), m_controlCounter(0.0),
  m_ite(Eigen::VectorXd::Zero(numClasses))
{
  //create pointers to the labelstats and counter - these will get passed down to child nodes
  m_rootLabelStats = &m_labelStats;
  m_rootCounter = &m_counter;

  // Creating random tests
  for (int nTest = 0; nTest < hp.numRandomTests; ++nTest) {
    m_onlineTests.push_back(new RandomTest(hp, numClasses, numFeatures, 
					   minFeatRange, maxFeatRange, 
					   *m_rootLabelStats, *m_rootCounter));
  }  
  setChildNodeNumbers(-1, -1);
  ++numNodes;


}

    
//version for those below the root node - not causal
OnlineNode::OnlineNode(const Hyperparameters& hp, const int& numClasses, 
		       const int& numFeatures, const Eigen::VectorXd& minFeatRange, 
		       const Eigen::VectorXd& maxFeatRange, 
                       const int& depth, const Eigen::VectorXd& parentStats, 
		       int nodeNumber, int parentNodeNumber, int& numNodes,
		       const Eigen::VectorXd &rootLabelStats, const double &rootCounter) :
  m_numClasses(&numClasses), m_depth(depth), m_isLeaf(true), m_hp(&hp), m_label(-1),
  m_counter(0.0), m_parentCounter(parentStats.sum()), m_labelStats(parentStats),
  m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange), m_nodeNumber(nodeNumber),
  m_parentNodeNumber(parentNodeNumber), m_numNodes(&numNodes), 
  m_treatCounter(0.0), m_controlCounter(0.0),
  m_ite(Eigen::VectorXd::Zero(numClasses)),
  m_rootLabelStats(&rootLabelStats), m_rootCounter(&rootCounter) {
  //calculate the label
  m_labelStats.maxCoeff(&m_label);
  
  // Creating random tests
  for (int nTest = 0; nTest < hp.numRandomTests; ++nTest) {
    m_onlineTests.push_back(new RandomTest(hp, numClasses, numFeatures, minFeatRange, maxFeatRange, rootLabelStats, rootCounter));
  }
  setChildNodeNumbers(-1, -1);
  ++numNodes;
}

//version below the root node - causal tree
OnlineNode::OnlineNode(const Hyperparameters& hp, const int& numClasses, 
		       const int& numFeatures, const Eigen::VectorXd& minFeatRange, 
		       const Eigen::VectorXd& maxFeatRange, 
                       const int& depth, 
		       const Eigen::VectorXd& treatParentStats, 
		       const Eigen::VectorXd& controlParentStats, 
		       int nodeNumber, int parentNodeNumber, int& numNodes,
		       const Eigen::VectorXd &rootLabelStats, const double &rootCounter) :
  m_numClasses(&numClasses), m_depth(depth), m_isLeaf(true), m_hp(&hp), m_label(-1),
  m_counter(0.0), m_parentCounter(0.0), m_treatCounter(treatParentStats.sum()), 
  m_controlCounter(controlParentStats.sum()),
  m_labelStats(Eigen::VectorXd::Zero(numClasses)),
  m_treatLabelStats(treatParentStats), m_controlLabelStats(controlParentStats),
  m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange), m_nodeNumber(nodeNumber),
  m_parentNodeNumber(parentNodeNumber), m_numNodes(&numNodes), 
  m_ite(Eigen::VectorXd::Zero(numClasses)),
  m_rootLabelStats(&rootLabelStats), m_rootCounter(&rootCounter) {


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
    m_onlineTests.push_back(new RandomTest(hp, numClasses, numFeatures, minFeatRange, maxFeatRange, rootLabelStats, rootCounter));
  }
  setChildNodeNumbers(-1, -1);
  ++numNodes;
}

//version to create from parameters - root version
OnlineNode::OnlineNode(const Eigen::VectorXd& nodeParms, const Hyperparameters& hp,
		       const int& numClasses, int& numNodes,
		       const Eigen::VectorXd& minFeatRange, const Eigen::VectorXd& maxFeatRange) : 
  m_hp(&hp), m_numNodes(&numNodes), m_numClasses(&numClasses),
  m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange),
  m_labelStats(Eigen::VectorXd::Zero(numClasses)),
  m_treatLabelStats(Eigen::VectorXd::Zero(numClasses)),
  m_controlLabelStats(Eigen::VectorXd::Zero(numClasses)),
  m_ite(Eigen::VectorXd::Zero(numClasses))
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

  m_rootLabelStats = &m_labelStats;
  m_rootCounter = &m_counter;

  
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
    Eigen::VectorXd treatTrueStats = Eigen::VectorXd::Zero(numClasses);
    Eigen::VectorXd treatFalseStats = Eigen::VectorXd::Zero(numClasses);
    Eigen::VectorXd controlTrueStats = Eigen::VectorXd::Zero(numClasses);
    Eigen::VectorXd controlFalseStats = Eigen::VectorXd::Zero(numClasses);
    
    //get the feature and threshold for the best test to point to later 
    int bt_feat = -1;
    double bt_threshold = 0;
    if(m_isLeaf == false) {
      bt_feat = nodeParms(pos);
      bt_threshold = nodeParms(pos + 1);
    } //close isLeaf

    //for all nodes (leaf or not) create the randomtests
    pos = 15 + 8 * numClasses;
    RandomTest* rt;
    for(int i=0; i < m_hp->numRandomTests; ++i) {
      int feature = nodeParms(pos + i * (2 * (1 + 2 * numClasses)));
      double threshold = nodeParms(pos + 1 + i * (2 * (1 + 2 * numClasses)));
      treatTrueStats = Eigen::VectorXd::Zero(numClasses);
      treatFalseStats = Eigen::VectorXd::Zero(numClasses);
      controlTrueStats = Eigen::VectorXd::Zero(numClasses);
      controlFalseStats = Eigen::VectorXd::Zero(numClasses);
      for(int j=0; j<numClasses; ++j) {
	treatTrueStats(j) = nodeParms(pos + 2 + j + i * (2 * (1 + 2 * numClasses)));
	treatFalseStats(j) = nodeParms(pos + 2 + j + numClasses + i * (2 * (1 + 2 * numClasses)));
	controlTrueStats(j) = nodeParms(pos + 2 + j + 2 * numClasses + i * (2 * (1 + 2 * numClasses)));
	controlFalseStats(j) = nodeParms(pos + 2 + j + 3 * numClasses + i * (2 * (1 + 2 * numClasses)));
      }
      rt = new RandomTest(hp, numClasses, feature, threshold, 
			  treatTrueStats, treatFalseStats,
			  controlTrueStats, controlFalseStats,
			  *m_rootLabelStats, *m_rootCounter);
      m_onlineTests.push_back(rt);
      
      //added 4/13/2019
      // check if the best test is the same as this random test.  if so point to it
      if(m_isLeaf == false) {
        if(bt_feat == feature & bt_threshold == threshold) {
          m_bestTest = rt;
        }
      }
    } //close i loop
    
  } else { //causal==false

    m_parentCounter = nodeParms(8);
  
    //copy in information for labelStats
    Eigen::VectorXd labelStats(*m_numClasses);


    int pos = 11; //starts at 11 - numClasses and numRandomTests already captured (9 and 10)
    for(int c=0; c < *m_numClasses; ++c) {
      labelStats(c) = nodeParms(pos + c);
    }
    m_labelStats = labelStats;
    
    pos = 11 + *m_numClasses;

    //set up random tests and best random test
    //advance by 2 - where the bestTest would be stored
    Eigen::VectorXd trueStats = Eigen::VectorXd::Zero(numClasses);
    Eigen::VectorXd falseStats = Eigen::VectorXd::Zero(numClasses);
    
    int bt_feat = -1;
    double bt_threshold = 0;
    if(m_isLeaf == false) { //when not a leaf create the best test
      bt_feat = nodeParms(pos);
      bt_threshold = nodeParms(pos + 1);
      
    } //close isLeaf

    //for all nodes (leaf or not) create the randomtests
    pos = 13 + 3 * numClasses;
    RandomTest* rt;
    for(int i=0; i < m_hp->numRandomTests; ++i) {
      int feature = nodeParms(pos + i * (2 * (1 + numClasses)));
      double threshold = nodeParms(pos + 1 + i * (2 * (1 + numClasses)));
      trueStats = Eigen::VectorXd::Zero(numClasses);
      falseStats = Eigen::VectorXd::Zero(numClasses);
      for(int j=0; j<numClasses; ++j) {
	trueStats(j) = nodeParms(pos + 2 + j + i * (2 * (1 + numClasses)));
	falseStats(j) = nodeParms(pos + 2 + j + numClasses + i * (2 * (1 + numClasses)));
      }
      rt = new RandomTest(hp, numClasses, feature, threshold, trueStats, falseStats,
			  *m_rootLabelStats, *m_rootCounter);
      m_onlineTests.push_back(rt);
      
      if(m_isLeaf == false) {
        if(bt_feat == feature & bt_threshold == threshold) {
          m_bestTest = rt;
        }
      }
      
    } //close i loop
  } //close causal==false
} //close method


//version to create from parameters - after root version
OnlineNode::OnlineNode(const Eigen::VectorXd& nodeParms, const Hyperparameters& hp,
		       const int& numClasses, int& numNodes,
		       const Eigen::VectorXd& minFeatRange, const Eigen::VectorXd& maxFeatRange,
		       const Eigen::VectorXd &rootLabelStats, const double &rootCounter) : 
  m_hp(&hp), m_numNodes(&numNodes), m_numClasses(&numClasses),
  m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange),
  m_labelStats(Eigen::VectorXd::Zero(numClasses)),
  m_treatLabelStats(Eigen::VectorXd::Zero(numClasses)),
  m_controlLabelStats(Eigen::VectorXd::Zero(numClasses)),
  m_ite(Eigen::VectorXd::Zero(numClasses)),
  m_rootLabelStats(&rootLabelStats), m_rootCounter(&rootCounter)
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
    Eigen::VectorXd treatTrueStats = Eigen::VectorXd::Zero(numClasses);
    Eigen::VectorXd treatFalseStats = Eigen::VectorXd::Zero(numClasses);
    Eigen::VectorXd controlTrueStats = Eigen::VectorXd::Zero(numClasses);
    Eigen::VectorXd controlFalseStats = Eigen::VectorXd::Zero(numClasses);
    
    //get the feature and threshold for the best test to point to later 
    int bt_feat = -1;
    double bt_threshold = 0;
    if(m_isLeaf == false) {
      bt_feat = nodeParms(pos);
      bt_threshold = nodeParms(pos + 1);
    } //close isLeaf

    //for all nodes (leaf or not) create the randomtests
    pos = 15 + 8 * numClasses;
    RandomTest* rt;
    for(int i=0; i < m_hp->numRandomTests; ++i) {
      int feature = nodeParms(pos + i * (2 * (1 + 2 * numClasses)));
      double threshold = nodeParms(pos + 1 + i * (2 * (1 + 2 * numClasses)));
      treatTrueStats = Eigen::VectorXd::Zero(numClasses);
      treatFalseStats = Eigen::VectorXd::Zero(numClasses);
      controlTrueStats = Eigen::VectorXd::Zero(numClasses);
      controlFalseStats = Eigen::VectorXd::Zero(numClasses);
      for(int j=0; j<numClasses; ++j) {
	treatTrueStats(j) = nodeParms(pos + 2 + j + i * (2 * (1 + 2 * numClasses)));
	treatFalseStats(j) = nodeParms(pos + 2 + j + numClasses + i * (2 * (1 + 2 * numClasses)));
	controlTrueStats(j) = nodeParms(pos + 2 + j + 2 * numClasses + i * (2 * (1 + 2 * numClasses)));
	controlFalseStats(j) = nodeParms(pos + 2 + j + 3 * numClasses + i * (2 * (1 + 2 * numClasses)));
      }
      rt = new RandomTest(hp, numClasses, feature, threshold, 
			  treatTrueStats, treatFalseStats,
			  controlTrueStats, controlFalseStats,
			  rootLabelStats, rootCounter
			  );
      m_onlineTests.push_back(rt);
      
      //added 4/13/2019
      // check if the best test is the same as this random test.  if so point to it
      if(m_isLeaf == false) {
        if(bt_feat == feature & bt_threshold == threshold) {
          m_bestTest = rt;
        }
      }
    } //close i loop
    
  } else { //causal==false

    m_parentCounter = nodeParms(8);
  
    //copy in information for labelStats
    Eigen::VectorXd labelStats(*m_numClasses);


    int pos = 11; //starts at 11 - numClasses and numRandomTests already captured (9 and 10)
    for(int c=0; c < *m_numClasses; ++c) {
      labelStats(c) = nodeParms(pos + c);
    }
    m_labelStats = labelStats;
    
    pos = 11 + *m_numClasses;

    //set up random tests and best random test
    //advance by 2 - where the bestTest would be stored
    Eigen::VectorXd trueStats = Eigen::VectorXd::Zero(numClasses);
    Eigen::VectorXd falseStats = Eigen::VectorXd::Zero(numClasses);
    
    int bt_feat = -1;
    double bt_threshold = 0;
    if(m_isLeaf == false) { //when not a leaf create the best test
      bt_feat = nodeParms(pos);
      bt_threshold = nodeParms(pos + 1);
      
    } //close isLeaf

    //for all nodes (leaf or not) create the randomtests
    pos = 13 + 3 * numClasses;
    RandomTest* rt;
    for(int i=0; i < m_hp->numRandomTests; ++i) {
      int feature = nodeParms(pos + i * (2 * (1 + numClasses)));
      double threshold = nodeParms(pos + 1 + i * (2 * (1 + numClasses)));
      trueStats = Eigen::VectorXd::Zero(numClasses);
      falseStats = Eigen::VectorXd::Zero(numClasses);
      for(int j=0; j<numClasses; ++j) {
	trueStats(j) = nodeParms(pos + 2 + j + i * (2 * (1 + numClasses)));
	falseStats(j) = nodeParms(pos + 2 + j + numClasses + i * (2 * (1 + numClasses)));
      }
      rt = new RandomTest(hp, numClasses, feature, threshold, trueStats, falseStats,
			  rootLabelStats, rootCounter);
      m_onlineTests.push_back(rt);
      
      if(m_isLeaf == false) {
        if(bt_feat == feature & bt_threshold == threshold) {
          m_bestTest = rt;
        }
      }
      
    } //close i loop
  } //close causal==false
} // close method

    
OnlineNode::~OnlineNode() {
  if (m_isLeaf == false) {
    delete m_leftChildNode;
    delete m_rightChildNode;
    //delete m_bestTest; //already deleted by the below
  }  //else { //removed 4/13/2019 - not deleting onlineTests once best test is defined
  for (int nTest = 0; nTest < m_hp->numRandomTests; nTest++) {
    delete m_onlineTests[nTest];
  }
  //}
}

//set the child node numbers if needed 
void OnlineNode::setChildNodeNumbers(int rightChildNodeNumber, int leftChildNodeNumber) {
  m_rightChildNodeNumber = rightChildNodeNumber;
  m_leftChildNodeNumber = leftChildNodeNumber;
}
    
void OnlineNode::update(const Sample& sample) {
  m_counter += sample.w;
  m_labelStats(sample.y) += sample.w;

  //increment treatment and control stats if a causal tree
  if(m_hp->causal == true) {
    if(sample.W) {
      m_treatCounter += sample.w;
      m_treatLabelStats(sample.y) += sample.w;
    } else {
      m_controlCounter += sample.w;
      m_controlLabelStats(sample.y) += sample.w;
    }

    //update the ITE
    for(int i=0; i < *m_numClasses; ++i) {
      if(m_treatCounter > 0 & m_controlCounter > 0) {
	m_ite(i) = (m_treatLabelStats(i)/m_treatCounter) - (m_controlLabelStats(i)/m_controlCounter); 
      } else {
	m_ite(i) = 0;
      }
    }
  } //close causal == TRUE

  if (m_isLeaf == true) {
    // Update online tests
    for (vector<RandomTest*>::iterator itr = m_onlineTests.begin(); 
	    itr != m_onlineTests.end(); ++itr) {
      (*itr)->update(sample);
    }
    
    // Update the label
    m_labelStats.maxCoeff(&m_label);

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
      //updated delete from online node destructor to delete these 
//       for (int nTest = 0; nTest < m_hp->numRandomTests; ++nTest) {
//       	If (minIndex != nTest) {
//       	  delete m_onlineTests[nTest];
//       	}
//       }
      
      //Figure out the next available nodeNumber - then increment the one for the tree
      int newNodeNumber = *m_numNodes;

      // Split - initializing with versions beyond the root node
      if(m_hp->causal == false) {
	      pair<Eigen::VectorXd, Eigen::VectorXd> parentStats = m_bestTest->getStats("all");      

	      m_rightChildNode = new OnlineNode(*m_hp, *m_numClasses,
						m_minFeatRange->rows(), *m_minFeatRange, 
						*m_maxFeatRange, m_depth + 1, 
						parentStats.first, newNodeNumber, 
						m_nodeNumber, *m_numNodes,
						*m_rootLabelStats, *m_rootCounter);
	      m_leftChildNode = new OnlineNode(*m_hp, *m_numClasses, m_minFeatRange->rows(),
					       *m_minFeatRange, *m_maxFeatRange, m_depth + 1,
					       parentStats.second, newNodeNumber + 1, 
					       m_nodeNumber, *m_numNodes,
					       *m_rootLabelStats, *m_rootCounter);
      } else { // causal==true
	      pair<Eigen::VectorXd, Eigen::VectorXd> treatParentStats = m_bestTest->getStats("treat");      
	      pair<Eigen::VectorXd, Eigen::VectorXd> controlParentStats = m_bestTest->getStats("control");      

	      m_rightChildNode = new OnlineNode(*m_hp, *m_numClasses,
						m_minFeatRange->rows(), *m_minFeatRange, 
						*m_maxFeatRange, m_depth + 1, 
						treatParentStats.first, 
						controlParentStats.first, 
						newNodeNumber, 
						m_nodeNumber, *m_numNodes,
						*m_rootLabelStats, *m_rootCounter);
	      m_leftChildNode = new OnlineNode(*m_hp, *m_numClasses, m_minFeatRange->rows(),
					       *m_minFeatRange, *m_maxFeatRange, m_depth + 1,
					       treatParentStats.second,
					       controlParentStats.second,
					       newNodeNumber + 1, 
					       m_nodeNumber, *m_numNodes,
					       *m_rootLabelStats, *m_rootCounter);

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
	result.ite = Eigen::VectorXd::Zero(*m_numClasses);
      }
    } else {
      result.confidence = Eigen::VectorXd::Constant(m_labelStats.rows(), 1.0 / *m_numClasses);
      result.prediction = 0;
      result.ite = Eigen::VectorXd::Zero(*m_numClasses);
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
void OnlineNode::update(const Eigen::MatrixXd& treeParms) {
  if(m_isLeaf == false) { // if its a leaf then theres no splitting to do
    
    //search through matrix of parms to find the correct rows and make node parms
    int found=0;
    for(int i=0; i < treeParms.rows(); ++i) {
      Eigen::VectorXd nodeParmsVec = treeParms.row(i);
      int npv_nodeNumber = static_cast<int>(nodeParmsVec(0));
      if(npv_nodeNumber == m_rightChildNodeNumber) {
	m_rightChildNode = new OnlineNode(nodeParmsVec, *m_hp, *m_numClasses, *m_numNodes,
					  *m_minFeatRange, *m_maxFeatRange,
					  *m_rootLabelStats, *m_rootCounter);
	found++;
      } else if(npv_nodeNumber == m_leftChildNodeNumber) {
	m_leftChildNode = new OnlineNode(nodeParmsVec, *m_hp, *m_numClasses, *m_numNodes,
					 *m_minFeatRange, *m_maxFeatRange,
					 *m_rootLabelStats, *m_rootCounter);
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

Eigen::VectorXd OnlineNode::exportParms() {
  //create vector to export
  
  //see layout spreadsheet for accounting of length
  int vec_size;
  if(m_hp->causal == true) {
    vec_size = 15 + 8 * *m_numClasses + 2 * m_hp->numRandomTests * (1 + 2 * *m_numClasses);
  } else {
    vec_size = 13 + 3 * *m_numClasses + 2 * m_hp->numRandomTests * (1 + *m_numClasses);
  }
  Eigen::VectorXd nodeParms = Eigen::VectorXd::Zero(vec_size);  //initialize the vector with zeros
  
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
    pair<Eigen::VectorXd, Eigen::VectorXd> bt_treatStats;
    pair<Eigen::VectorXd, Eigen::VectorXd> bt_controlStats;
    if(m_isLeaf == false) { //if NOT a leaf then we dont have a best test but do have randomtests
      bt_parms = m_bestTest->getParms();
      bt_treatStats = m_bestTest->getStats("treat");
      bt_controlStats = m_bestTest->getStats("control");
    } else { //otherwise use zeros (and -1 for the feature)
      int bt1 = -1;
      double bt2 = 0;
      Eigen::VectorXd bt3 = Eigen::VectorXd::Zero(*m_numClasses);

      bt_parms = pair<int, double> (bt1, bt2);
      bt_treatStats=pair<Eigen::VectorXd, Eigen::VectorXd> (bt3, bt3);
      bt_controlStats=pair<Eigen::VectorXd, Eigen::VectorXd> (bt3, bt3);
    }
    //write bt information to the vector
    nodeParms(pos) = bt_parms.first;
    nodeParms(pos + 1) = bt_parms.second;
    
    //copy the information from trueStats and falseStats into the parms
    //m_numClass columns for m_trueStats and m_numClass cols for m_falseStats
    pos = 15 + 4 * *m_numClasses;
    Eigen::VectorXd treatTrueStats = bt_treatStats.first;
    Eigen::VectorXd treatFalseStats = bt_treatStats.second;
    Eigen::VectorXd controlTrueStats = bt_controlStats.first;
    Eigen::VectorXd controlFalseStats = bt_controlStats.second;
    
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
      pair<Eigen::VectorXd, Eigen::VectorXd> rt_treatStats = rt.getStats("treat");
      pair<Eigen::VectorXd, Eigen::VectorXd> rt_controlStats = rt.getStats("control");
      //feature
      nodeParms(pos) = static_cast<double>(rt_parms.first);
      //threshold
      nodeParms(pos + 1) = static_cast<double>(rt_parms.second);
      //copy in the true and false stats
      Eigen::VectorXd treatTrueStats = rt_treatStats.first;
      Eigen::VectorXd treatFalseStats = rt_treatStats.second;
      Eigen::VectorXd controlTrueStats = rt_controlStats.first;
      Eigen::VectorXd controlFalseStats = rt_controlStats.second;
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
    pair<Eigen::VectorXd, Eigen::VectorXd> bt_stats;
    if(m_isLeaf == false) { //if NOT a leaf then we dont have a best test but do have randomtests
      bt_parms = m_bestTest->getParms();
      bt_stats = m_bestTest->getStats();
    } else { //otherwise use zeros (and -1 for the feature)
      int bt1 = -1;
      double bt2 = 0;
      Eigen::VectorXd bt3 = Eigen::VectorXd::Zero(*m_numClasses);;
      Eigen::VectorXd bt4 = Eigen::VectorXd::Zero(*m_numClasses);

      bt_parms = pair<int, double> (bt1, bt2);
      bt_stats=pair<Eigen::VectorXd, Eigen::VectorXd> (bt3, bt4);
    }
    //write bt information to the vector
    nodeParms(pos) = bt_parms.first;
    nodeParms(pos + 1) = bt_parms.second;
    
    //copy the information from trueStats and falseStats into the parms
    //m_numClass columns for m_trueStats and m_numClass cols for m_falseStats
    pos = 13 + *m_numClasses;
    Eigen::VectorXd trueStats = bt_stats.first;
    Eigen::VectorXd falseStats = bt_stats.second;
    
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
      pair<Eigen::VectorXd, Eigen::VectorXd> rt_stats = rt.getStats();
      //feature
      nodeParms(pos) = static_cast<double>(rt_parms.first);
      //threshold
      nodeParms(pos + 1) = static_cast<double>(rt_parms.second);
      //copy in the true and false stats
      Eigen::VectorXd trueStats = rt_stats.first;
      Eigen::VectorXd falseStats = rt_stats.second;
      for(int j=0; j < *m_numClasses; ++j) {
	nodeParms(pos + 2 + j) = trueStats(j);
	nodeParms(pos + 2 + j + *m_numClasses) = falseStats(j);
      } //loop j
    } //loop i
  } //causal condition
  return(nodeParms);
}

//method to recursively return information - updating matrix at the tree level
void OnlineNode::exportChildParms(vector<Eigen::VectorXd> &treeParmsVector) {
  //add the right and left child parms to parms for the tree
  if(m_isLeaf == false) {
    //collect and export the parms if this is not a leaf
    Eigen::VectorXd rightParms = m_rightChildNode->exportParms();
    treeParmsVector.push_back(rightParms);

    Eigen::VectorXd leftParms = m_leftChildNode->exportParms();
    treeParmsVector.push_back(leftParms);

    //recurse to the next level if NOT a leaf
    m_rightChildNode->exportChildParms(treeParmsVector);
    m_leftChildNode->exportChildParms(treeParmsVector);
  }
}

double OnlineNode::getCount() {
  double out = m_counter + m_parentCounter;
  return(out);
}

//self score for importance calculations
double OnlineNode::score() {
  Eigen::VectorXd stats=m_labelStats;
  int count = m_counter + m_parentCounter; 
  std::string method=m_hp->method;

  double out=0.0;
  int numClasses = stats.size();
  if(method == "gini") {
    double score = 0.0, p;
    if (count) {
      for (int nClass = 0; nClass < numClasses; ++nClass) {
        p = stats[nClass] / count;
        score += p * (1 - p);
      }
    }      
    out = score;
  } else if(method == "entropy") {
    double score = 0.0, p;
    if (count) {
      for (int nClass = 0; nClass < numClasses; ++nClass) {
        p = stats[nClass] / count;
	if(p > 0) {
	  score += p * log2(p);
	}
      }
    out = score;
    }
  } else if(m_hp->method == "hellinger") {
    double score = 0.0, p, p_root;
    Eigen::VectorXd rootLabelStats = *m_rootLabelStats;
    if(count) {
      for (int nClass = 0; nClass < numClasses; ++nClass) {
	p_root = rootLabelStats(nClass) / *m_rootCounter;
	p = stats[nClass] / count;
	score += pow(sqrt(p) - sqrt(p_root),2);
      }
      out = sqrt(score);
    }
  }
  return(out);
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
                       const Eigen::VectorXd& minFeatRange, const Eigen::VectorXd& maxFeatRange) :
  //  Classifier(hp, numClasses), 
  m_numClasses(&numClasses), m_hp(&hp), m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange) {
  
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
OnlineTree::OnlineTree(const Eigen::MatrixXd& treeParms, const Hyperparameters& hp, 
		       const int& numClasses, double oobe, double counter,
		       const Eigen::VectorXd& minFeatRange, const Eigen::VectorXd& maxFeatRange) :
  //  Classifier(hp, treeParms(0,9)), 
  m_hp(&hp),
  m_oobe(oobe), m_counter(counter), 
  m_numClasses(&numClasses), m_numNodes(0),
  m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange) {

  //find the max node number from the treeParms matrix - position 0
  m_numNodes = treeParms.rows();
  
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

vector<Eigen::MatrixXd> OnlineTree::exportParms() {
  vector<Eigen::MatrixXd> ret;

  if(m_numNodes > 0) {
    Eigen::VectorXd nodeParms;
    nodeParms = m_rootNode->exportParms(); // parms for the root node
    Eigen::MatrixXd treeParms(m_numNodes, nodeParms.size()); // matrix to collect everything
  
    //initialize the collector of tree information
    vector<Eigen::VectorXd> treeParmsVector;
  
    //add information from the root node to the vector
    treeParmsVector.push_back(nodeParms);

    //proceed recursively through tree adding info vector for each node
     if(m_numNodes > 1) {
       m_rootNode->exportChildParms(treeParmsVector);
     }

    //combine information from the vector back into the Eigen::MatrixXd
    for(int i=0; i < treeParmsVector.size(); ++i) {
      treeParms.row(i) = treeParmsVector[i];
    }
  
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
  vector<Eigen::MatrixXd> treeParms = exportParms();
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

pair<Eigen::VectorXd,Eigen::VectorXd> OnlineTree::getFeatRange() {
  return(pair<Eigen::VectorXd, Eigen::VectorXd> (*m_minFeatRange, *m_maxFeatRange));
}

void OnlineTree::updateFeatRange(Eigen::VectorXd minFeatRange, Eigen::VectorXd maxFeatRange) {
  //update the min and max feature range to extend if needed
  Eigen::VectorXd newMinFeatRange = *m_minFeatRange;
  Eigen::VectorXd newMaxFeatRange = *m_maxFeatRange;

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
		   Eigen::VectorXd minFeatRange, Eigen::VectorXd maxFeatRange) :
  //Classifier(hp, numClasses), 
  m_counter(0.0), m_oobe(0.0), m_numClasses(&numClasses), 
  m_hp(&hp), m_minFeatRange(minFeatRange), m_maxFeatRange(maxFeatRange) {
  OnlineTree *tree;
  for (int nTree = 0; nTree < hp.numTrees; ++nTree) {
    tree = new OnlineTree(hp, numClasses, numFeatures, m_minFeatRange, m_maxFeatRange);
    m_trees.push_back(tree);
  }
  m_name = "OnlineRF";
}

//version to construction from a set of parameters
OnlineRF::OnlineRF(const vector<Eigen::MatrixXd> orfParms, const Hyperparameters& hp,
		   const int& numClasses, double oobe, double counter,
		   Eigen::VectorXd minFeatRange, Eigen::VectorXd maxFeatRange) :
  //Classifier(hp, numClasses), 
  m_counter(counter), m_oobe(oobe),
  m_hp(&hp), m_numClasses(&numClasses),
  m_minFeatRange(minFeatRange), m_maxFeatRange(maxFeatRange) {
  OnlineTree *tree;
  for (int nTree = 0; nTree < m_hp->numTrees; ++nTree) {
    //create the trees using method to construct from parameters
    //initializing oobe and counter to 0 until i can figure that out
    tree = new OnlineTree(orfParms[nTree], hp, numClasses, 0, 0.0, 
    			  m_minFeatRange, m_maxFeatRange);
    m_trees.push_back(tree);

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
  Eigen::MatrixXd iteAll(m_hp->numTrees, *m_numClasses);

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
vector<Eigen::MatrixXd> OnlineRF::exportParms() {
  vector<Eigen::MatrixXd> out;
  for(int nTree=0; nTree < m_trees.size(); ++nTree) {
    vector<Eigen::MatrixXd> treeParmsVec = m_trees[nTree]->exportParms();
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
  vector<Eigen::MatrixXd> rfParms = exportParms();
  for(int nTree=0; nTree < rfParms.size(); ++nTree) {
    cout << "\tTree: " << nTree << std::endl;
    cout << "\t\t";
    cout << rfParms[nTree] << std::endl;
  }
}

pair<Eigen::VectorXd,Eigen::VectorXd> OnlineRF::getFeatRange() {
  return(pair<Eigen::VectorXd, Eigen::VectorXd> (m_minFeatRange, m_maxFeatRange));
}

void OnlineRF::updateFeatRange(Eigen::VectorXd minFeatRange, Eigen::VectorXd maxFeatRange) {
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


Eigen::MatrixXd OnlineNode::getFeatureImportance() {
  //initialize matrix of zeros to collect the importances
  //first column is the importance, second column number of obs for weighting
  Eigen::MatrixXd featImp = Eigen::MatrixXd::Zero(m_minFeatRange->size(), 2);  
  //if not a leaf, get the importances (if leaf ignore)
  if(m_isLeaf == false) {
    //get the importance from the bestTest
    //    score = m_bestTest->score();
    double score, selfScore;
    double rightChildScore = m_rightChildNode->score();
    double leftChildScore = m_leftChildNode->score();
    double rightChildCount = m_rightChildNode->getCount();
    double leftChildCount = m_leftChildNode->getCount();
    double childrenScore = (rightChildScore * rightChildCount + leftChildScore * leftChildCount) / (rightChildCount + leftChildCount + 1e-16);
    
    if(m_hp->causal == true) { //causal models have negative MSE, need to flip to get max variance
      score = -childrenScore; 
    } else { //if not a causal tree then get change score from this node to best split
      selfScore = this->score();
      score = selfScore - childrenScore; //positive numbers are better for gini and entropy
    }

//     cout << "nodeNumber: " << m_nodeNumber << "\n";
//     cout << "selfScore: " << selfScore << "\n";
//     cout << "childrenScore: " << childrenScore << "\n";
//     cout << "score: " << score << "\n";

    //save into matrix 
    pair<int, double> bt_parms = m_bestTest->getParms();
    featImp(bt_parms.first, 0) += score;
    featImp(bt_parms.first, 1) += this->getCount();
    
    //add child node feature importances
    Eigen::MatrixXd rc_featImp = m_rightChildNode->getFeatureImportance();
    Eigen::MatrixXd lc_featImp = m_leftChildNode->getFeatureImportance();

    for(int i=0; i < m_minFeatRange->size(); i++) {
       featImp(i,0) += rc_featImp(i,0);
       featImp(i,1) += rc_featImp(i,1);
       featImp(i,0) += lc_featImp(i,0);
       featImp(i,1) += lc_featImp(i,1);
    }
  } //close m_isLeaf==false
  return(featImp);
}

Eigen::MatrixXd OnlineTree::getFeatureImportance() {
  //go through each node recursively to get the feature importances
  Eigen::MatrixXd featImp = m_rootNode->getFeatureImportance();
  return(featImp);
}

Eigen::MatrixXd OnlineRF::getFeatureImportance() {
  //total feature importance from the individual trees
  Eigen::MatrixXd featImp = Eigen::MatrixXd::Zero(m_minFeatRange.size(), 2);
  double totWgtImp=0;

  for(int nTree = 0; nTree < m_hp->numTrees; nTree++) {
    Eigen::MatrixXd treeFeatImp = m_trees[nTree]->getFeatureImportance();
    for(int i=0; i<m_minFeatRange.size(); i++) {
      featImp(i,0) += treeFeatImp(i,0);
      featImp(i,1) += treeFeatImp(i,1);
    }
  }

  //normalize to ensure totaling to 1.0
      //calculate the total importance for normalizing
  for(int i=0; i<featImp.rows(); i++) {
    totWgtImp += featImp(i,0) * featImp(i,1);
  } 

  //divide by total
  for(int i=0; i<featImp.rows(); i++) {
    featImp(i,0) = featImp(i,0) * featImp(i,1) / totWgtImp;
  } 
  return(featImp);
}

/// Method for training model with data
void OnlineRF::train(DataSet& dataset) {
  vector<int> randIndex;
  int sampRatio = dataset.m_numSamples / 10;
  vector<double> trainError(m_hp->numEpochs, 0.0);
  for (int nEpoch = 0; nEpoch < m_hp->numEpochs; ++nEpoch) {
    //permute the dataset
    randPerm(dataset.m_numSamples, randIndex);
    for (int nSamp = 0; nSamp < dataset.m_numSamples; ++nSamp) {
      if (m_hp->findTrainError == true) {
	// 	Result result(dataset.m_numClasses);
 	Result result(dataset.m_numClasses);
 	this->eval(dataset.m_samples[randIndex[nSamp]], result);
 	if (result.prediction != dataset.m_samples[randIndex[nSamp]].y) {
 	  trainError[nEpoch]++;
 	}
      }
      //update RF with datapoint
      this->update(dataset.m_samples[randIndex[nSamp]]);
    } //close nSamp loop
  } //close epoch loop 
} //close method

//// method for providing predictions from the model
vector<Result> OnlineRF::test(DataSet& dataset) {
    vector<Result> results;
    for (int nSamp = 0; nSamp < dataset.m_numSamples; nSamp++) {
        Result result(dataset.m_numClasses);
	this->eval(dataset.m_samples[nSamp], result);
        results.push_back(result);
    }
    return results;
}
