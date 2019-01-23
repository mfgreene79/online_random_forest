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

#include "online_rf.h"

RandomTest::RandomTest(const int& numClasses, const int& numFeatures, const VectorXd &minFeatRange, const VectorXd &maxFeatRange) :
    m_numClasses(&numClasses), m_trueCount(0.0), m_falseCount(0.0),
    m_trueStats(VectorXd::Zero(numClasses)), m_falseStats(VectorXd::Zero(numClasses)) {
    m_feature = randDouble(0, numFeatures + 1);
    m_threshold = randDouble(minFeatRange(m_feature), maxFeatRange(m_feature));
}

void RandomTest::update(const Sample& sample) {
    updateStats(sample, eval(sample));
}
    
bool RandomTest::eval(const Sample& sample) const {
    return (sample.x(m_feature) > m_threshold) ? true : false;
}
    
double RandomTest::score() const {
    double trueScore = 0.0, falseScore = 0.0, p;
    if (m_trueCount) {
        for (int nClass = 0; nClass < *m_numClasses; nClass++) {
            p = m_trueStats[nClass] / m_trueCount;
            trueScore += p * (1 - p);
        }
    }
        
    if (m_falseCount) {
        for (int nClass = 0; nClass < *m_numClasses; nClass++) {
            p = m_falseStats[nClass] / m_falseCount;
            falseScore += p * (1 - p);
        }
    }
        
    return (m_trueCount * trueScore + m_falseCount * falseScore) / (m_trueCount + m_falseCount + 1e-16);
}
    
pair<VectorXd, VectorXd > RandomTest::getStats() const {
    return pair<VectorXd, VectorXd> (m_trueStats, m_falseStats);
}

void RandomTest::updateStats(const Sample& sample, const bool& decision) {
    if (decision) {
        m_trueCount += sample.w;
        m_trueStats(sample.y) += sample.w;
    } else {
        m_falseCount += sample.w;
        m_falseStats(sample.y) += sample.w;
    }
}    

pair<int,double> RandomTest::getParms() {
  //fetch the parms for the RandomTest as a vector
  return pair<int, double> (m_feature, m_threshold);
}

//version for the root node
OnlineNode::OnlineNode(const Hyperparameters& hp, const int& numClasses, const int& numFeatures, const VectorXd& minFeatRange, const VectorXd& maxFeatRange, 
                       const int& depth, int nodeNumber, MatrixXd& parms) :
    m_numClasses(&numClasses), m_depth(depth), m_isLeaf(true), m_hp(&hp), m_label(-1),
    m_counter(0.0), m_parentCounter(0.0), m_labelStats(VectorXd::Zero(numClasses)),
    m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange), m_nodeNumber(nodeNumber), m_parms(&parms) {
    // Creating random tests
    for (int nTest = 0; nTest < hp.numRandomTests; nTest++) {
        m_onlineTests.push_back(new RandomTest(numClasses, numFeatures, minFeatRange, maxFeatRange));
    }
    //information about the node changed - update to matrix
    //cout << "updating parms when creating node: " << nodeNumber << std::endl ;
    VectorXd nodeParms = getParms();
    updateParms(nodeParms);
    //cout << "m_parms: " << *m_parms << std::endl ;
}
    
//version for those below the root node
OnlineNode::OnlineNode(const Hyperparameters& hp, const int& numClasses, const int& numFeatures, const VectorXd& minFeatRange, const VectorXd& maxFeatRange, 
                       const int& depth, const VectorXd& parentStats, int nodeNumber, int parentNodeNumber,  MatrixXd& parms) :
    m_numClasses(&numClasses), m_depth(depth), m_isLeaf(true), m_hp(&hp), m_label(-1),
    m_counter(0.0), m_parentCounter(parentStats.sum()), m_labelStats(parentStats),
    m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange), m_nodeNumber(nodeNumber), m_parentNodeNumber(parentNodeNumber), m_parms(&parms) {
    m_labelStats.maxCoeff(&m_label);
    // Creating random tests
    for (int nTest = 0; nTest < hp.numRandomTests; nTest++) {
        m_onlineTests.push_back(new RandomTest(numClasses, numFeatures, minFeatRange, maxFeatRange));
    }
    //information about the node changed - update to matrix
    //cout << "updating parms when creating node: " << nodeNumber << std::endl ;
    VectorXd nodeParms = getParms();
    updateParms(nodeParms);
    //cout << "m_parms: " << *m_parms << std::endl ;
}
    
OnlineNode::~OnlineNode() {
    if (!m_isLeaf) {
        delete m_leftChildNode;
        delete m_rightChildNode;
        delete m_bestTest;
    } else {
        for (int nTest = 0; nTest < m_hp->numRandomTests; nTest++) {
            delete m_onlineTests[nTest];
        }
    }
}

//append parms overwriting the pointer adding one row
void OnlineNode::updateParms(VectorXd nodeParms) {

  //first - check to see if parms with this node number exist in the matrix
  int newNodeNumber = nodeParms(0);
  MatrixXd m_parms_vals = *m_parms;
  VectorXd nodeNumbers = m_parms->col(0);

  //cout << "newNodeNumber: " << newNodeNumber << std::endl;

  //get the index that is equal to this node number if it exists
  int idx = -1;
  for(int i=0; i < nodeNumbers.size(); ++i) {
    if(newNodeNumber == nodeNumbers(i)) {
      idx = i;
    }
  }

  if(idx >= 0 || newNodeNumber == 0) {
    //overwrite the row if this is not a new node number OR this is the first row
    //cout << "overwriting node " << newNodeNumber << std::endl;
    //copy matrix
    MatrixXd m_parms_new = m_parms_vals;

    //overwrite appropriate row
    m_parms_new.row(idx) = nodeParms;

    //write back into the m_parms the new matrix
    *m_parms = m_parms_new;

  } else { // or idx == -1
    //otherwise add a new row at the end
    //cout << "adding new node " << newNodeNumber << std::endl;
    MatrixXd m_parms_new(m_parms_vals.rows() + 1, m_parms_vals.cols());
    //copy in existing parms in the matrix
    for(int i=0;i < m_parms_vals.rows(); ++i) {
      m_parms_new.row(i) = m_parms_vals.row(i);
    }
    //set last element equal to the nodeParms from this node
    m_parms_new.row(m_parms_new.rows() - 1) = nodeParms;
    //write back into the m_parms the new matrix
    *m_parms = m_parms_new;
  }
}
    
void OnlineNode::update(const Sample& sample) {
    m_counter += sample.w;
    m_labelStats(sample.y) += sample.w;

    //information about the node changed - update to matrix
    //cout << "updating node after counters: " << m_nodeNumber << std::endl ;
    VectorXd nodeParms = getParms();
    updateParms(nodeParms);
    //cout << "m_parms: " << *m_parms << std::endl ;

    if (m_isLeaf) {
        // Update online tests
        for (vector<RandomTest*>::iterator itr = m_onlineTests.begin(); itr != m_onlineTests.end(); ++itr) {
            (*itr)->update(sample);
        }

        // Update the label
        m_labelStats.maxCoeff(&m_label);

	//information about the node changed - update to matrix
	//cout << "updating leaf node: " << m_nodeNumber << std::endl ;
	VectorXd nodeParms = getParms();
	updateParms(nodeParms);
	//cout << "m_parms: " << *m_parms << std::endl ;

        // Decide for split
        if (shouldISplit()) {
            m_isLeaf = false;

	    //information about the node changed - update to matrix
	    //cout << "updating node after split decision: " << m_nodeNumber << std::endl ;
	    VectorXd nodeParms = getParms();
	    updateParms(nodeParms);
	    //cout << "m_parms: " << *m_parms << std::endl ;

            // Find the best online test
            int nTest = 0, minIndex = 0;
            double minScore = 1, score;
            for (vector<RandomTest*>::const_iterator itr(m_onlineTests.begin()); itr != m_onlineTests.end(); ++itr, nTest++) {
                score = (*itr)->score();
                if (score < minScore) {
                    minScore = score;
                    minIndex = nTest;
                }
            }
            m_bestTest = m_onlineTests[minIndex];
            for (int nTest = 0; nTest < m_hp->numRandomTests; nTest++) {
                if (minIndex != nTest) {
                    delete m_onlineTests[nTest];
                }
            }

	    //Figure out the next available nodeNumber - saved in first location
	    int newNodeNumber = m_parms->col(0).maxCoeff();
	    ++newNodeNumber;
	    
            // Split - initializing with versions beyond the root node
            pair<VectorXd, VectorXd> parentStats = m_bestTest->getStats();
            m_rightChildNode = new OnlineNode(*m_hp, *m_numClasses,
					      m_minFeatRange->rows(), *m_minFeatRange, 
					      *m_maxFeatRange, m_depth + 1, 
					      parentStats.first, newNodeNumber, 
					      m_nodeNumber, *m_parms);
            m_leftChildNode = new OnlineNode(*m_hp, *m_numClasses, m_minFeatRange->rows(),
					     *m_minFeatRange, *m_maxFeatRange, m_depth + 1,
                                             parentStats.second, newNodeNumber + 1, 
					     m_nodeNumber, *m_parms);
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
    if (m_isLeaf) {
        if (m_counter + m_parentCounter) {
            result.confidence = m_labelStats / (m_counter + m_parentCounter);
            result.prediction = m_label;
        } else {
            result.confidence = VectorXd::Constant(m_labelStats.rows(), 1.0 / *m_numClasses);
            result.prediction = 0;
        }
    } else {
        if (m_bestTest->eval(sample)) {
            m_rightChildNode->eval(sample, result);
        } else {
            m_leftChildNode->eval(sample, result);
        }
    }
}

bool OnlineNode::shouldISplit() const {
    bool isPure = false;
    for (int nClass = 0; nClass < *m_numClasses; nClass++) {
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

VectorXd OnlineNode::getParms() {
  //returns information about the node
    //7 variables: m_nodeNumber, m_parentNodeNumber, m_depth, m_isLeaf, m_label, m_counter, m_parentCounter,
    //2 sizes: m_labelStats.size(), m_onlineTests.size()
    //m_labelStats.size(): one entry for each value of labelStats
  //m_onlineTests.size(): two entries for each value of onlineTests
    int vec_size = 7 + 2 + m_labelStats.size() + 2 * m_onlineTests.size();
    //cout << "\t vec_size: " << vec_size << std::endl ;
    VectorXd nodeParms(vec_size);

    //fetch the private parameters and save into the Node parms object
    int pos = 0;
    nodeParms(0) = static_cast<double>(m_nodeNumber);
    nodeParms(1) = static_cast<double>(m_parentNodeNumber);
    nodeParms(2) = static_cast<double>(m_depth);
    nodeParms(3) = static_cast<double>(m_isLeaf);
    nodeParms(4) = static_cast<double>(m_label);
    nodeParms(5) = static_cast<double>(m_counter);
    nodeParms(6) = static_cast<double>(m_parentCounter);
    nodeParms(7) = static_cast<double>(m_labelStats.size());
    nodeParms(8) = static_cast<double>(m_onlineTests.size());
    pos = 9;

    //copy in the label stats
    for(int l=0; l < m_labelStats.size();++l) {
      nodeParms(pos + l) = static_cast<double>(m_labelStats(l));
    } 
    pos = pos + m_labelStats.size();

    //copy in the random test information
    for(int rt=0; rt <  m_onlineTests.size(); ++rt) {
      //pair<int, double> (m_feature, m_threshold);
      pair<int,double> rt_parms = m_onlineTests[rt]->getParms();
       nodeParms(pos + 2 * rt) = static_cast<double>(rt_parms.first);
       nodeParms(pos + 2 * rt + 1) = static_cast<double>(rt_parms.second);
    }

    //cout << "nodeParms: " << nodeParms <<  std::endl ;

    return(nodeParms);
}

OnlineTree::OnlineTree(const Hyperparameters& hp, const int& numClasses, const int& numFeatures, 
                       const VectorXd& minFeatRange, const VectorXd& maxFeatRange) :
  Classifier(hp, numClasses), m_numClasses(&numClasses), m_hp(&hp) {

  //cout << "m_numClasses: " << *m_numClasses << std::endl;
  //cout << "m_hp->numRandomTests: " << m_hp->numRandomTests << std::endl;

    //initialize the parms matrix at the tree level - start with 1 rows and set to 0
  // what is to be the size of parms?
    int vec_size = 7 + 2 + *m_numClasses + 2 * (m_hp->numRandomTests);
    //cout << "vec_size: " << vec_size << std::endl;

    MatrixXd m_parms_temp(1, vec_size);
    m_parms = m_parms_temp;

  //initialize with root node version
    m_rootNode = new OnlineNode(hp, numClasses, numFeatures, minFeatRange, maxFeatRange, 0, 0, m_parms);
    m_name = "OnlineTree";
}

OnlineTree::~OnlineTree() {
    delete m_rootNode;
}
    
void OnlineTree::update(Sample& sample) {
  m_rootNode->update(sample);
}

void OnlineTree::eval(Sample& sample, Result& result) {
    m_rootNode->eval(sample, result);
}

// void OnlineTree::updateParms() {
//     cout << "4. here in tree->updateParms " <<  std::endl ;
//     //    m_rootNode->updateParms(parms);
// }

MatrixXd OnlineTree::getParmsMatrix() {
//     vector<MatrixXd> tree_parms_vec;
//     MatrixXd tree_parms = m_parms;
//     tree_parms_vec[0] = tree_parms;
//     return(tree_parms_vec);
  return(m_parms);
}

vector<MatrixXd> OnlineTree::getParms() {
  vector<MatrixXd> tree_parms_vec;
  MatrixXd tree_parms = getParmsMatrix();
  tree_parms_vec[0] = tree_parms;
  return(tree_parms_vec);
}


OnlineRF::OnlineRF(const Hyperparameters& hp, const int& numClasses, const int& numFeatures, const VectorXd& minFeatRange, const VectorXd& maxFeatRange) :
    Classifier(hp, numClasses), m_counter(0.0), m_oobe(0.0), m_numClasses(&numClasses), m_hp(&hp) {
    OnlineTree *tree;
    for (int nTree = 0; nTree < hp.numTrees; nTree++) {
        tree = new OnlineTree(hp, numClasses, numFeatures, minFeatRange, maxFeatRange);
        m_trees.push_back(tree);
    }
    m_name = "OnlineRF";
}

OnlineRF::~OnlineRF() {
    for (int nTree = 0; nTree < m_hp->numTrees; nTree++) {
        delete m_trees[nTree];
    }
}
    
void OnlineRF::update(Sample& sample) {
    m_counter += sample.w;

    Result result(*m_numClasses), treeResult;

    int numTries;
    for (int nTree = 0; nTree < m_hp->numTrees; nTree++) {
        numTries = poisson(1.0);
        if (numTries) {
            for (int nTry = 0; nTry < numTries; nTry++) {
	      //PROBLEM COMES FROM THE FOLLOWING LINE - TRACE
	      m_trees[nTree]->update(sample);

	      MatrixXd treeParms = m_trees[nTree]->getParmsMatrix();
	      //check if there is already an entry for this tree
	      if(rf_parms.size() <= nTree) {
		rf_parms.push_back(treeParms);
	      } else {
		rf_parms[nTree] = treeParms;
	      }

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
    for (int nTree = 0; nTree < m_hp->numTrees; nTree++) {
        m_trees[nTree]->eval(sample, treeResult);
        result.confidence += treeResult.confidence;
    }

    result.confidence /= m_hp->numTrees;
    result.confidence.maxCoeff(&result.prediction);
}


//return the parameters updated by the update method
vector<MatrixXd> OnlineRF::getParms() {
   return(rf_parms);


}

MatrixXd OnlineRF::getParmsMatrix() {
  MatrixXd rf_parmsMat(1,1);
  return(rf_parmsMat);

}
