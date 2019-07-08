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
 *  added causal forest functionality, regression forest functionality
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

////// Classification Tree Versions
//version to construct with randomization
RandomTest::RandomTest(const Hyperparameters& hp,
		       const int& numClasses, const int& numFeatures, 
		       const Eigen::VectorXd &minFeatRange, const Eigen::VectorXd &maxFeatRange,
		       const Eigen::VectorXd &rootLabelStats, const double &rootCounter) :
  m_hp(&hp), m_numClasses(&numClasses), 
  m_rootLabelStats(&rootLabelStats), m_rootCounter(&rootCounter),
  m_trueCount(0), m_falseCount(0),
  m_trueStats(Eigen::VectorXd::Zero(numClasses)), m_falseStats(Eigen::VectorXd::Zero(numClasses)),
  m_treatTrueCount(0), m_treatFalseCount(0),
  m_treatTrueStats(Eigen::VectorXd::Zero(numClasses)), m_treatFalseStats(Eigen::VectorXd::Zero(numClasses)),
  m_controlTrueCount(0), m_controlFalseCount(0),
  m_controlTrueStats(Eigen::VectorXd::Zero(numClasses)), m_controlFalseStats(Eigen::VectorXd::Zero(numClasses))
 {
  m_feature = floor(randDouble(0, numFeatures));
  m_threshold = randDouble(minFeatRange(m_feature), maxFeatRange(m_feature));
}

//version to construct from parameters - not causal
RandomTest::RandomTest(const Hyperparameters& hp, const int& numClasses, 
		       int feature, double threshold,
		       Eigen::VectorXd trueStats, Eigen::VectorXd falseStats,
		       const Eigen::VectorXd &rootLabelStats, const double &rootCounter) : 
  m_hp(&hp), m_numClasses(&numClasses), 
  m_rootLabelStats(&rootLabelStats), m_rootCounter(&rootCounter),
  m_feature(feature), m_threshold(threshold), 
  m_trueCount(0.0),
  m_falseCount(0.0),
  m_trueStats(trueStats), 
  m_falseStats(falseStats),
  m_treatTrueCount(0.0), 
  m_treatFalseCount(0.0),
  m_treatTrueStats(Eigen::VectorXd::Zero(numClasses)), 
  m_treatFalseStats(Eigen::VectorXd::Zero(numClasses)),
  m_controlTrueCount(0.0), 
  m_controlFalseCount(0.0),
  m_controlTrueStats(Eigen::VectorXd::Zero(numClasses)), m_controlFalseStats(Eigen::VectorXd::Zero(numClasses))
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
  m_hp(&hp), m_numClasses(&numClasses), 
  m_rootLabelStats(&rootLabelStats), m_rootCounter(&rootCounter),
  m_feature(feature), m_threshold(threshold), 
  m_trueCount(0), m_falseCount(0), 
  m_trueStats(Eigen::VectorXd::Zero(numClasses)),
  m_falseStats(Eigen::VectorXd::Zero(numClasses)),
  m_treatTrueCount(0), 
  m_treatFalseCount(0),
  m_treatTrueStats(treatTrueStats), m_treatFalseStats(treatFalseStats),
  m_controlTrueCount(0), m_controlFalseCount(0),
  m_controlTrueStats(controlTrueStats), m_controlFalseStats(controlFalseStats)
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

////// Regression Tree Versions
//version to construct with randomization
RandomTest::RandomTest(const Hyperparameters& hp,
		       const int& numFeatures, 
		       const Eigen::VectorXd &minFeatRange, 
		       const Eigen::VectorXd &maxFeatRange,
		       const Eigen::VectorXd &rootYStats, 
		       const double &rootCounter) :
  m_hp(&hp),
  m_rootYStats(&rootYStats), m_rootCounter(&rootCounter),
  m_trueCount(0), m_falseCount(0),
  m_trueYMean(0.0), m_falseYMean(0.0),
  m_trueYVar(0.0), m_falseYVar(0.0),
  m_trueErr(0.0), m_falseErr(0.0),
  m_trueWCounts(Eigen::VectorXd::Zero(hp.numTreatments)),
  m_falseWCounts(Eigen::VectorXd::Zero(hp.numTreatments)),
  m_trueYStats(Eigen::VectorXd::Zero(hp.numTreatments)),
  m_falseYStats(Eigen::VectorXd::Zero(hp.numTreatments)),
  m_trueYVarStats(Eigen::VectorXd::Zero(hp.numTreatments)),
  m_falseYVarStats(Eigen::VectorXd::Zero(hp.numTreatments))
{
  m_feature = floor(randDouble(0, numFeatures));
  m_threshold = randDouble(minFeatRange(m_feature), maxFeatRange(m_feature));
}

//version to construct from known parameters - not causal
RandomTest::RandomTest(const Hyperparameters& hp, 
		       int feature, double threshold,
		       double trueYMean, double falseYMean,
		       double trueYVar, double falseYVar,
		       int trueCount, int falseCount,
		       double trueErr, double falseErr,
		       const Eigen::VectorXd &rootYStats, 
		       const double &rootCounter) :
  m_hp(&hp),
  m_rootYStats(&rootYStats), m_rootCounter(&rootCounter),
  m_feature(feature), m_threshold(threshold), 
  m_trueCount(trueCount), m_falseCount(falseCount),
  m_trueYMean(trueYMean), m_falseYMean(falseYMean),
  m_trueYVar(trueYVar), m_falseYVar(falseYVar),
  m_trueErr(trueErr), m_falseErr(falseErr)
{
}


//version to construct from known parameters - causal
RandomTest::RandomTest(const Hyperparameters& hp, 
		       int feature, double threshold,
		       double trueYMean, double falseYMean,
		       double trueYVar, double falseYVar,
		       int trueCount, int falseCount,
		       double trueErr, double falseErr,
		       Eigen::VectorXd trueWCounts, Eigen::VectorXd falseWCounts,
		       Eigen::VectorXd trueYStats, Eigen::VectorXd falseYStats,
		       Eigen::VectorXd trueYVarStats, Eigen::VectorXd falseYVarStats,
		       const Eigen::VectorXd &rootYStats, 
		       const double &rootCounter
		       ) :
  m_hp(&hp),
  m_rootYStats(&rootYStats), m_rootCounter(&rootCounter),
  m_feature(feature), m_threshold(threshold), 
  m_trueCount(trueCount), m_falseCount(falseCount),
  m_trueYMean(trueYMean), m_falseYMean(falseYMean),
  m_trueYVar(trueYVar), m_falseYVar(falseYVar),
  m_trueErr(trueErr), m_falseErr(falseErr),
  m_trueWCounts(trueWCounts),
  m_falseWCounts(falseWCounts),
  m_trueYStats(trueYStats),
  m_falseYStats(falseYStats),
  m_trueYVarStats(trueYVarStats),
  m_falseYVarStats(falseYVarStats)
{
}


void RandomTest::update(const Sample& sample) {
    updateStats(sample, eval(sample));
}
    
bool RandomTest::eval(const Sample& sample) const {
    return (sample.x(m_feature) > m_threshold) ? true : false;
}
 
double RandomTest::score() const {
  double out;
  if(m_hp->type == "classification") {
    out = this->scoreClassification();
  } else {
    out = this->scoreRegression();
  }
  return(out);
}

//Classification Tree Version   
double RandomTest::scoreClassification() const {
  double theta=0.0; //value to minimize

  if(m_hp->causal == true) {
    //score the treatment and control counts
    //causal version - 
    // squared difference between ratios of treatment and control 
    // summed over classes
    // weighted average for left and right sides    
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

/////Regression Tree Version
double RandomTest::scoreRegression() const {
  double theta=0.0; //value to minimize

  if(m_hp->causal == true) {
    //score the treatment and control statistics
    // weighted avg over treatments
    // weighted average for left and right sides
    //methods - 
    // mse: squared difference between ratios of each treatment versus control 
    // hellinger: difference between all treatment versions and overall 

    if(m_hp->type=="mse") {
      double trueMSE=0.0, falseMSE=0.0, trueTauHat2=0.0, falseTauHat2=0.0;
      //total sum of square difference on the left side and right side
      for(int nTreat=0; nTreat < m_hp->numTreatments; nTreat++) {
	//tauHat2 = (y(treat) - y(0))^2: squared error of treatment - control summed over all treatments
	trueTauHat2 += pow(m_trueYStats(nTreat) - m_trueYStats(0),2);
	falseTauHat2 += pow(m_falseYStats(nTreat) - m_falseYStats(0),2);
	
	//MSE is the weighted average of tauHats^2
	trueMSE += trueTauHat2 * m_trueWCounts(nTreat); 
	falseMSE += falseTauHat2 * m_falseWCounts(nTreat); 
      }
      //divide out weights to complete weighted average calculation
      if(m_trueWCounts.sum() > 0) {
	trueMSE /= static_cast<double>(m_trueWCounts.sum());
      }
      if(m_falseWCounts.sum() > 0) {
	falseMSE /= static_cast<double>(m_falseWCounts.sum());
      }
      
      theta = (m_trueWCounts.sum() * trueMSE + m_falseWCounts.sum() * falseMSE) / (m_trueWCounts.sum() + m_falseWCounts.sum() + 1e-16);
      
      //searching for minimum.  but goal is to maximize sum of squares between treatment and control
      //so looking to minimize  -SS
      theta = -theta;
    } else if(m_hp->type == "hellinger") {
      double trueScore = 0.0, falseScore = 0.0, p, p_root;
      Eigen::VectorXd rootYStats = *m_rootYStats;
      double trueCount = m_trueWCounts.sum();
      double falseCount = m_falseWCounts.sum();

      if(trueCount > 0) {
	for (int nTreat = 0; nTreat < m_hp->numTreatments; nTreat++) {
	  if(*m_rootCounter > 0) {
	    p_root = rootYStats(nTreat) / static_cast<double>(*m_rootCounter);
	  }
	  p = m_trueYStats(nTreat) / static_cast<double>(trueCount);
	  trueScore += pow(sqrt(p) - sqrt(p_root),2);
	}
      }
      if(falseCount > 0) {
	for (int nTreat = 0; nTreat < m_hp->numTreatments; nTreat++) {
	  if(*m_rootCounter > 0) {
	    p_root = rootYStats(nTreat) / static_cast<double>(*m_rootCounter);
	  }
	  p = m_falseYStats(nTreat) / static_cast<double>(falseCount);
	  falseScore += pow(sqrt(p) - sqrt(p_root),2);
	}
      }
      theta = sqrt((trueCount * trueScore + falseCount * falseScore) / (trueCount + falseCount + 1e-16));
    }
  } else { //not causal tree
    //minimizing the score directly
    // weighted average for left and right sides
    //methods -
    // mse: squared difference between y and yhat
    double trueMSE=0.0, falseMSE=0.0;
    if(m_trueCount > 0) {
      trueMSE = pow(m_trueErr,2) / static_cast<double>(m_trueCount);
    }
    if(m_falseCount > 0) {
      falseMSE = pow(m_falseErr,2) / static_cast<double>(m_falseCount);
    }
    theta = (m_trueCount * trueMSE + m_falseCount * falseMSE) / (m_trueCount + m_falseCount + 1e-16);

  } //if not causal
  return(theta);
}

//// Methods to fetch the statistics from the Random Test    
pair<int, double> RandomTest::getParms() {
  //fetch the parms for the RandomTest as a vector
  return pair<int, double> (m_feature, m_threshold);
}

pair<Eigen::VectorXd, Eigen::VectorXd > RandomTest::getStatsClassification(std::string type) const {
  
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

pair<int, int> RandomTest::getTotCounts() {
  pair<int, int> outCounts;
  outCounts = pair<int, int> (m_trueCount, m_falseCount);
  return outCounts;
}

pair<double, double> RandomTest::getYMeans() {
  pair<double, double> outMeans;
  outMeans = pair<double, double> (m_trueYMean, m_falseYMean);
  return outMeans;
}

pair<double, double> RandomTest::getYVars() {
  pair<double, double> outVars;
  outVars = pair<double, double> (m_trueYVar, m_falseYVar);
  return outVars;
}

pair<double, double> RandomTest::getErrs() {
  pair<double, double> outErrs;
  outErrs = pair<double, double> (m_trueErr, m_falseErr);
  return outErrs;
}

pair<Eigen::VectorXd, Eigen::VectorXd> RandomTest::getWCounts() {
  pair<Eigen::VectorXd, Eigen::VectorXd> outWCounts;
  outWCounts = pair<Eigen::VectorXd, Eigen::VectorXd> (m_trueWCounts, m_falseWCounts);
  return outWCounts;
}

pair<Eigen::VectorXd, Eigen::VectorXd> RandomTest::getYStats() {
  pair<Eigen::VectorXd, Eigen::VectorXd> outYStats;
  outYStats = pair<Eigen::VectorXd, Eigen::VectorXd> (m_trueYStats, m_falseYStats);
  return outYStats;
}

pair<Eigen::VectorXd, Eigen::VectorXd> RandomTest::getYVarStats() {
  pair<Eigen::VectorXd, Eigen::VectorXd> outYVarStats;
  outYVarStats = pair<Eigen::VectorXd, Eigen::VectorXd> (m_trueYVarStats, m_falseYVarStats);
  return outYVarStats;
}

///////Methods for updating the statistics
void RandomTest::updateStats(const Sample& sample, const bool& decision) {
  if(m_hp->type == "classification") {
    updateStatsClassification(sample, decision);
  } else {
    updateStatsRegression(sample, decision);
  }
}

void RandomTest::updateStatsClassification(const Sample& sample, const bool& decision) {
  if (decision) {
    m_trueCount += sample.w;
    m_trueStats(sample.yClass) += sample.w;
    if(sample.W) {
      m_treatTrueCount += sample.w;
      m_treatTrueStats(sample.yClass) += sample.w;
    } else {
      m_controlTrueCount += sample.w;
      m_controlTrueStats(sample.yClass) += sample.w;
    }
  } else {
    m_falseCount += sample.w;
    m_falseStats(sample.yClass) += sample.w;
    if(sample.W) {
      m_treatFalseCount += sample.w;
      m_treatFalseStats(sample.yClass) += sample.w;
    } else {
      m_controlFalseCount += sample.w;
      m_controlFalseStats(sample.yClass) += sample.w;
    }
  }
}

void RandomTest::updateStatsRegression(const Sample& sample, const bool& decision) {
  if (decision) { //update right side
    //update error
    m_trueErr += sample.w * (sample.yReg - m_trueYMean);

    //update mean and variance of y
    m_trueYVar = (m_trueYVar * m_trueCount + sample.w * pow(m_trueYMean - sample.yReg,2)) / (m_trueCount + sample.w);
    m_trueYMean = (m_trueYMean * m_trueCount + sample.yReg * sample.w) / (m_trueCount + sample.w);

    //update count
    m_trueCount += sample.w;
    //if causal, update the counts
    if(m_hp->causal==true) {
      m_trueYStats(sample.W) = (m_trueYStats(sample.W) * m_trueWCounts(sample.W) + sample.yReg * sample.w) / (m_trueWCounts(sample.W) + sample.w);
      m_trueWCounts(sample.W) += sample.w;

    }
  } else { //update left side
    //update error
    m_falseErr += sample.w * (sample.yReg - m_falseYMean);

    //update mean of y
    m_falseYVar = (m_falseYVar * m_falseCount + sample.w * pow(m_falseYMean - sample.yReg,2)) / (m_falseCount + sample.w);
    m_falseYMean = (m_falseYMean * m_falseCount + sample.yReg * sample.w) / (m_falseCount + sample.w);
    //update count
    m_falseCount += sample.w;

    //if causal, update the counts
    if(m_hp->causal==true) {
      m_falseYStats(sample.W) = (m_falseYStats(sample.W) * m_falseWCounts(sample.W) + sample.yReg * sample.w) / (m_falseWCounts(sample.W) + sample.w);
      m_falseWCounts(sample.W) += sample.w;

    } 
  } //close update left side
} //close method


// void RandomTest::print() {
//   cout << "m_feature: " << m_feature << ", threshold: " << m_threshold << std::endl;
// }

/****************************************************************************************
 *
 *  ONLINE NODE CONSTRUCTORS AND METHODS 
 *
 ******************************************************************************************/

/////// Classification Forest Constructors
//version for the root node
OnlineNode::OnlineNode(const Hyperparameters& hp, const int& numClasses, 
		       const int& numFeatures, const Eigen::VectorXd& minFeatRange, 
		       const Eigen::VectorXd& maxFeatRange, 
                       const int& depth, int& numNodes) :
  m_nodeNumber(0),
  m_numClasses(&numClasses),
  m_depth(depth), 
  m_isLeaf(true),
  m_hp(&hp),  
  m_label(-1),
  m_counter(0.0),
  m_treatCounter(0.0), m_controlCounter(0.0),
  m_parentCounter(0.0), 
  m_labelStats(Eigen::VectorXd::Zero(numClasses)),
  m_treatLabelStats(Eigen::VectorXd::Zero(numClasses)), 
  m_controlLabelStats(Eigen::VectorXd::Zero(numClasses)),
  m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange), 
  m_numNodes(&numNodes),
  m_tauHat(Eigen::VectorXd::Zero(numClasses))
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
  m_nodeNumber(nodeNumber),
  m_parentNodeNumber(parentNodeNumber), 
  m_numClasses(&numClasses), 
  m_depth(depth), 
  m_isLeaf(true), 
  m_hp(&hp), 
  m_label(-1),
  m_counter(0.0), 
  m_treatCounter(0.0), 
  m_controlCounter(0.0),
  m_parentCounter(parentStats.sum()), 
  m_labelStats(parentStats),
  m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange), 
  m_numNodes(&numNodes), 
  m_rootLabelStats(&rootLabelStats), m_rootCounter(&rootCounter),
  m_tauHat(Eigen::VectorXd::Zero(numClasses))
{
  //calculate the label
  m_labelStats.maxCoeff(&m_label);
  
  // Creating random tests
  for (int nTest = 0; nTest < hp.numRandomTests; ++nTest) {
    m_onlineTests.push_back(new RandomTest(hp, numClasses, numFeatures, 
					   minFeatRange, maxFeatRange, 
					   rootLabelStats, rootCounter));
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
  m_nodeNumber(nodeNumber),
  m_parentNodeNumber(parentNodeNumber), 
  m_numClasses(&numClasses), 
  m_depth(depth), 
  m_isLeaf(true), 
  m_hp(&hp), 
  m_label(-1), 
  m_counter(0.0),
  m_treatCounter(treatParentStats.sum()), 
  m_controlCounter(controlParentStats.sum()),
  m_parentCounter(0.0), 
  m_labelStats(Eigen::VectorXd::Zero(numClasses)),
  m_treatLabelStats(treatParentStats), m_controlLabelStats(controlParentStats),
  m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange), 
  m_numNodes(&numNodes), 
  m_rootLabelStats(&rootLabelStats), m_rootCounter(&rootCounter),
  m_tauHat(Eigen::VectorXd::Zero(numClasses))
{


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
      m_tauHat(i) = (m_treatLabelStats(i)/m_treatCounter) - (m_controlLabelStats(i)/m_controlCounter);
    } else {
      m_tauHat(i) = 0;
    }
  }
  
  // Creating random tests
  for (int nTest = 0; nTest < hp.numRandomTests; ++nTest) {
    m_onlineTests.push_back(new RandomTest(hp, numClasses, numFeatures, 
					   minFeatRange, maxFeatRange, 
					   rootLabelStats, rootCounter));
  }
  setChildNodeNumbers(-1, -1);
  ++numNodes;
}

//version to create from parameters - root version
OnlineNode::OnlineNode(const Eigen::VectorXd& nodeParms, const Hyperparameters& hp,
		       const int& numClasses, int& numNodes,
		       const Eigen::VectorXd& minFeatRange, const Eigen::VectorXd& maxFeatRange) : 
  m_numClasses(&numClasses), m_hp(&hp),
  m_labelStats(Eigen::VectorXd::Zero(numClasses)),
  m_treatLabelStats(Eigen::VectorXd::Zero(numClasses)),
  m_controlLabelStats(Eigen::VectorXd::Zero(numClasses)),
  m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange),
  m_numNodes(&numNodes),
  m_tauHat(Eigen::VectorXd::Zero(numClasses))
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

  //create pointers to the labelstats and counter - these will get passed down to child nodes
  m_rootLabelStats = &m_labelStats;
  m_rootCounter = &m_counter;
  
  //if causal need to extract treatment and control separately
  if(m_hp->causal == true) {
     int pos=8;
     //ite
     for(int l=0; l < numClasses; ++l) {
       m_tauHat(l) = static_cast<double>(nodeParms(pos+l));
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
  m_numClasses(&numClasses),  m_hp(&hp), 
  m_labelStats(Eigen::VectorXd::Zero(numClasses)),
  m_treatLabelStats(Eigen::VectorXd::Zero(numClasses)),
  m_controlLabelStats(Eigen::VectorXd::Zero(numClasses)),
  m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange),
  m_numNodes(&numNodes), 
  m_rootLabelStats(&rootLabelStats), 
  m_rootCounter(&rootCounter),
  m_tauHat(Eigen::VectorXd::Zero(numClasses))
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
       m_tauHat(l) = static_cast<double>(nodeParms(pos+l));
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
  
/////// Regression Forest Constructors
//version for the root node
OnlineNode::OnlineNode(const Hyperparameters& hp, const int& numFeatures, 
		       const Eigen::VectorXd& minFeatRange, 
		       const Eigen::VectorXd& maxFeatRange, 
		       const int& depth, int& numNodes):

  m_nodeNumber(0),
  m_numClasses(0),
  m_depth(depth), m_isLeaf(true),  
  m_hp(&hp),
  m_counter(0.0), 
  m_parentCounter(0.0),  
  m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange),
  m_numNodes(&numNodes), 
  m_wCounts(Eigen::VectorXd::Zero(hp.numTreatments)),
  m_yStats(Eigen::VectorXd::Zero(hp.numTreatments)),
  m_yVarStats(Eigen::VectorXd::Zero(hp.numTreatments)),
  m_yMean(0.0), m_yVar(0.0), m_err(0.0), 
  m_tauHat(Eigen::VectorXd::Zero(hp.numTreatments)),
  m_tauVarHat(Eigen::VectorXd::Zero(hp.numTreatments))
{

  //create pointers to the labelstats and counter - these will get passed down to child nodes
  m_rootYStats = &m_yStats;
  m_rootCounter = &m_counter;

  // Creating random tests
  for (int nTest = 0; nTest < hp.numRandomTests; ++nTest) {
    m_onlineTests.push_back(new RandomTest(hp, numFeatures, minFeatRange, maxFeatRange,
					   *m_rootYStats, *m_rootCounter));
  }  
  setChildNodeNumbers(-1, -1);
  ++numNodes;
}

//version for those below the root node - not causal
OnlineNode::OnlineNode(const Hyperparameters& hp, const int& numFeatures, 
		       const Eigen::VectorXd& minFeatRange, 
		       const Eigen::VectorXd& maxFeatRange, 
		       const int& depth,
		       const double parentCounter,
		       const double parentYMean,
		       const double parentYVar,
		       const double parentErr,
		       int nodeNumber, int parentNodeNumber, int& numNodes,
		       const Eigen::VectorXd &rootYStats, const double &rootCounter) :
  m_nodeNumber(nodeNumber),
  m_parentNodeNumber(parentNodeNumber),
  m_depth(depth),
  m_isLeaf(true), 
  m_hp(&hp),
  m_counter(0.0), m_parentCounter(parentCounter), 
  m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange), 
  m_numNodes(&numNodes),
  m_rootYStats(&rootYStats), 
  m_rootCounter(&rootCounter),
  m_yMean(parentYMean), m_yVar(parentYVar),
  m_err(parentErr)
{

  // Creating random tests
  for (int nTest = 0; nTest < hp.numRandomTests; ++nTest) {
    m_onlineTests.push_back(new RandomTest(hp, numFeatures, minFeatRange, maxFeatRange,
					   *m_rootYStats, *m_rootCounter));
  }
  setChildNodeNumbers(-1, -1);
  ++numNodes;
}

//version for those below the root node - causal
OnlineNode::OnlineNode(const Hyperparameters& hp, const int& numFeatures, 
		       const Eigen::VectorXd& minFeatRange, 
		       const Eigen::VectorXd& maxFeatRange, 
		       const int& depth,
		       const double parentCounter,
		       const double parentYMean,
		       const double parentYVar,
		       const double parentErr,
		       const Eigen::VectorXd& parentWCounts,
		       const Eigen::VectorXd& parentYVarStats,
		       const Eigen::VectorXd& parentYStats,
		       int nodeNumber, int parentNodeNumber, int& numNodes,
		       const Eigen::VectorXd &rootYStats, const double &rootCounter) :
  m_nodeNumber(nodeNumber),
  m_parentNodeNumber(parentNodeNumber), 
  m_depth(depth), 
  m_isLeaf(true), 
  m_hp(&hp), 
  m_counter(0.0), 
  m_parentCounter(parentCounter), 
  m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange), 
  m_numNodes(&numNodes),
  m_rootYStats(&rootYStats), 
  m_rootCounter(&rootCounter),
  m_wCounts(parentWCounts),
  m_yStats(parentYStats),
  m_yVarStats(parentYVarStats),
  m_yMean(parentYMean), 
  m_yVar(parentYVar),  
  m_err(parentErr),
  m_tauHat(Eigen::VectorXd::Zero(hp.numTreatments)),
  m_tauVarHat(Eigen::VectorXd::Zero(hp.numTreatments))
{

  // Creating random tests
  for (int nTest = 0; nTest < hp.numRandomTests; ++nTest) {
    m_onlineTests.push_back(new RandomTest(hp, numFeatures, minFeatRange, maxFeatRange,
					   *m_rootYStats, *m_rootCounter));
  }
  setChildNodeNumbers(-1, -1);
  ++numNodes;
}

//Version to initialize from a vector of information about the node - root version
OnlineNode::OnlineNode(const Eigen::VectorXd& nodeParms, const Hyperparameters& hp,
		       int& numNodes, const Eigen::VectorXd& minFeatRange, 
		       const Eigen::VectorXd& maxFeatRange):
  m_numClasses(0), m_hp(&hp),
  m_counter(0.0), 
  m_parentCounter(0.0), 
  m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange),
  m_numNodes(&numNodes),
  m_wCounts(Eigen::VectorXd::Zero(hp.numTreatments)),
  m_yStats(Eigen::VectorXd::Zero(hp.numTreatments)),
  m_yVarStats(Eigen::VectorXd::Zero(hp.numTreatments)),
  m_tauHat(Eigen::VectorXd::Zero(hp.numTreatments)),
  m_tauVarHat(Eigen::VectorXd::Zero(hp.numTreatments))
  
{

  //create pointers to the labelstats and counter - these will get passed down to child nodes
  m_rootYStats = &m_yStats;
  m_rootCounter = &m_counter;

  int numTreatments = hp.numTreatments; //save this for use in below positions. ==1 for non-causal

  //extract information about the node from the vector
  //common information whether causal or not
  m_nodeNumber = static_cast<int>(nodeParms(0));
  m_parentNodeNumber = static_cast<int>(nodeParms(1));
  m_rightChildNodeNumber = static_cast<int>(nodeParms(2));
  m_leftChildNodeNumber = static_cast<int>(nodeParms(3));
  m_depth = static_cast<int>(nodeParms(4));
  m_isLeaf = static_cast<bool>(nodeParms(5));
  m_counter = static_cast<double>(nodeParms(6));
  m_parentCounter = static_cast<double>(nodeParms(7));
  m_yMean = static_cast<double>(nodeParms(8)); //prediction for the node
  m_yVar = static_cast<double>(nodeParms(9)); //variance estimate for the node
  m_err = static_cast<double>(nodeParms(10)); //prediction for the node

  int pos=11;
  if(m_hp->causal == true) {
    //if causal need to extract tauHat mean and variance, treatment counts, and y statistics
     //mean
    for(int nTreat=0; nTreat < numTreatments; nTreat++) {
      m_tauHat(nTreat) = static_cast<double>(nodeParms(pos+nTreat));
    }
    pos=pos+numTreatments;
 
     //variance
    for(int nTreat=0; nTreat < numTreatments; nTreat++) {
      m_tauVarHat(nTreat) = static_cast<double>(nodeParms(pos+nTreat));
    }
    pos=pos+numTreatments;

    //extract treatment counts
    for(int nTreat=0; nTreat < numTreatments; nTreat++) {
      m_wCounts(nTreat) = static_cast<double>(nodeParms(pos+nTreat));
    }
    pos=pos+numTreatments;

    //extract statistics about y
    for(int nTreat=0; nTreat < numTreatments; nTreat++) {
      m_yStats(nTreat) = static_cast<double>(nodeParms(pos+nTreat));
    }
    pos=pos+numTreatments;

    //extract statistics about yvar
    for(int nTreat=0; nTreat < numTreatments; nTreat++) {
      m_yVarStats(nTreat) = static_cast<double>(nodeParms(pos+nTreat));
    }
    pos=pos+numTreatments;

  } //close causal == true
  
  //get the feature and threshold for the best test to point to later 
  int bt_feat = -1;
  double bt_threshold = 0;
  if(m_isLeaf == false) {
    bt_feat = nodeParms(pos);
    bt_threshold = nodeParms(pos + 1);
  } //close isLeaf
  pos = pos + 2;
  
  //for all nodes (leaf or not) create the randomtests
  
  RandomTest* rt;
  for(int nRandTest=0; nRandTest < m_hp->numRandomTests; nRandTest++) {
    int feature = nodeParms(pos);
    double threshold = nodeParms(pos + 1);
    pos = pos + 2;

    int trueYMean = nodeParms(pos);
    int trueYVar = nodeParms(pos+1);
    int trueCount = static_cast<int>(nodeParms(pos+2));
    int trueErr = nodeParms(pos+3);
    pos = pos + 4;
    
    int falseYMean = nodeParms(pos);
    int falseYVar = nodeParms(pos+1);
    int falseCount = static_cast<int>(nodeParms(pos+2));
    int falseErr = nodeParms(pos+3);
    pos = pos + 4;
    

    if(m_hp->causal == true) {
      //prepare vectors to create the random test
      Eigen::VectorXd trueWCounts = Eigen::VectorXd::Zero(numTreatments);
      Eigen::VectorXd falseWCounts = Eigen::VectorXd::Zero(numTreatments);
      Eigen::VectorXd trueYStats = Eigen::VectorXd::Zero(numTreatments);
      Eigen::VectorXd falseYStats = Eigen::VectorXd::Zero(numTreatments);
      Eigen::VectorXd trueYVarStats = Eigen::VectorXd::Zero(numTreatments);
      Eigen::VectorXd falseYVarStats = Eigen::VectorXd::Zero(numTreatments);
    
      //add counts
      for(int nTreat=0; nTreat < numTreatments; nTreat++) {
	trueWCounts(nTreat) = nodeParms(pos + nTreat);
      }
      pos = pos + numTreatments;
      for(int nTreat=0; nTreat < numTreatments; nTreat++) {
	falseWCounts(nTreat) = nodeParms(pos + nTreat);
      }
      pos = pos + numTreatments;
      
      //add y stats
      for(int nTreat=0; nTreat < numTreatments; nTreat++) {
	trueYStats(nTreat) = nodeParms(pos + nTreat);
      }
      pos = pos + numTreatments;
      for(int nTreat=0; nTreat < numTreatments; nTreat++) {
	falseYStats(nTreat) = nodeParms(pos + nTreat);
      }
      pos = pos + numTreatments;

      //add yvar stats
      for(int nTreat=0; nTreat < numTreatments; nTreat++) {
	trueYVarStats(nTreat) = nodeParms(pos + nTreat);
      }
      pos = pos + numTreatments;
      for(int nTreat=0; nTreat < numTreatments; nTreat++) {
	falseYVarStats(nTreat) = nodeParms(pos + nTreat);
      }
      pos = pos + numTreatments;

    
      //create the random test from the vectors
      rt = new RandomTest(hp, 
			  feature, threshold,
			  trueYMean, falseYMean,
			  trueYVar, falseYVar,
			  trueCount, falseCount,
			  trueErr, falseErr,
			  trueWCounts, falseWCounts,
			  trueYStats, falseYStats,
			  trueYVarStats, falseYVarStats,
			  *m_rootYStats, *m_rootCounter);
      
      m_onlineTests.push_back(rt);
    } else {//causal == false 
      rt = new RandomTest(hp, 
			  feature, threshold,
			  trueYMean, falseYMean,
			  trueYVar, falseYVar,
			  trueCount, falseCount,
			  trueErr, falseErr,
			  *m_rootYStats, *m_rootCounter);
      
      m_onlineTests.push_back(rt);
    }
    // check if the best test is the same as this random test.  if so point to it
    if(m_isLeaf == false) {
      if(bt_feat == feature & bt_threshold == threshold) {
	m_bestTest = rt;
      }
    }
  } //close nRandTest loop
} // close method
  

//Version to initialize from a vector of information about the node - below root version
OnlineNode::OnlineNode(const Eigen::VectorXd& nodeParms, const Hyperparameters& hp,
		       int& numNodes, const Eigen::VectorXd& minFeatRange, 
		       const Eigen::VectorXd& maxFeatRange,
		       const Eigen::VectorXd &rootYStats, const double &rootCounter):
  m_numClasses(0), m_hp(&hp),
  m_counter(0.0), 
  m_parentCounter(0.0), 
  m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange),
  m_numNodes(&numNodes),
  m_rootYStats(&rootYStats), 
  m_rootCounter(&rootCounter),
  m_wCounts(Eigen::VectorXd::Zero(hp.numTreatments)),
  m_yStats(Eigen::VectorXd::Zero(hp.numTreatments)),
  m_yVarStats(Eigen::VectorXd::Zero(hp.numTreatments)),
  m_tauHat(Eigen::VectorXd::Zero(hp.numTreatments)),
  m_tauVarHat(Eigen::VectorXd::Zero(hp.numTreatments))
  
{
  int numTreatments = hp.numTreatments; //save this for use in below positions. ==1 for non-causal

  //extract information about the node from the vector
  //common information whether causal or not
  m_nodeNumber = static_cast<int>(nodeParms(0));
  m_parentNodeNumber = static_cast<int>(nodeParms(1));
  m_rightChildNodeNumber = static_cast<int>(nodeParms(2));
  m_leftChildNodeNumber = static_cast<int>(nodeParms(3));
  m_depth = static_cast<int>(nodeParms(4));
  m_isLeaf = static_cast<bool>(nodeParms(5));
  m_counter = static_cast<double>(nodeParms(6));
  m_parentCounter = static_cast<double>(nodeParms(7));
  m_yMean = static_cast<double>(nodeParms(8)); //prediction for the node
  m_yVar = static_cast<double>(nodeParms(9)); //variance estimate for the node
  m_err = static_cast<double>(nodeParms(10)); //prediction for the node

  int pos=11;
  if(m_hp->causal == true) {
    //if causal need to extract tauHat mean and variance, treatment counts, and y statistics
     //mean
    for(int nTreat=0; nTreat < numTreatments; nTreat++) {
      m_tauHat(nTreat) = static_cast<double>(nodeParms(pos+nTreat));
    }
    pos=pos+numTreatments;
 
     //variance
    for(int nTreat=0; nTreat < numTreatments; nTreat++) {
      m_tauVarHat(nTreat) = static_cast<double>(nodeParms(pos+nTreat));
    }
    pos=pos+numTreatments;

    //extract treatment counts
    for(int nTreat=0; nTreat < numTreatments; nTreat++) {
      m_wCounts(nTreat) = static_cast<double>(nodeParms(pos+nTreat));
    }
    pos=pos+numTreatments;

    //extract statistics about y
    for(int nTreat=0; nTreat < numTreatments; nTreat++) {
      m_yStats(nTreat) = static_cast<double>(nodeParms(pos+nTreat));
    }
    pos=pos+numTreatments;

    //extract statistics about yvar
    for(int nTreat=0; nTreat < numTreatments; nTreat++) {
      m_yVarStats(nTreat) = static_cast<double>(nodeParms(pos+nTreat));
    }
    pos=pos+numTreatments;

  } //close causal == true
  
  //get the feature and threshold for the best test to point to later 
  int bt_feat = -1;
  double bt_threshold = 0;
  if(m_isLeaf == false) {
    bt_feat = nodeParms(pos);
    bt_threshold = nodeParms(pos + 1);
  } //close isLeaf
  pos = pos + 2;
  
  //for all nodes (leaf or not) create the randomtests
  
  RandomTest* rt;
  for(int nRandTest=0; nRandTest < m_hp->numRandomTests; nRandTest++) {
    int feature = nodeParms(pos);
    double threshold = nodeParms(pos + 1);
    pos = pos + 2;

    int trueYMean = nodeParms(pos);
    int trueYVar = nodeParms(pos+1);
    int trueCount = static_cast<int>(nodeParms(pos+2));
    int trueErr = nodeParms(pos+3);
    pos = pos + 4;
    
    int falseYMean = nodeParms(pos);
    int falseYVar = nodeParms(pos+1);
    int falseCount = static_cast<int>(nodeParms(pos+2));
    int falseErr = nodeParms(pos+3);
    pos = pos + 4;
    

    if(m_hp->causal == true) {
      //prepare vectors to create the random test
      Eigen::VectorXd trueWCounts = Eigen::VectorXd::Zero(numTreatments);
      Eigen::VectorXd falseWCounts = Eigen::VectorXd::Zero(numTreatments);
      Eigen::VectorXd trueYStats = Eigen::VectorXd::Zero(numTreatments);
      Eigen::VectorXd falseYStats = Eigen::VectorXd::Zero(numTreatments);
      Eigen::VectorXd trueYVarStats = Eigen::VectorXd::Zero(numTreatments);
      Eigen::VectorXd falseYVarStats = Eigen::VectorXd::Zero(numTreatments);
    
      //add counts
      for(int nTreat=0; nTreat < numTreatments; nTreat++) {
	trueWCounts(nTreat) = nodeParms(pos + nTreat);
      }
      pos = pos + numTreatments;
      for(int nTreat=0; nTreat < numTreatments; nTreat++) {
	falseWCounts(nTreat) = nodeParms(pos + nTreat);
      }
      pos = pos + numTreatments;
      
      //add y stats
      for(int nTreat=0; nTreat < numTreatments; nTreat++) {
	trueYStats(nTreat) = nodeParms(pos + nTreat);
      }
      pos = pos + numTreatments;
      for(int nTreat=0; nTreat < numTreatments; nTreat++) {
	falseYStats(nTreat) = nodeParms(pos + nTreat);
      }
      pos = pos + numTreatments;

      //add yvar stats
      for(int nTreat=0; nTreat < numTreatments; nTreat++) {
	trueYVarStats(nTreat) = nodeParms(pos + nTreat);
      }
      pos = pos + numTreatments;
      for(int nTreat=0; nTreat < numTreatments; nTreat++) {
	falseYVarStats(nTreat) = nodeParms(pos + nTreat);
      }
      pos = pos + numTreatments;

    
      //create the random test from the vectors
      rt = new RandomTest(hp, 
			  feature, threshold,
			  trueYMean, falseYMean,
			  trueYVar, falseYVar,
			  trueCount, falseCount,
			  trueErr, falseErr,
			  trueWCounts, falseWCounts,
			  trueYStats, falseYStats,
			  trueYVarStats, falseYVarStats,
			  *m_rootYStats, *m_rootCounter);
      
      m_onlineTests.push_back(rt);
    } else {//causal == false 
      rt = new RandomTest(hp, 
			  feature, threshold,
			  trueYMean, falseYMean,
			  trueYVar, falseYVar,
			  trueCount, falseCount,
			  trueErr, falseErr,
			  *m_rootYStats, *m_rootCounter);
      
      m_onlineTests.push_back(rt);
    }
    // check if the best test is the same as this random test.  if so point to it
    if(m_isLeaf == false) {
      if(bt_feat == feature & bt_threshold == threshold) {
	m_bestTest = rt;
      }
    }
  } //close nRandTest loop
} // close method
  


    
OnlineNode::~OnlineNode() {
  if (m_isLeaf == false) {
    delete m_leftChildNode;
    delete m_rightChildNode;
  }
  for (int nTest = 0; nTest < m_hp->numRandomTests; nTest++) {
    delete m_onlineTests[nTest];
  }
}

//set the child node numbers if needed 
void OnlineNode::setChildNodeNumbers(int rightChildNodeNumber, int leftChildNodeNumber) {
  m_rightChildNodeNumber = rightChildNodeNumber;
  m_leftChildNodeNumber = leftChildNodeNumber;
}
 
/// update node from a sample   
void OnlineNode::update(const Sample& sample) {
  if(m_hp->type=="classification") {
    this->updateClassification(sample);
  } else {
    this->updateRegression(sample);
  }
}

void OnlineNode::updateClassification(const Sample& sample) {
  m_counter += sample.w;
  m_labelStats(sample.yClass) += sample.w;

  //increment treatment and control stats if a causal tree
  if(m_hp->causal == true) {
    if(sample.W) {
      m_treatCounter += sample.w;
      m_treatLabelStats(sample.yClass) += sample.w;
    } else {
      m_controlCounter += sample.w;
      m_controlLabelStats(sample.yClass) += sample.w;
    }

    //update the ITE
    for(int i=0; i < *m_numClasses; ++i) {
      if(m_treatCounter > 0 & m_controlCounter > 0) {
	m_tauHat(i) = (m_treatLabelStats(i)/m_treatCounter) - (m_controlLabelStats(i)/m_controlCounter); 
      } else {
	m_tauHat(i) = 0;
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
	      pair<Eigen::VectorXd, Eigen::VectorXd> parentStats = m_bestTest->getStatsClassification("all");      

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
	      pair<Eigen::VectorXd, Eigen::VectorXd> treatParentStats = m_bestTest->getStatsClassification("treat");      
	      pair<Eigen::VectorXd, Eigen::VectorXd> controlParentStats = m_bestTest->getStatsClassification("control");      

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
} //close updateClassification


void OnlineNode::updateRegression(const Sample& sample) {

  //setting this value here breaking somewhere else
  int counter = m_counter + m_parentCounter;

  m_yMean = (m_yMean * counter + sample.yReg * sample.w) / (counter + sample.w);
  if(counter + sample.w > 1) {
    m_yVar = (m_yVar * (counter-1) + sample.w * pow(m_yMean - sample.yReg,2)) / (counter + sample.w - 1);
  } else {
    m_yVar = sample.w * pow(m_yMean - sample.yReg,2);
  }
  m_counter += sample.w;
  m_err += sample.w * (sample.yReg - m_yMean);


  //variance: Infinitesmal Jacknife (Wager 2014) - resample bootstrap
  //Var_IJ = \Sum_{i=1}^n Cov(N_i, t(x))^2
  // summing for each sample i. 
  // N_i = # times bootstrap included
  // t(x) = prediction for the tree
  // this function called for each try sampled from Pois(1.0)
  // Cov(N_i, t(x)) = (b-1)^{-1} sum((x - mean(x)) * (N - 1))
  //   => increment by (x-mean(x))^2 for each bootstrap - as given above


  //increment treatment and control stats if a causal tree
  if(m_hp->causal == true) {
    //update means for treatment associated with the sample case
    m_yStats(sample.W) = (m_yStats(sample.W) * m_wCounts(sample.W) + 
			  sample.yReg * sample.w) / static_cast<double>(m_wCounts(sample.W) + sample.w);
      
    //update variances for treatment associated with the sample case
    if(m_wCounts(sample.W) + sample.w > 1) {
      m_yVarStats(sample.W) = (m_yVarStats(sample.W) * (m_wCounts(sample.W)-1) + 
			       pow(m_yStats(sample.W) - sample.yReg,2) * sample.w) /
	(m_wCounts(sample.W) + sample.w - 1);
    } else {
      m_yVarStats(sample.W) = pow(m_yStats(sample.W) - sample.yReg,2) * sample.w;
    }
    //increment count
    m_wCounts(sample.W) += sample.w;

    

    //update the Treatment Effect estimates (note - update all just in case needed for a comparison)
    for(int nTreat=0; nTreat < m_hp->numTreatments; nTreat++) {
      m_tauHat(nTreat) = m_yStats(nTreat) - m_yStats(0);
      
      //calculate tau variance estimate as a pooled estimate between treatment and control
      m_tauVarHat(nTreat) = 0;
      int denom=0;
      if(m_wCounts(nTreat) > 1) {
	m_tauVarHat(nTreat) = m_yVarStats(nTreat) * (1.0 / (m_wCounts(nTreat)-1));
	denom += 1.0 / (m_wCounts(nTreat)-1);
      }
      if(m_wCounts(0) > 1) {
	m_tauVarHat(nTreat) = m_yVarStats(0) * (1.0 / (m_wCounts(0)-1));
	denom += 1.0 / (m_wCounts(0)-1);
      }
      if(denom > 0) {
	m_tauVarHat(nTreat) = m_tauVarHat(nTreat) / denom;
      } else {
	m_tauVarHat(nTreat) = 0;
      }
    }
  } //close causal == TRUE

  if (m_isLeaf == true) {
    // Update online tests
    for (vector<RandomTest*>::iterator itr = m_onlineTests.begin(); 
	 itr != m_onlineTests.end(); ++itr) {
      (*itr)->update(sample);
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
      
      //Figure out the next available nodeNumber - then increment the one for the tree
      int newNodeNumber = *m_numNodes;
      
      // Split - initializing with versions beyond the root node
      pair<int, int> parentTotCounts=m_bestTest->getTotCounts();
      pair<double, double> parentYMeans=m_bestTest->getYMeans();
      pair<double, double> parentYVars=m_bestTest->getYVars();
      pair<double, double> parentErrs=m_bestTest->getErrs();
      if(m_hp->causal == false) {
	m_rightChildNode = new OnlineNode(*m_hp,
					  m_minFeatRange->rows(), *m_minFeatRange, 
					  *m_maxFeatRange, m_depth + 1, 
					  parentTotCounts.first, 
					  parentYMeans.first, 
					  parentYVars.first, 
					  parentErrs.first, 
					  newNodeNumber, 
					  m_nodeNumber, *m_numNodes,
					  *m_rootYStats, *m_rootCounter
					  );
	m_leftChildNode = new OnlineNode(*m_hp, m_minFeatRange->rows(),
					 *m_minFeatRange, *m_maxFeatRange, m_depth + 1,
					 parentTotCounts.second, 
					 parentYMeans.second, 
					 parentYVars.second, 
					 parentErrs.second, 
					 newNodeNumber + 1, 
					 m_nodeNumber, *m_numNodes,
					  *m_rootYStats, *m_rootCounter
					 );
      } else { // causal==true
	pair<Eigen::VectorXd, Eigen::VectorXd> parentWCounts = m_bestTest->getWCounts();
	pair<Eigen::VectorXd, Eigen::VectorXd> parentYStats = m_bestTest->getYStats();
	pair<Eigen::VectorXd, Eigen::VectorXd> parentYVarStats = m_bestTest->getYVarStats();
	
	m_rightChildNode = new OnlineNode(*m_hp, m_minFeatRange->rows(), 
					  *m_minFeatRange, *m_maxFeatRange, 
					  m_depth + 1, 
					  parentTotCounts.first, 
					  parentYMeans.first, 
					  parentYVars.first, 
					  parentErrs.first, 
					  parentWCounts.first,
					  parentYStats.first,
					  parentYVarStats.first,
					  newNodeNumber, 
					  m_nodeNumber, *m_numNodes,
					  *m_rootYStats, *m_rootCounter);
	m_leftChildNode = new OnlineNode(*m_hp, m_minFeatRange->rows(),
					 *m_minFeatRange, *m_maxFeatRange, 
					 m_depth + 1,
					 parentTotCounts.second, 
					 parentYMeans.second, 
					 parentYVars.second, 
					 parentErrs.second, 
					 parentWCounts.second,
					 parentYStats.second,
					 parentYVarStats.second,
					 newNodeNumber + 1, 
					 m_nodeNumber, *m_numNodes,
					  *m_rootYStats, *m_rootCounter);
      }
      
      //set the child node numbers now that nodes have been created
      setChildNodeNumbers(newNodeNumber, newNodeNumber + 1);
    } //close isLeaf==true
  } else {
    if (m_bestTest->eval(sample)) {
      m_rightChildNode->update(sample);
    } else {
      m_leftChildNode->update(sample);
    }
  }
}


void OnlineNode::eval(const Sample& sample, Result& result) {
  if(m_hp->type=="classification") {
    this->evalClassification(sample, result);
  } else {
    this->evalRegression(sample, result);
  }
}

void OnlineNode::evalClassification(const Sample& sample, Result& result) {
  if (m_isLeaf == true) {
    if (m_counter + m_parentCounter) {
      result.confidence = m_labelStats / static_cast<double>(m_counter + m_parentCounter);
      result.predictionClassification = m_label;
      if(m_hp->causal == true) {
	result.tauHat = m_tauHat;
      } else {
	result.tauHat = Eigen::VectorXd::Zero(*m_numClasses);
      }
    } else {
      result.confidence = Eigen::VectorXd::Constant(m_labelStats.rows(), 1.0 / *m_numClasses);
      result.predictionClassification = 0;
      result.tauHat = Eigen::VectorXd::Zero(*m_numClasses);
    }
  } else {
    if (m_bestTest->eval(sample)) {
      m_rightChildNode->eval(sample, result);
    } else {
      m_leftChildNode->eval(sample, result);
    }
  }
}

void OnlineNode::evalRegression(const Sample& sample, Result& result) {
  if (m_isLeaf == true) {
    if (m_counter + m_parentCounter > 0) {
      result.predictionVarianceRegression = m_yVar;
      result.predictionRegression = m_yMean;
      //result.weight = m_counter+m_parentCounter;
      if(m_hp->causal == true) {
	result.tauHat = m_tauHat;
      }
    } else {
      result.predictionVarianceRegression = 1.0;
      result.predictionRegression = 0;
      //result.weight = 0;
      result.tauHat = Eigen::VectorXd::Zero(m_hp->numTreatments);
    }
  } else { //if not a leaf - recurse
    if (m_bestTest->eval(sample)) {
      //first check if the child node has any counts - if not, use this node
      double count = m_rightChildNode->getCount();
      if(count > 0) {
	m_rightChildNode->eval(sample, result);
      } else { //if no values have flowed to the child node yet, use the value from this node
	//does the right side have any?
	pair<int, int> childCounts=m_bestTest->getTotCounts();
	if(childCounts.first == 0) {
	  //child has seen no data, use the parent
	  result.predictionVarianceRegression = m_yVar;
	  result.predictionRegression = m_yMean;
	  //result.weight = m_counter+m_parentCounter;
	  if(m_hp->causal == true) {
	    result.tauHat = m_tauHat;
	  }
	} else { //child has seen some data, use from best test

	  pair<double, double> childYMeans=m_bestTest->getYMeans();
	  pair<double, double> childYVars=m_bestTest->getYVars();

	  result.predictionVarianceRegression = childYVars.first;
	  result.predictionRegression = childYMeans.first;
	  //result.weight = childCounts.first;

	  if(m_hp->causal == true) {
	    pair<Eigen::VectorXd, Eigen::VectorXd> childWCounts = m_bestTest->getWCounts();
	    pair<Eigen::VectorXd, Eigen::VectorXd> childYStats = m_bestTest->getYStats();
	    pair<Eigen::VectorXd, Eigen::VectorXd> childYVarStats = m_bestTest->getYVarStats();	    
	    for(int nTreat=0; nTreat < m_hp->numTreatments; nTreat++) {
	      result.tauHat(nTreat) = childYStats.first(nTreat) - childYStats.first(0);
	      //calculate tau variance estimate as a pooled estimate between treatment and control
	      result.tauVarHat(nTreat) = 0;
	      int denom=0;
	      if(childWCounts.first(nTreat) > 1) {
		result.tauVarHat(nTreat) = childYVarStats.first(nTreat) * (1.0 / (childWCounts.first(nTreat)-1));
		denom += 1.0 / (childWCounts.first(nTreat)-1);
	      }
	      if(childWCounts.first(0) > 1) {
		result.tauVarHat(nTreat) = childYVarStats.first(0) * (1.0 / (childWCounts.first(0)-1));
		denom += 1.0 / (childWCounts.first(0)-1);
	      }
	      if(denom > 0) {
		result.tauVarHat(nTreat) = result.tauVarHat(nTreat) / denom;
	      } else {
		result.tauVarHat(nTreat) = 0;
	      }
	    } //loop nTreat
	  } //close causal==true
	} // close child has data
      } //close else count > 0
    } else {
      double count = m_leftChildNode->getCount();
      if(count > 0) {
	m_leftChildNode->eval(sample, result);
      } else { //if no values have flowed to the child node yet, use the value from this node
	//does the right side have any?
	pair<int, int> childCounts=m_bestTest->getTotCounts();
	if(childCounts.second == 0) {
	  //child has seen no data, use the parent
	  result.predictionVarianceRegression = m_yVar;
	  result.predictionRegression = m_yMean;
	  //result.weight = m_counter+m_parentCounter;
	  if(m_hp->causal == true) {
	    result.tauHat = m_tauHat;
	  }
	} else { //child has seen some data, use from best test	  
	  pair<double, double> childYMeans=m_bestTest->getYMeans();
	  pair<double, double> childYVars=m_bestTest->getYVars();
	  
	  result.predictionVarianceRegression = childYVars.second;
	  result.predictionRegression = childYMeans.second;
	  //result.weight = childCounts.second;

	  if(m_hp->causal == true) {
	    pair<Eigen::VectorXd, Eigen::VectorXd> childWCounts = m_bestTest->getWCounts();
	    pair<Eigen::VectorXd, Eigen::VectorXd> childYStats = m_bestTest->getYStats();
	    pair<Eigen::VectorXd, Eigen::VectorXd> childYVarStats = m_bestTest->getYVarStats();
	    for(int nTreat=0; nTreat < m_hp->numTreatments; nTreat++) {
	      result.tauHat(nTreat) = childYStats.second(nTreat) - childYStats.second(0);
	      //calculate tau variance estimate as a pooled estimate between treatment and control
	      result.tauVarHat(nTreat) = 0;
	      int denom=0;
	      if(childWCounts.second(nTreat) > 1) {
		result.tauVarHat(nTreat) = childYVarStats.second(nTreat) * (1.0 / (childWCounts.second(nTreat)-1));
		denom += 1.0 / (childWCounts.second(nTreat)-1);
	      }
	      if(childWCounts.second(0) > 1) {
		result.tauVarHat(nTreat) = childYVarStats.second(0) * (1.0 / (childWCounts.second(0)-1));
		denom += 1.0 / (childWCounts.second(0)-1);
	      }
	      if(denom > 0) {
		result.tauVarHat(nTreat) = result.tauVarHat(nTreat) / denom;
	      } else {
		result.tauVarHat(nTreat) = 0;
	      }
	    } //loop nTreat
	  } //close causal==true
	} // close child has data
      } //close else count > 0
    }
  } //close not a leaf
}

//version of update to grow from a set of parameters
void OnlineNode::update(const Eigen::MatrixXd& treeParms) {
  if(m_hp->type=="classification") {
    this->updateClassification(treeParms);
  } else {
    this->updateRegression(treeParms);
  }
}

void OnlineNode::updateClassification(const Eigen::MatrixXd& treeParms) {
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

void OnlineNode::updateRegression(const Eigen::MatrixXd& treeParms) {
  if(m_isLeaf == false) { // if its a leaf then theres no splitting to do
    //search through matrix of parms to find the correct rows and make node parms
    int found=0;
    for(int i=0; i < treeParms.rows(); ++i) {
      Eigen::VectorXd nodeParmsVec = treeParms.row(i);
      int npv_nodeNumber = static_cast<int>(nodeParmsVec(0));
      if(npv_nodeNumber == m_rightChildNodeNumber) {
	m_rightChildNode = new OnlineNode(nodeParmsVec, *m_hp, *m_numNodes,
					  *m_minFeatRange, *m_maxFeatRange,
					  *m_rootYStats, *m_rootCounter);
	found++;
      } else if(npv_nodeNumber == m_leftChildNodeNumber) {
	m_leftChildNode = new OnlineNode(nodeParmsVec, *m_hp, *m_numNodes,
					 *m_minFeatRange, *m_maxFeatRange,
					  *m_rootYStats, *m_rootCounter);
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
  bool ret;
  if(m_hp->type=="classification") {
    ret = this->shouldISplitClassification();
  } else {
    ret = this->shouldISplitRegression();
  }
  return(ret);
}

bool OnlineNode::shouldISplitClassification() const {
  //check if all obs in the node have the same value
  bool isPure = false;
  for (int nClass = 0; nClass < *m_numClasses; ++nClass) {
    if (m_labelStats(nClass) == m_counter + m_parentCounter) {
      isPure = true;
      break;
    }
  }
  if(isPure == false) {
    if(m_hp->causal == true) {
      //check if all obs in treatment are pure and all in control are pure
      // can result in pure==true when treatment pure and control pure but not equal
      bool isTreatPure = false;
      bool isControlPure = false;
      for (int nClass = 0; nClass < *m_numClasses; ++nClass) {
	if (m_treatLabelStats(nClass) == m_treatCounter) {
	  isTreatPure = true;
	  break;
	}
      }
      for (int nClass = 0; nClass < *m_numClasses; ++nClass) {
	if (m_controlLabelStats(nClass) == m_treatCounter) {
	  isControlPure = true;
	  break;
	}      
      }    
      if(isTreatPure & isControlPure) {
	isPure = true;
      }
    } //close causal == true
  } //close isPure==false
  //only split if not pure & depth < max depth & counter > counterthreshould
  if ((isPure) || (m_depth >= m_hp->maxDepth) || (m_counter < m_hp->counterThreshold)) {
    return false;
  } else {
    return true;
  }
}

bool OnlineNode::shouldISplitRegression() const {
  //check if all obs in the node have the same value
  //only split if not pure & depth < max depth & counter > counterthreshould
  if ((m_depth >= m_hp->maxDepth) || (m_counter < m_hp->counterThreshold)) {
    return false;
  } else {
    return true;
  }
}


/// Methods to export parameters to vector for saving
Eigen::VectorXd OnlineNode::exportParms() { 
  Eigen::VectorXd out;
  if(m_hp->type == "classification") {
    if(m_hp->causal == false) {
      out=exportParmsClassification();
    } else {
      out=exportParmsClassificationCausal();      
    } 
  } else {
    if(m_hp->causal == false) {
      out=exportParmsRegression();
    } else {
      out=exportParmsRegressionCausal();      
    } 
  }
  return(out);
}

///// Classification Tree Methods
Eigen::VectorXd OnlineNode::exportParmsClassification() {
  //create vector to export
  
  //see layout spreadsheet for accounting of length
  int vec_size;
  vec_size = 13 + 3 * *m_numClasses + 2 * m_hp->numRandomTests * (1 + *m_numClasses);
 
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
    bt_stats = m_bestTest->getStatsClassification();
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
    pair<Eigen::VectorXd, Eigen::VectorXd> rt_stats = rt.getStatsClassification();
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
  return(nodeParms);
} //close exportParmsClassification

Eigen::VectorXd OnlineNode::exportParmsClassificationCausal() {
  //create vector to export  
  //see layout spreadsheet for accounting of length
  int vec_size;
  vec_size = 15 + 8 * *m_numClasses + 2 * m_hp->numRandomTests * (1 + 2 * *m_numClasses);
  
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
  
  //put in the ite estimates
  pos=8;
  for(int l=0; l < *m_numClasses; ++l) {
    nodeParms(pos + l) = m_tauHat(l);
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
  if(m_isLeaf == false) { //if NOT a leaf then we have a best test
    bt_parms = m_bestTest->getParms();
    bt_treatStats = m_bestTest->getStatsClassification("treat");
    bt_controlStats = m_bestTest->getStatsClassification("control");
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
    pair<Eigen::VectorXd, Eigen::VectorXd> rt_treatStats = rt.getStatsClassification("treat");
    pair<Eigen::VectorXd, Eigen::VectorXd> rt_controlStats = rt.getStatsClassification("control");
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
  
  return(nodeParms);
} //close exportParmsClassificationCausal



Eigen::VectorXd OnlineNode::exportParmsRegression() {
  //create vector to export
  
  int vec_size;
  vec_size = 13 + 10 * m_hp->numRandomTests;
 
  Eigen::VectorXd nodeParms = Eigen::VectorXd::Zero(vec_size);  //initialize the vector with zeros
  
  //fetch the private parameters and save into the Node parms object
  int pos = 0;
  nodeParms(0) = static_cast<double>(m_nodeNumber);
  nodeParms(1) = static_cast<double>(m_parentNodeNumber);
  nodeParms(2) = static_cast<double>(m_rightChildNodeNumber);
  nodeParms(3) = static_cast<double>(m_leftChildNodeNumber);
  nodeParms(4) = static_cast<double>(m_depth);
  nodeParms(5) = static_cast<double>(m_isLeaf);
  nodeParms(6) = static_cast<double>(m_counter);
  nodeParms(7) = static_cast<double>(m_parentCounter);
  nodeParms(8) = static_cast<double>(m_yMean);
  nodeParms(9) = static_cast<double>(m_yVar);
  nodeParms(10) = static_cast<double>(m_err);

  pos = 11;

  //save best test information
  int bt_feature=-1;
  double bt_threshold=0;
  
  if(m_isLeaf == false) { //if NOT a leaf then we dont have a best test but do have randomtests
    pair<int, double> bt_parms;
    //    pair<double, double> bt_yMeans;
    //    pair<int, int> bt_counts;
    //    pair<double, double> bt_err;

    bt_parms = m_bestTest->getParms();
    bt_feature=bt_parms.first;
    bt_threshold=bt_parms.second;
  }
  //write bt information to the vector
  nodeParms(pos) = bt_feature;
  nodeParms(pos + 1) = bt_threshold;
  pos = pos + 2;

  //copy the information for each random test
  for(int nRandTest=0; nRandTest <  m_hp->numRandomTests; nRandTest++) {
    
    RandomTest rt = *m_onlineTests[nRandTest];
    pair<int, double> rt_parms = rt.getParms();
    int rtFeature = rt_parms.first;
    double rtThreshold = rt_parms.second;

    //feature & threshold
    nodeParms(pos) = static_cast<int>(rtFeature);
    nodeParms(pos + 1) = static_cast<double>(rtThreshold);

    pos = pos + 2;

    //statistics for the random test
    pair<double, double> rt_yMeans;
    pair<double, double> rt_yVars;
    pair<int, int> rt_counts;
    pair<double, double> rt_err;

    double rt_trueYMean=0;
    double rt_falseYMean=0;
    double rt_trueYVar=0;
    double rt_falseYVar=0;
    int rt_trueCount=0;
    int rt_falseCount=0;
    double rt_trueErr=0;
    double rt_falseErr=0;

    rt_yMeans = rt.getYMeans();
    rt_yVars = rt.getYVars();
    rt_counts = rt.getTotCounts();
    rt_err = rt.getErrs();

    rt_trueYMean=rt_yMeans.first;
    rt_falseYMean=rt_yMeans.second;
    rt_trueYVar=rt_yVars.first;
    rt_falseYVar=rt_yVars.second;
    rt_trueCount=rt_counts.first;
    rt_falseCount=rt_counts.second;
    rt_trueErr=rt_err.first;
    rt_falseErr=rt_err.second;

    //copy in the true and false stats
    nodeParms(pos) = static_cast<double>(rt_trueYMean);
    nodeParms(pos+1) = static_cast<double>(rt_trueYVar);
    nodeParms(pos+2) = static_cast<int>(rt_trueCount);
    nodeParms(pos+3) = static_cast<double>(rt_trueErr);
    
    pos = pos + 4;

    nodeParms(pos) = static_cast<double>(rt_falseYMean);
    nodeParms(pos+1) = static_cast<double>(rt_falseYVar);
    nodeParms(pos+2) = static_cast<int>(rt_falseCount);
    nodeParms(pos+3) = static_cast<double>(rt_falseErr);
    
    pos = pos + 4;
  } //loop nRandTest
  return(nodeParms);
}

Eigen::VectorXd OnlineNode::exportParmsRegressionCausal() {
  //create vector to export
  int numTreatments = m_hp->numTreatments; //save this for use in below positions. ==1 for non-causal
  
  int vec_size;
  vec_size = 13 + 10 * m_hp->numRandomTests + 5 * numTreatments + 6 * numTreatments * m_hp->numRandomTests;
 
  Eigen::VectorXd nodeParms = Eigen::VectorXd::Zero(vec_size);  //initialize the vector with zeros
  
  //fetch the private parameters and save into the Node parms object
  int pos = 0;
  nodeParms(0) = static_cast<double>(m_nodeNumber);
  nodeParms(1) = static_cast<double>(m_parentNodeNumber);
  nodeParms(2) = static_cast<double>(m_rightChildNodeNumber);
  nodeParms(3) = static_cast<double>(m_leftChildNodeNumber);
  nodeParms(4) = static_cast<double>(m_depth);
  nodeParms(5) = static_cast<double>(m_isLeaf);
  nodeParms(6) = static_cast<double>(m_counter);
  nodeParms(7) = static_cast<double>(m_parentCounter);
  nodeParms(8) = static_cast<double>(m_yMean);
  nodeParms(9) = static_cast<double>(m_yVar);
  nodeParms(10) = static_cast<double>(m_err);

  pos = 11;

  //add tau hats
  for(int nTreat=0; nTreat < numTreatments; nTreat++) {
    nodeParms(pos+nTreat) = static_cast<double>(m_tauHat(nTreat));
  }
  pos=pos+numTreatments;

  //add tau variance hat
  for(int nTreat=0; nTreat < numTreatments; nTreat++) {
    nodeParms(pos+nTreat) = static_cast<double>(m_tauVarHat(nTreat));
  }
  pos=pos+numTreatments;
 
  //add wCounts
  for(int nTreat=0; nTreat < numTreatments; nTreat++) {
    nodeParms(pos+nTreat) = static_cast<double>(m_wCounts(nTreat));
  }
  pos=pos+numTreatments;
 
  //add y means variance
  for(int nTreat=0; nTreat < numTreatments; nTreat++) {
    nodeParms(pos+nTreat) = static_cast<double>(m_yStats(nTreat));
  }
  pos=pos+numTreatments;

  //add y means variance
  for(int nTreat=0; nTreat < numTreatments; nTreat++) {
    nodeParms(pos+nTreat) = static_cast<double>(m_yVarStats(nTreat));
  }
  pos=pos+numTreatments;
 
  //save best test information
  int bt_feature=-1;
  double bt_threshold=0;
  
  if(m_isLeaf == false) { //if NOT a leaf then we dont have a best test but do have randomtests
    pair<int, double> bt_parms;
//     pair<double, double> bt_yMeans;
//     pair<int, int> bt_counts;
//     pair<double, double> bt_err;

    bt_parms = m_bestTest->getParms();
    bt_feature=bt_parms.first;
    bt_threshold=bt_parms.second;
  }
  //write bt information to the vector
  nodeParms(pos) = bt_feature;
  nodeParms(pos + 1) = bt_threshold;
  pos = pos + 2;

  //copy the information for each random test
  for(int nRandTest=0; nRandTest <  m_hp->numRandomTests; nRandTest++) {
    
    RandomTest rt = *m_onlineTests[nRandTest];
    pair<int, double> rt_parms = rt.getParms();
    int rtFeature = rt_parms.first;
    double rtThreshold = rt_parms.second;

    //feature & threshold
    nodeParms(pos) = static_cast<int>(rtFeature);
    nodeParms(pos + 1) = static_cast<double>(rtThreshold);

    pos = pos + 2;

    //statistics for the random test
    pair<double, double> rt_yMeans;
    pair<double, double> rt_yVars;
    pair<int, int> rt_counts;
    pair<double, double> rt_err;

    double rt_trueYMean=0;
    double rt_falseYMean=0;
    double rt_trueYVar=0;
    double rt_falseYVar=0;
    int rt_trueCount=0;
    int rt_falseCount=0;
    double rt_trueErr=0;
    double rt_falseErr=0;

    rt_yMeans = rt.getYMeans();
    rt_yVars = rt.getYVars();
    rt_counts = rt.getTotCounts();
    rt_err = rt.getErrs();

    rt_trueYMean=rt_yMeans.first;
    rt_falseYMean=rt_yMeans.second;
    rt_trueYVar=rt_yVars.first;
    rt_falseYVar=rt_yVars.second;
    rt_trueCount=rt_counts.first;
    rt_falseCount=rt_counts.second;
    rt_trueErr=rt_err.first;
    rt_falseErr=rt_err.second;

    //copy in the true and false stats
    nodeParms(pos) = static_cast<double>(rt_trueYMean);
    nodeParms(pos+1) = static_cast<double>(rt_trueYVar);
    nodeParms(pos+2) = static_cast<int>(rt_trueCount);
    nodeParms(pos+3) = static_cast<int>(rt_trueErr);
    
    pos = pos + 4;

    nodeParms(pos) = static_cast<double>(rt_falseYMean);
    nodeParms(pos+1) = static_cast<double>(rt_falseYVar);
    nodeParms(pos+2) = static_cast<int>(rt_falseCount);
    nodeParms(pos+3) = static_cast<int>(rt_falseErr);
    
    pos = pos + 4;

    //save additional parameters needed for causal trees
    pair<Eigen::VectorXd, Eigen::VectorXd> rt_yStats=rt.getYStats();
    pair<Eigen::VectorXd, Eigen::VectorXd> rt_yVarStats=rt.getYVarStats();
    pair<Eigen::VectorXd, Eigen::VectorXd> rt_wCounts=rt.getWCounts();

    Eigen::VectorXd rtTrueYStats = rt_yStats.first;
    Eigen::VectorXd rtFalseYStats = rt_yStats.second;
    Eigen::VectorXd rtTrueYVarStats = rt_yVarStats.first;
    Eigen::VectorXd rtFalseYVarStats = rt_yVarStats.second;
    Eigen::VectorXd rtTrueWCounts = rt_wCounts.first;
    Eigen::VectorXd rtFalseWCounts = rt_wCounts.second;

    //add counts
    for(int nTreat=0; nTreat < numTreatments; nTreat++) {
      nodeParms(pos+nTreat) = static_cast<double>(rtTrueWCounts(nTreat));
    }
    pos = pos + numTreatments;

    for(int nTreat=0; nTreat < numTreatments; nTreat++) {
      nodeParms(pos+nTreat) = static_cast<double>(rtFalseWCounts(nTreat));
    }
    pos = pos + numTreatments;
      
    //add y stats
    for(int nTreat=0; nTreat < numTreatments; nTreat++) {
      nodeParms(pos+nTreat) = static_cast<double>(rtTrueYStats(nTreat));
    }
    pos = pos + numTreatments;
    for(int nTreat=0; nTreat < numTreatments; nTreat++) {
      nodeParms(pos+nTreat) = static_cast<double>(rtFalseYStats(nTreat));
    }
    pos = pos + numTreatments;

    //add y var stats
    for(int nTreat=0; nTreat < numTreatments; nTreat++) {
      nodeParms(pos+nTreat) = static_cast<double>(rtTrueYVarStats(nTreat));
    }
    pos = pos + numTreatments;
    for(int nTreat=0; nTreat < numTreatments; nTreat++) {
      nodeParms(pos+nTreat) = static_cast<double>(rtFalseYVarStats(nTreat));
    }
    pos = pos + numTreatments;

  } //loop nRandTest
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
double OnlineNode::scoreClassification() {
  Eigen::VectorXd stats=m_labelStats;
  double count = m_counter + m_parentCounter; 
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

double OnlineNode::scoreRegression() {
  double score = 0;
  if(m_hp->causal == true) {
    //causal: difference in mse from treatment to control
    if(m_hp->method == "mse") {
      int cnt = 0;
      for(int nTreat = 0; nTreat < m_hp->numTreatments; nTreat++) {
	//if method = mse - total difference from each group to control
	if(nTreat > 0) {
	  score += m_wCounts(nTreat) * pow(m_yStats(nTreat) - m_yStats(0),2);
	  cnt += m_wCounts(nTreat);
	}
      }
      if(cnt > 0) {
	score = score / static_cast<double>(cnt);
      } 
    } else if(m_hp->type=="hellinger") {
      double p=0, p_root=0;
      Eigen::VectorXd rootYStats = *m_rootYStats;
      for(int nTreat = 0; nTreat < m_hp->numTreatments; nTreat++) {
	if(*m_rootCounter > 0) {
	  p_root = rootYStats(nTreat) /  static_cast<double>(*m_rootCounter);
	}
	if(m_counter + m_parentCounter> 0) {
	  p = m_yStats(nTreat) /  static_cast<double>(m_counter);
	}
	score += pow(sqrt(p) - sqrt(p_root),2);
      }//loop nTreat
      score = sqrt(score);
    } //close hellinger 
  } else { //non causal - just calculate mse
    if(m_counter + m_parentCounter > 0) {
      score = pow(m_err,2) / static_cast<double>(m_counter); 
    }
  } //close non-causal
  return(score);
}

double OnlineNode::score() {
  double out;
  if(m_hp->type=="classification") {
    out = this->scoreClassification();
  } else {
    out = this->scoreRegression();
  }
  return(out);
}



// void OnlineNode::printInfo() {
//   cout << "Node Information about Node " << m_nodeNumber << std::endl;
//   cout << "\tisLeaf: " << m_isLeaf << ", rightChildNodeNumber: " << m_rightChildNodeNumber << ", leftChildNodeNumber: " << m_leftChildNodeNumber << std::endl;
// }

// void OnlineNode::print() {
//   cout << "Node details: " << m_nodeNumber << std::endl;
//   cout << exportParms() << std::endl;
// }


/****************************************************************************************
 *
 *  ONLINE TREE CONSTRUCTORS AND METHODS 
 *
 ******************************************************************************************/

////// Classification Tree Constructors
//version to construct with randomization
OnlineTree::OnlineTree(const Hyperparameters& hp, const int& numClasses, 
		       const int& numFeatures, 
                       const Eigen::VectorXd& minFeatRange, const Eigen::VectorXd& maxFeatRange) :
  m_numClasses(&numClasses), m_hp(&hp), 
  m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange) {
  
  //initialize here - will get updated later in the program during update
  m_oobe = 0.0;
  m_counter = 0.0;
  m_numNodes = 0;
  //initialize with root node version
  m_rootNode = new OnlineNode(hp, *m_numClasses, numFeatures, minFeatRange, maxFeatRange, 
  			      0, m_numNodes);
    
  m_name = "OnlineTree";
}

//version to construct from a set of parameters
OnlineTree::OnlineTree(const Eigen::MatrixXd& treeParms, const Hyperparameters& hp, 
		       const int& numClasses, double oobe, double counter,
		       const Eigen::VectorXd& minFeatRange, const Eigen::VectorXd& maxFeatRange) :
  m_numNodes(0),
  m_oobe(oobe), m_counter(counter), 
  m_numClasses(&numClasses), 
  m_hp(&hp),
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

/////// Regression Tree Constructors
//version to construct with randomization
OnlineTree::OnlineTree(const Hyperparameters& hp, const int& numFeatures, 
                       const Eigen::VectorXd& minFeatRange, 
		       const Eigen::VectorXd& maxFeatRange) :
  m_numClasses(0), m_hp(&hp), m_minFeatRange(&minFeatRange), 
  m_maxFeatRange(&maxFeatRange) {
  
  //initialize here - will get updated later in the program during update
  m_oobe = 0.0;
  m_counter = 0.0;
  m_numNodes = 0;
  //initialize with root node version
  m_rootNode = new OnlineNode(hp, numFeatures, 
			      minFeatRange, maxFeatRange, 
  			      0, m_numNodes);
  m_name = "OnlineTree";
}

//version to construct from a set of parameters
OnlineTree::OnlineTree(const Eigen::MatrixXd& treeParms, const Hyperparameters& hp, 
		       double oobe, double counter,
		       const Eigen::VectorXd& minFeatRange, const Eigen::VectorXd& maxFeatRange) :
  m_numNodes(0), m_oobe(oobe), 
  m_counter(counter), m_numClasses(0), 
  m_hp(&hp),
  m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange) {

  //find the max node number from the treeParms matrix - position 0
  m_numNodes = treeParms.rows();
  
  //initialize with the version that takes parameters
  m_rootNode = new OnlineNode(treeParms.row(0), hp, m_numNodes, 
  			      minFeatRange, maxFeatRange);

  //grow the tree based on matrix of parameters - recursive
  m_rootNode->update(treeParms);

  m_name = "OnlineTree";
  
}


OnlineTree::~OnlineTree() {
    delete m_rootNode;
}
    
void OnlineTree::update(Sample& sample) {
  if(m_hp->type=="classification") {
    this->updateClassification(sample);
  } else {
    this->updateRegression(sample);
  }
}

void OnlineTree::updateClassification(Sample& sample) {
  //increment counter for obs passing through
  m_counter += sample.w;

  //make a prediction about this obs before update
  Result treeResult;
  eval(sample, treeResult);
  
  if (treeResult.predictionClassification != sample.yClass) {
    m_oobe += sample.w;
  }
  
  //update tree parms using this obs
  m_rootNode->update(sample);
}

void OnlineTree::updateRegression(Sample& sample) {
  //increment counter for obs passing through
  m_counter += sample.w;

  //make a prediction about this obs before update
  Result treeResult;
  eval(sample, treeResult);
  
  m_oobe += sample.w * pow(sample.yReg - treeResult.predictionRegression,2);  
  
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

// void OnlineTree::printInfo() {
//   cout << "Tree Info: ";
//   cout << "m_numNodes: " << m_numNodes << std::endl;
// }

// void OnlineTree::print() {
//   cout << "Tree details: " << std::endl;
//   vector<Eigen::MatrixXd> treeParms = exportParms();
//   if(treeParms.size() > 0) {
//     cout << treeParms[0] << std::endl;
//   }
// }

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

///// Classification Forest Constructors
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
  m_numClasses(&numClasses), m_hp(&hp), 
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

///// Regression Forest Constructors
//version to construct using randomization
OnlineRF::OnlineRF(const Hyperparameters& hp, const int& numFeatures,
		   Eigen::VectorXd minFeatRange, Eigen::VectorXd maxFeatRange) :
  m_counter(0.0), m_oobe(0.0), m_numClasses(0), 
  m_hp(&hp), m_minFeatRange(minFeatRange), m_maxFeatRange(maxFeatRange) {
  OnlineTree *tree;
  for (int nTree = 0; nTree < hp.numTrees; ++nTree) {
    tree = new OnlineTree(hp, numFeatures, m_minFeatRange, m_maxFeatRange);
    m_trees.push_back(tree);
  }
  m_name = "OnlineRF";
}

//version to construction from a set of parameters
OnlineRF::OnlineRF(const vector<Eigen::MatrixXd> orfParms, const Hyperparameters& hp,
		   double oobe, double counter,
		   Eigen::VectorXd minFeatRange, Eigen::VectorXd maxFeatRange) :
  m_counter(counter), m_oobe(oobe),
  m_numClasses(0),  
  m_hp(&hp), 
  m_minFeatRange(minFeatRange), m_maxFeatRange(maxFeatRange) {
  OnlineTree *tree;
  for (int nTree = 0; nTree < m_hp->numTrees; ++nTree) {
    //create the trees using method to construct from parameters
    //initializing oobe and counter to 0 until i can figure that out
    tree = new OnlineTree(orfParms[nTree], hp, 0, 0.0, 
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
  if(m_hp->type=="classification") {
    this->updateClassification(sample);
  } else {
    this->updateRegression(sample);
  }
}

void OnlineRF::updateClassification(Sample& sample) {
  m_counter += sample.w;
  Result result(*m_numClasses), treeResult;
  int numTries;
  for (int nTree = 0; nTree < m_hp->numTrees; ++nTree) {
    numTries = poisson(1.0);
    if (numTries > 0) {
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
  if (pre != sample.yClass) {
    m_oobe += sample.w;
  }
}

void OnlineRF::updateRegression(Sample& sample) {
  m_counter += sample.w;
  Result result, treeResult;
  result.predictionRegression = 0;

  int numTries;
  int counter = 0;
  //loop through all trees - boosting to update trees with data
  for (int nTree = 0; nTree < m_hp->numTrees; ++nTree) {
    numTries = poisson(1.0);
    if (numTries > 0) {
      for (int nTry = 0; nTry < numTries; ++nTry) {
	m_trees[nTree]->update(sample);	
      }
    } else {
      //if no training with this sample
      //out of bag estimation of error (variance and tau not needed)
      m_trees[nTree]->eval(sample, treeResult);
      result.predictionRegression += treeResult.predictionRegression;
      counter++;
    }
  }
  if(counter > 0) {
    m_oobe += sample.w * pow(sample.yReg - result.predictionRegression / static_cast<double>(counter),2);  
  }
}

void OnlineRF::eval(Sample& sample, Result& result) {
  if(m_hp->type=="classification") {
    this->evalClassification(sample, result);
  } else {
    this->evalRegression(sample, result);
  }
}

void OnlineRF::evalClassification(Sample& sample, Result& result) {
  Result treeResult;
  Eigen::MatrixXd tauHatAll(m_hp->numTrees, *m_numClasses);

  for (int nTree = 0; nTree < m_hp->numTrees; ++nTree) {
    //calculate the prediction for the tree
    m_trees[nTree]->eval(sample, treeResult);
    //calculate the aggregate confidences and ITEs
    result.confidence += treeResult.confidence;
    if(m_hp->causal == true) {
      result.tauHat += treeResult.tauHat;

      //copy all ITE estimates from the tree into the matrix
      tauHatAll.row(nTree) = treeResult.tauHat;
    }
  }

  //average confidence
  result.confidence /= m_hp->numTrees;

  //prediction is associated with the max confidence
  result.confidence.maxCoeff(&result.predictionClassification);

  if(m_hp->causal == true) {
    //mean ITE estimate
    result.tauHat /= m_hp->numTrees;

    //all ITE estimates
    result.tauHatAllTrees = tauHatAll;
  }
}

void OnlineRF::evalRegression(Sample& sample, Result& result) {
  Eigen::MatrixXd tauHatAll(m_hp->numTrees, m_hp->numTreatments);
  double yVarHat=0; 
  double correction=0;
  Eigen::VectorXd tauVarHat = Eigen::VectorXd::Zero(m_hp->numTreatments);
  double correction2=0;
  Eigen::VectorXd yHatAll(m_hp->numTrees);
  //int totWeight=0;
  
  for(int nTree = 0; nTree < m_hp->numTrees; nTree++) {
    Result treeResult;

    //calculate the prediction for the tree
    m_trees[nTree]->eval(sample, treeResult);
    
    //prediction: average of prediction from individual trees (not weighted)
    //result.predictionRegression += treeResult.predictionRegression * static_cast<double>(treeResult.weight);
    //totWeight += treeResult.weight;    
    result.predictionRegression += treeResult.predictionRegression;

    yHatAll(nTree) = treeResult.predictionRegression;
    
    if(m_hp->causal == true) {
      result.tauHat += treeResult.tauHat;
      //copy all ITE estimates from the tree into the matrix
      tauHatAll.row(nTree) = treeResult.tauHat;

    } //close causal==true
  } // loop nTree
  
  //divide out for averages
  result.predictionRegression = result.predictionRegression / static_cast<double>(m_hp->numTrees);
//   if(totWeight > 0) {
//     result.predictionRegression = result.predictionRegression / totWeight;
//   }
  result.yHatAllTrees = yHatAll;
  if(m_hp->causal == true) {
    result.tauHat /= m_hp->numTrees;
    result.tauHatAllTrees = tauHatAll;
  }

  //Infinitesmal Jacknife - Wager 2014, bootstrap across trees
  // V_IJ = sum(Cov(^2))
  // Cov = B * sum((y - yhat) * (n - nhat))
  // cycle back through the trees to calc the cov
  for (int nTree = 0; nTree < m_hp->numTrees; nTree++) {
    //calculate the prediction for the tree
    yVarHat += pow(yHatAll(nTree) - result.predictionRegression, 2);
    correction += pow(yHatAll(nTree) - result.predictionRegression, 2);

    if(m_hp->causal==true) {
      for(int nTreat=0; nTreat < m_hp->numTreatments; nTreat++) {
	tauVarHat(nTreat) += pow(tauHatAll(nTree, nTreat) - result.tauHat(nTreat), 2);
	correction2 += pow(tauHatAll(nTree, nTreat) - result.tauHat(nTreat), 2);
      }
    }

  }
  //IJ variance estimate = Cov(N, t)^2
  yVarHat = yVarHat / (m_hp->numTrees - 1);

  if(m_hp->causal==true) {
    for(int nTreat=0; nTreat < m_hp->numTreatments; nTreat++) {
      tauVarHat(nTreat) = tauVarHat(nTreat) / (m_hp->numTrees - 1);
    }
  }
 
  //IJ-unbiased correction in case of small number of bootstrap samples: V_IJ-U = V_IJ - n/b^2 * sum((t-that)^2)
  //  yHatVar = yHatVar - 1/pow(b,2) * correction;
  //  if(m_hp->causal==true) {
  //    for(int nTreat=0; nTreat < m_hp->numTreatments; nTreat++) {
  //      tauHatVar(nTreat) = tauHatVar(nTreat) - 1/pow(b2,2) * correction2;
  //    }
  //  }

  //save results
  result.predictionVarianceRegression = yVarHat;
  if(m_hp->causal==true) {
    result.tauVarHat = tauVarHat;
  }
}

      
//   //if more than one unit in nodes pooled variance estimate
//   if(m_hp->counterThreshold > 2) { 
//     result.predictionVarianceRegression /= static_cast<double>(m_hp->numTrees - 1);
//     for(int nTreat=1; nTreat < m_hp->numTreatments; nTreat++) {
//       result.tauVarHat(nTreat) /= static_cast<double>(m_hp->numTrees - 1);
//     }
//   }
//   //Variances for counterThreshold == 2:
//   // use empirical variance from each tree
//   if(m_hp->counterThreshold <= 2) {
//     double sampVar = 0;
//     Eigen::VectorXd sampVarTau = Eigen::VectorXd::Zero(m_hp->numTreatments);

//     for (int nTree = 0; nTree < m_hp->numTrees; nTree++) {
//       m_trees[nTree]->eval(sample, treeResult);
//       //sample variance across trees
//       sampVar += pow(treeResult.predictionRegression - result.predictionRegression,2);

//       for(int nTreat=1; nTreat < m_hp->numTreatments; nTreat++) {
// 	sampVarTau(nTreat) += pow(treeResult.tauHat(nTreat) - result.tauHat(nTreat), 2);
//       }
//     }
//     //divide out by weights
//     if((m_hp->numTrees - 1) > 0) {
//       sampVar /= (m_hp->numTrees - 1);
//     }
//     result.predictionVarianceRegression = sampVar;

//     for(int nTreat=1; nTreat < m_hp->numTreatments; nTreat++) {
//     if((m_hp->numTrees - 1) > 0) {
//       sampVarTau(nTreat) /= (m_hp->numTrees - 1);
//     }
//     result.tauVarHat(nTreat) = sampVarTau(nTreat);
//     }
//   }
//}

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

// void OnlineRF::printInfo() {
//   cout << "RF Info: ";
//   cout << "Number of trees: " << m_trees.size() << std::endl;
// }

// void OnlineRF::print() {
//   cout << "RF details: " << std::endl;  
//   vector<Eigen::MatrixXd> rfParms = exportParms();
//   for(int nTree=0; nTree < rfParms.size(); ++nTree) {
//     cout << "\tTree: " << nTree << std::endl;
//     cout << "\t\t";
//     cout << rfParms[nTree] << std::endl;
//   }
// }

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

    //recursively get the score from the child nodes
    double rightChildScore = m_rightChildNode->score();
    double leftChildScore = m_leftChildNode->score();
    double rightChildCount = m_rightChildNode->getCount();
    double leftChildCount = m_leftChildNode->getCount();
    double childrenScore = (rightChildScore * rightChildCount + leftChildScore * leftChildCount) / (rightChildCount + leftChildCount + 1e-16);
     
    //score for node is improvement from split - difference in children
    score = childrenScore;

    if(m_hp->causal == true) { //causal models have negative MSE, need to flip to get max variance
      score = -score; 
    } else if(m_hp->type=="regression") { //regression models have negative MSE - need to get max variance
      score = -score;
    } else { //if non-causal classification - improvement from this to children
      selfScore = this->score();
      score = selfScore - childrenScore; //positive numbers are better for gini and entropy
    }

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
  vector<double> trainError(m_hp->numEpochs, 0.0);
  for (int nEpoch = 0; nEpoch < m_hp->numEpochs; ++nEpoch) {
    //permute the dataset
    randPerm(dataset.m_numSamples, randIndex);
    for (int nSamp = 0; nSamp < dataset.m_numSamples; ++nSamp) {
      if (m_hp->findTrainError == true) {
	if(m_hp->type=="classification") {
	  Result result(dataset.m_numClasses);
	  this->eval(dataset.m_samples[randIndex[nSamp]], result);
	  if (result.predictionClassification != dataset.m_samples[randIndex[nSamp]].yClass) {
	    trainError[nEpoch]++;
	  }
	} else { //type=="regression"
	  Result result(m_hp->numTreatments);
	  this->eval(dataset.m_samples[randIndex[nSamp]], result);
	  trainError[nEpoch]+=pow(result.predictionRegression - dataset.m_samples[randIndex[nSamp]].yReg, 2);
	}
      } //close findTrainError
      //update RF with datapoint
      this->update(dataset.m_samples[randIndex[nSamp]]);
    } //close nSamp loop
  } //close epoch loop 
} //close method

//// method for providing predictions from the model
vector<Result> OnlineRF::test(DataSet& dataset) {
  vector<Result> results;
  for (int nSamp = 0; nSamp < dataset.m_numSamples; nSamp++) {
    int num;      
    if(m_hp->type=="classification") {
      num = dataset.m_numClasses;
      Result result(num);
      this->eval(dataset.m_samples[nSamp], result);
      results.push_back(result);
    } else { //type=="regression"
      if(m_hp->causal == true) {
	num = m_hp->numTreatments;
	Result result(num);
	this->eval(dataset.m_samples[nSamp], result);
	results.push_back(result);
      } else { // if regression but not causal
	Result result;
	this->eval(dataset.m_samples[nSamp], result);
	results.push_back(result);
      }
    }
  }
  return results;
}
