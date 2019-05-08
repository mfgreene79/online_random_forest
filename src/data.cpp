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

#include "data.h"
#include "hyperparameters.h"

void DataSet::findFeatRange() {
  m_minFeatRange = Eigen::VectorXd(m_numFeatures);
  m_maxFeatRange = Eigen::VectorXd(m_numFeatures);

  double minVal, maxVal;
  for (int nFeat = 0; nFeat < m_numFeatures; nFeat++) {
    minVal = m_samples[0].x(nFeat);
    maxVal = m_samples[0].x(nFeat);
    for (int nSamp = 1; nSamp < m_numSamples; nSamp++) {
      if (m_samples[nSamp].x(nFeat) < minVal) {
	minVal = m_samples[nSamp].x(nFeat);
      }
      if (m_samples[nSamp].x(nFeat) > maxVal) {
	maxVal = m_samples[nSamp].x(nFeat);
      }
    }
    
    m_minFeatRange(nFeat) = minVal;
    m_maxFeatRange(nFeat) = maxVal;
  }
}

Result::Result() {
}

Result::Result(const int& numClasses) : confidence(Eigen::VectorXd::Zero(numClasses)),
					ite(Eigen::VectorXd::Zero(numClasses))
 {
}

DataSet::DataSet() {
}


/////version to apply when using a non-causal random forest
/////create a DataSet item from input matrix and y
DataSet::DataSet(Eigen::MatrixXd x, Eigen::VectorXd y) {
  //creates a DataSet class from matrices x and y
  //  DataSet ds;
  m_numFeatures = x.cols();
  m_numSamples = x.rows();

  set<int> labels;
  for (int nSamp = 0; nSamp < x.rows(); ++nSamp) {
    Sample sample;
    sample.x = Eigen::VectorXd(m_numFeatures);
    sample.id = nSamp;
    sample.w = 1.0;
    sample.y = y(nSamp);
    labels.insert(sample.y);
    for (int nFeat = 0; nFeat < m_numFeatures; ++nFeat) {
      sample.x(nFeat) = x(nSamp, nFeat);
    } //loop nFeat
    m_samples.push_back(sample); // push sample into dataset
  } //loop nSamp
  m_numClasses = labels.size();

  this->findFeatRange();
}

/////version to apply when using a causal random forest - includes treatment indicators
DataSet::DataSet(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd W) {
  //creates a DataSet class from matrices x and y
//   DataSet ds;
//   ds.m_numFeatures = x.cols();
//   ds.m_numSamples = x.rows();

  m_numFeatures = x.cols();
  m_numSamples = x.rows();

  set<int> labels;
  for (int nSamp = 0; nSamp < x.rows(); ++nSamp) {
    Sample sample;  
    sample.x = Eigen::VectorXd(m_numFeatures);
    sample.id = nSamp;
    sample.w = 1.0;
    sample.y = y(nSamp);
    sample.W = W(nSamp);
    labels.insert(sample.y);
    for (int nFeat = 0; nFeat < m_numFeatures; ++nFeat) {
      sample.x(nFeat) = x(nSamp, nFeat);
    } //loop nFeat
    m_samples.push_back(sample); // push sample into dataset
  } //loop nSamp
  m_numClasses = labels.size();
  
  this->findFeatRange();
}

//create dataset for testing purposes
DataSet::DataSet(Eigen::MatrixXd x, int numClasses) {
  //creates a DataSet class from matrix x
  //  DataSet ds;
  m_numFeatures = x.cols();
  m_numSamples = x.rows();
  
  set<int> labels;  
  for (int nSamp = 0; nSamp < x.rows(); ++nSamp) {
    Sample sample;
    sample.x = Eigen::VectorXd(m_numFeatures);
    sample.id = nSamp;
    sample.w = 1.0;
    for (int nFeat = 0; nFeat < m_numFeatures; ++nFeat) {
      sample.x(nFeat) = x(nSamp, nFeat);
    } //loop nFeat
    m_samples.push_back(sample); // push sample into dataset
  } //loop nSamp
  m_numClasses = numClasses;
}

