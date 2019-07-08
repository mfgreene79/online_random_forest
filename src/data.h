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

#include <RcppEigen.h>
//[[Rcpp::depends(RcppEigen)]]


#ifndef DATA_H_
#define DATA_H_

#include <fstream>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <set>
#include <string>

using namespace std;

// DATA CLASSES
class Sample {
public:
  Eigen::VectorXd x; //features
  int yClass; //class target
  double w; //weight
  int id; //id
  int W; //treatment identifier
  double yReg; //regression target
};

class DataSet {
 public:

  //constructor for data without treatment assignments
  DataSet();
  DataSet(Eigen::MatrixXd x, Eigen::VectorXd y, 
	  std::string type);
  //constructor for data with treatment assignments
  DataSet(Eigen::MatrixXd x, Eigen::VectorXd y, 
	  Eigen::VectorXd W, 
	  std::string type);
  //constructor for data without treatment assignments or 
  //  outcome (for testing) - classification
  DataSet(Eigen::MatrixXd x, int numClasses);
  //  regression
  DataSet(Eigen::MatrixXd x);
  void findFeatRange();

  vector<Sample> m_samples;
  int m_numSamples;
  int m_numFeatures;
  int m_numClasses;
  int m_numTreatments;

  Eigen::VectorXd m_minFeatRange;
  Eigen::VectorXd m_maxFeatRange;
};

class Result {
public:
  Result();
  Result(const int& num);

  Eigen::VectorXd confidence; //probability of prediction classes
  int predictionClassification; //class prediction for classification
  double predictionRegression; //outcome prediction for regression
  double predictionVarianceRegression; //outcome prediction variance for regression
  int weight; //weighting for regression averaging
  Eigen::VectorXd tauHat; //individual treatment effect - difference over control - one value per class or treatment
  Eigen::VectorXd tauVarHat; //variance estimate in the individual treatment effect - one value per treatment
  
  Eigen::MatrixXd tauHatAllTrees; //capture values for all trees - one col per class one row per tree
  Eigen::VectorXd yHatAllTrees; //capture values for all trees - one entry per tree
};

#endif /* DATA_H_ */
