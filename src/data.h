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
  int y; //target
  double w; //weight
  int id; //id
  int W; //treatment identifier
};

class DataSet {
 public:

  //constructor for data without treatment assignments
  DataSet();
  DataSet(Eigen::MatrixXd x, Eigen::VectorXd y);
  //constructor for data with treatment assignments
  DataSet(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd W);
  //constructor for data without treatment assignments or outcome (for testing)
  DataSet(Eigen::MatrixXd x, int numClasses);
  void findFeatRange();

  vector<Sample> m_samples;
  int m_numSamples;
  int m_numFeatures;
  int m_numClasses;

  Eigen::VectorXd m_minFeatRange;
  Eigen::VectorXd m_maxFeatRange;
};

class Result {
public:
  Result();
  Result(const int& numClasses);

  Eigen::VectorXd confidence;
  int prediction;
  Eigen::VectorXd ite; //individual treatment effect - difference over control - one value per class
  
  Eigen::MatrixXd iteAllTrees; //capture values for all trees - one col per class one row per tree
};

#endif /* DATA_H_ */
