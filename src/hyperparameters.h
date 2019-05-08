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

#ifndef HYPERPARAMETERS_H_
#define HYPERPARAMETERS_H_

#include <RcppEigen.h>
//[[Rcpp::depends(RcppEigen)]]

#include <string>
using namespace std;

class Hyperparameters {
 public:

  // Forest
  int numRandomTests;
  int counterThreshold;
  int maxDepth;
  int numTrees;
  
  string method; //splitting criteria.  gini, mse etc
  string type; //rf type: classification, regression

  bool causal; //causal rf indicator vs not causal

  // Experimenter
  bool findTrainError;
  int numEpochs;

  // Data
  string trainData;
  string trainLabels;
  string testData;
  string testLabels;

  // Output
  bool verbose;

  Rcpp::List hpToList();

};

#endif /* HYPERPARAMETERS_H_ */

