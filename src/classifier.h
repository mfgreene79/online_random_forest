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

#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_

#include "data.h"
#include "hyperparameters.h"

using namespace std;

class Classifier {
public:
  Classifier(const Hyperparameters& hp, const int& numClasses);
    
  virtual ~Classifier();
  
  virtual void update(Sample& sample) = 0;
  virtual vector<Eigen::MatrixXd> exportParms() = 0; 
  virtual void eval(Sample& sample, Result& result) = 0;

  virtual double getOOBE() = 0;
  virtual double getCounter() = 0;

  virtual void printInfo() = 0;
  virtual void print() = 0;

  //functions to access and edit the min and max feature ranges
  virtual pair<VectorXd,VectorXd> getFeatRange() = 0;
  virtual void updateFeatRange(VectorXd minFeatRange, VectorXd maxFeatRange) = 0;  

  //method for getting the feature importance
  virtual MatrixXd getFeatureImportance() = 0;

  const string name() const {
    return m_name;
  }

protected:
  const int* m_numClasses;
  const Hyperparameters* m_hp;
  string m_name;


};

#endif /* CLASSIFIER_H_ */
