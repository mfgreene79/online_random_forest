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

#include "data.h"

void DataSet::findFeatRange() {
    m_minFeatRange = VectorXd(m_numFeatures);
    m_maxFeatRange = VectorXd(m_numFeatures);

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

Result::Result(const int& numClasses) : confidence(VectorXd::Zero(numClasses)),
					ite(VectorXd::Zero(numClasses))
 {
}
