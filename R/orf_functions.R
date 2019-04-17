##########################################################################
# orf_functions.R - R functions to call Online Random Forests
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Written (W) 2019 Michael Greene, mfgreene79@yahoo.com
# Copyright (C) 2019 Michael Greene

##########################################################################

#' Initialize Online Random Forest
#'
#' This function initializes the online random forest (or causal online random forest).
#' @param numClasses Number of classes expected to be classified.
#' @param numFeatures Number of features that will be in the learning dataset.
#' @param numRandomTests Number of random tests, i.e., number of features to be selected at each node for evaluation.
#' @param counterThreshold Threshold for number of observations before a split will occur.
#' @param maxDepth Maximum depth for each tree.
#' @param numTrees Number of trees in the forest.
#' @param numEpochs Number of epochs for processing during each training step.
#' @param type Type of Random Forest.  Only `classification` is implemented at this time.
#' @param method Method used for determining if a node should split.  Implemented methods are `gini` for Gini Impurity or `entropy` for entropy.
#' @param causal Is the Random Forest a Causal Random Forest?  Defaults to FALSE.
#' @param minFeatRange If provided, the minimum expected values for the features.  Must be a vector of the same length as numFeatures.  The min and max feature ranges are used to draw random thresholds when creating random tests at each node.  The forest will expand the range as necessary based on new data.  Defaults to NULL.
#' @param maxFeatRange If provided, the maximum expected values for the features.  Must be a vector of the same length as numFeatures.  The min and max feature ranges are used to draw random thresholds when creating random tests at each node.  The forest will expand the range as necessary based on new data.  Defaults to NULL.
#' @param labels Labels for the classes.  Length must be equal to numClasses.  Defaults to sequence 1:numClasses.
#' @param findTrainError Should the forest calculate the out of back error on hte training data.
#' @keywords causal random forest, online learning, incremental learning, out-of-core learning, online random forest
#' @export oobe Out of bag error count
#' @export n Count of observations that have passed through the algorithm
#' @export numClasses Number of classes passed from input
#' @export featRange List with two elements, each a vector of length numFeatures representing the minimum and maximum observed from the features.
#' @export forest A list comprising the forest parameters.  Each element is a matrix representing a single tree in the forest.  Each row in the matrix represents a node in the tree.
#' @export hyperparameters A list with the hyperparameters passed to the model.
#' @export labels Labels passed from the input.
#' @examples
#' ## simulate a data point with 10 columns
#' x <- matrix(runif(10), nrow=1)
#'
#' ## initialize the model object
#' orfmod <- init_orf(numClasses = 2, numFeatures = 10, numRandomTests = 2, counterThreshold = 10, maxDepth = 5, numTrees = 10, numEpochs = 1)
#'
#' ## train the model with the data
#' orfmod <- train_orf(model = orfmod, x = x, y=0)
#' 


init_orf <- function(numClasses, numFeatures, numRandomTests, counterThreshold, maxDepth, numTrees, numEpochs, 
                     type = 'classification', method = 'gini', causal=FALSE,
                     minFeatRange = NULL, maxFeatRange = NULL, labels=c(1:numClasses),
                     findTrainError=FALSE) {
  
  #prepare model object to return
  out <- list()
  out$oobe = 0
  out$n = 0
  out$numClasses = numClasses
  
  #feature range
  if(is.null(minFeatRange) || is.null(maxFeatRange)) {
    minFeatRange = rep(0, numFeatures)
    maxFeatRange = rep(0, numFeatures)
  }
  featRange = list("minFeatRange" = minFeatRange, "maxFeatRange" = maxFeatRange)
  out$featRange = featRange
  
  forest_list = list()
  #process for every tree in the forest - initializing randomTests differently each time
  for(nTree in 0:(numTrees-1)) {
      
    #forest matrix
    if(causal == FALSE) {
      forest = matrix(data=NA, nrow = 1, ncol=13 + 3 * numClasses + 2 * numRandomTests * (1 + numClasses))
      colnames_forest = c("nodeNumber", "parentNodeNumber", "rightChildNodeNumber", "leftChildNodeNumber",
                           "depth", "isLeaf","label","counter","parentCounter","numClasses","numRandomTests")
      fdat = c(0, -1, -1, -1, 0, 1, 0, 0, 0, numClasses, numRandomTests)
      
      colnames_forest = c(colnames_forest, paste0("labelStats_",c(0:(numClasses-1))))
      fdat = c(fdat, rep(0, numClasses))
        
      colnames_forest = c(colnames_forest, "bestTest_feature", "bestTest_threshold")
      fdat = c(fdat, -1, 0)
      
      for(i in c(0:(numClasses-1))) {
        colnames_forest = c(colnames_forest, paste0("bestTest_trueStats",i))
      }
      fdat = c(fdat, rep(0, numClasses))
      
      for(i in c(0:(numClasses-1))) {
        colnames_forest = c(colnames_forest, paste0("bestTest_falseStats",i))
      }
      fdat = c(fdat, rep(0, numClasses))
  
      randFeats = floor(runif(numRandomTests, min = 0, max = numFeatures))
      randThreshs = runif(numRandomTests, min=minFeatRange[randFeats+1], max=maxFeatRange[randFeats+1])
      
      for(j in c(0:(numRandomTests - 1))) {
        colnames_forest = c(colnames_forest, paste0("randomTest", j, "_feature"))
        colnames_forest = c(colnames_forest, paste0("randomTest", j, "_threshold"))
        randFeat = randFeats[j+1]
        randThresh = randThreshs[j+1]
        fdat = c(fdat, randFeat, randThresh)
        for(i in c(0:(numClasses-1))) {
          colnames_forest = c(colnames_forest, paste0("randomTest", j, "_trueStats", i))
        }
        for(i in c(0:(numClasses-1))) {
          colnames_forest = c(colnames_forest, paste0("randomTest", j, "_falseStats", i))
        }
        fdat = c(fdat, rep(0, numClasses*2))
      }
    
    } else { #causal == TRUE                 
      forest = matrix(data=NA, nrow = 1, ncol=15 + 8 * numClasses + 2 * numRandomTests * (1 + 2 * numClasses))
      colnames_forest = c("nodeNumber", "parentNodeNumber", "rightChildNodeNumber", "leftChildNodeNumber",
                          "depth", "isLeaf","label","counter")
      fdat = c(0, -1, -1, -1, 0, 1, 0, 0)
      
      colnames_forest = c(colnames_forest, paste0("ite_",c(0:(numClasses-1))))
      fdat = c(fdat, 0, 0)
      
      colnames_forest = c(colnames_forest, "treatCounter", "controlCounter")
      fdat = c(fdat, 0, 0)
      
      colnames_forest = c(colnames_forest,"parentCounter","numClasses","numRandomTests")
      fdat = c(fdat, 0, numClasses, numRandomTests)
  
      colnames_forest = c(colnames_forest, paste0("labelStats_",c(0:(numClasses-1))))
      fdat = c(fdat, rep(0, numClasses))
  
      colnames_forest = c(colnames_forest, paste0("treatLabelStats_",c(0:(numClasses-1))))
      fdat = c(fdat, rep(0, numClasses))
  
      colnames_forest = c(colnames_forest, paste0("controlLabelStats_",c(0:(numClasses-1))))
      fdat = c(fdat, rep(0, numClasses))
      
      colnames_forest = c(colnames_forest, "bestTest_feature", "bestTest_threshold")
      fdat = c(fdat, -1, 0)
      
      for(i in c(0:(numClasses-1))) {
        colnames_forest = c(colnames_forest, paste0("bestTest_treatTrueStats",i))
      }
      fdat = c(fdat, rep(0, numClasses))
      
      for(i in c(0:(numClasses-1))) {
        colnames_forest = c(colnames_forest, paste0("bestTest_treatFalseStats",i))
      }
      fdat = c(fdat, rep(0, numClasses))
  
      for(i in c(0:(numClasses-1))) {
        colnames_forest = c(colnames_forest, paste0("bestTest_controlTrueStats",i))
      }
      fdat = c(fdat, rep(0, numClasses))
      
      for(i in c(0:(numClasses-1))) {
        colnames_forest = c(colnames_forest, paste0("bestTest_controlFalseStats",i))
      }
      fdat = c(fdat, rep(0, numClasses))
      
      for(j in c(0:(numRandomTests - 1))) {
        colnames_forest = c(colnames_forest, paste0("randomTest", j, "_feature"))
        colnames_forest = c(colnames_forest, paste0("randomTest", j, "_threshold"))
        randFeat = floor(runif(1, min = 0, max = numFeatures))
        randThresh = runif(1, min=minFeatRange[randFeat+1], max=maxFeatRange[randFeat+1])
        fdat = c(fdat, randFeat, randThresh)
        for(i in c(0:(numClasses-1))) {
          colnames_forest = c(colnames_forest, paste0("randomTest", j, "_treatTrueStats", i))
        }
        for(i in c(0:(numClasses-1))) {
          colnames_forest = c(colnames_forest, paste0("randomTest", j, "_treatFalseStats", i))
        }
        for(i in c(0:(numClasses-1))) {
          colnames_forest = c(colnames_forest, paste0("randomTest", j, "_controlTrueStats", i))
        }
        for(i in c(0:(numClasses-1))) {
          colnames_forest = c(colnames_forest, paste0("randomTest", j, "_controlFalseStats", i))
        }
        fdat = c(fdat, rep(0, numClasses*4))
      }
      
    } #causal == TRUE  
    forest[1,] = fdat
    colnames(forest) = colnames_forest
  
    forest_list[[paste0("tree",nTree)]] = forest
  } #close loop on forest - nTree
    
  out$forest = forest_list
  
  #hyperparameters
  hp = list()
  hp$numRandomTests = numRandomTests
  hp$counterThreshold = counterThreshold
  hp$maxDepth = maxDepth
  hp$numTrees = numTrees
  hp$numEpochs = numEpochs
  hp$findTrainError = findTrainError
  hp$verbose = FALSE
  hp$type = type
  hp$method = method
  hp$causal = causal

  out$hyperparameters = hp  
  
  out$labels = labels
  
  return(out)
}

#' Train Online Random Forest
#'
#' This function trains an initialized online random forest (or causal online random forest).  The output is a list representing an ORF.  For more information see the `init_orf()` function.
#' @param model An initialized online random forest object created by the `init_orf()` function.
#' @param x A matrix of features to train the forest.  Must be equal to `numFeatures` when forest was initialized.
#' @param y Vector of classes.  Multiclass classification is supported.  Must be integers.
#' @param w Vector of treatment assignments.  Must be 0 or 1.  Defaults to NULL.  Must be provided if the model object was initialized with `causal=TRUE`
#' @param trainModel Should the forest be trained on the new data?  Defaults to TRUE.  Useful for testing.
#' @examples
#' ## simulate a data point with 10 columns
#' x <- matrix(runif(10), nrow=1)
#'
#' ## initialize the model object
#' orfmod <- init_orf(numClasses = 2, numFeatures = 10, numRandomTests = 2, counterThreshold = 10, maxDepth = 5, numTrees = 10, numEpochs = 1)
#'
#' ## train the model with the data
#' orfmod <- train_orf(model = orfmod, x = x, y=0)
#' 
#' @keywords causal random forest, online learning, incremental learning, out-of-core learning, online random forest
#' 


train_orf <- function(model, x, y, w=NULL, trainModel=TRUE) {
  if(model$hyperparameters$causal==FALSE) {
    newmodel = orf(x, y, model, trainModel=trainModel)
  } else {
    newmodel = corf(x, y, w, model, trainModel=trainModel)
  }
  newmodel$labels = model$labels
  return(newmodel)
}


#' Predict Online Random Forest
#'
#' This function trains an initialized online random forest (or causal online random forest).
#' @param model An initialized online random forest object created by the `init_orf()` function and ideally trained with data using the `train_orf()` function
#' @param x A matrix of features on which to generate predictions.  Must be equal to `numFeatures` when forest was initialized.
#' @param iteAll If a causal forest, should all ITE estimates be returned?  Default=FALSE.
#' @keywords causal random forest, online learning, incremental learning, out-of-core learning, online random forest
#' @export prediction A vector of class predictions
#' @export confidence A matrix of probabilities with a column for each class
#' @export ite If a causal forest, the individual treatment effects
#' @export ite_all If a causal forest, a list with one element for each class, each a matrix with the individual treatment effects for every tree in the forest.
#' @examples
#' ## simulate a data point with 10 columns
#' x <- matrix(runif(10), nrow=1)
#'
#' ## initialize the model object
#' orfmod <- init_orf(numClasses = 2, numFeatures = 10, numRandomTests = 2, counterThreshold = 10, maxDepth = 5, numTrees = 10, numEpochs = 1)
#'
#' ## train the model with the data
#' orfmod <- train_orf(model = orfmod, x = x, y=0)
#' 
#' ## make predictions on new data
#' x2 <- matrix(runif(10), nrow=1)
#' p <- predict_orf(orfmod, x)
#' 

predict_orf <- function(model, x, iteAll=FALSE) {
  return(predictOrf(as.matrix(x), model, iteAll))
}

#' Get Feature Importances
#'
#' This function gets the feature importances from an Online Random Forest
#' @param model An initialized online random forest object created by the `init_orf()` function and ideally trained with data using the `train_orf()` function
#' @keywords causal random forest, online learning, incremental learning, out-of-core learning, online random forest
#' @export importances A vector of variable importances averaged across the trees in the random forest.  Standardized to sum to 1.
#' #' @examples
#' ## simulate a data point with 10 columns
#' x <- matrix(runif(10), nrow=1)
#'
#' ## initialize the model object
#' orfmod <- init_orf(numClasses = 2, numFeatures = 10, numRandomTests = 2, counterThreshold = 10, maxDepth = 5, numTrees = 10, numEpochs = 1)
#'
#' ## train the model with the data
#' orfmod <- train_orf(model = orfmod, x = x, y=0)
#' 
#' get_importance(orfmod)
#' 

get_importance <- function(model) {
  return(getImps_(model));
}


