##########################################################################
# orf_functions.R - R functions to call Online Random Forests
##########################################################################

library(Rcpp)
library(RcppEigen)
sourceCpp("orf.cpp")

init_orf <- function(numClasses, numFeatures, numRandomTests, counterThreshold, maxDepth, numTrees, numEpochs, 
                     minFeatRange = NULL, maxFeatRange = NULL, labels=c(1:numClasses),
                     findTrainError=FALSE, verbose=FALSE) {
  
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
  
  #forest matrix
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
  
  for(j in c(0:(numRandomTests - 1))) {
    colnames_forest = c(colnames_forest, paste0("randomTest", j, "_feature"))
    colnames_forest = c(colnames_forest, paste0("randomTest", j, "_threshold"))
    randFeat = floor(runif(1, min = 1, max = numFeatures + 1))
    randThresh = runif(1, min=minFeatRange[randFeat], max=maxFeatRange[randFeat])
    fdat = c(fdat, randFeat, randThresh)
    for(i in c(0:(numClasses-1))) {
      colnames_forest = c(colnames_forest, paste0("randomTest", j, "_trueStats", i))
    }
    for(i in c(0:(numClasses-1))) {
      colnames_forest = c(colnames_forest, paste0("randomTest", j, "_falseStats", i))
    }
    fdat = c(fdat, rep(0, numClasses*2))
  }

  forest[1,] = fdat
  colnames(forest) = colnames_forest

  forest_list = list()
  for(i in 1:numTrees) {
    forest_list[[paste0("tree",i)]] = forest
  }
    
  out$forest = forest_list
  
  #hyperparameters
  hp = list()
  hp$numRandomTests = numRandomTests
  hp$counterThreshold = counterThreshold
  hp$maxDepth = maxDepth
  hp$numTrees = numTrees
  hp$numEpochs = numEpochs
  hp$findTrainError = findTrainError
  hp$verbose = verbose

  out$hyperparameters = hp  
  
  out$labels = labels
  
  return(out)
}

train_orf <- function(model, x, y, trainModel=TRUE) {
  newmodel = orf(x, y, model, trainModel=trainModel)
  newmodel$labels = model$labels
  return(newmodel)
}

