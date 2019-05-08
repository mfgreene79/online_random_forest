\name{corf-package}
\alias{corf-package}
\alias{corf}
\docType{package}
\title{
\packageTitle{corf}
}
\description{
\packageDescription{corf}
}
\details{

The DESCRIPTION file:
\packageDESCRIPTION{corf}
\packageIndices{corf}
  
}
\author{
\packageAuthor{corf}

Maintainer: \packageMaintainer{corf}
}
\references{
Athey, Susan, and Guido Imbens. "Recursive partitioning for heterogeneous causal effects." Proceedings of the National Academy of Sciences 113.27 (2016): 7353-7360.
\cr
Saffari, Amir, et al. "On-line random forests." 2009 ieee 12th international conference on computer vision workshops, iccv workshops. IEEE, 2009.
\cr
Wager, Stefan, and Susan Athey. "Estimation and inference of heterogeneous treatment effects using random forests." Journal of the American Statistical Association 113.523 (2018): 1228-1242.
}
\keyword{ package }
\examples{
## simulate a data point with 10 columns
x <- matrix(runif(10), nrow=1)

## initialize the model object
orfmod <- init_orf(numClasses = 2, numFeatures = 10, numRandomTests = 2,
                   counterThreshold = 10, maxDepth = 5, numTrees = 10,
                   numEpochs = 1)

## train the model with the data
orfmod <- train_orf(model = orfmod, x = x, y=as.matrix(0))
}
