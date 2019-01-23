########### orf_tst.R ################

### this program is test of the online_random_forest functionality from Rcpp
library(Rcpp)
library(RcppEigen)

######### SIMULATE DATA - note: used in analyzing real data this would mean importing a dataset ###########
### simulate some predictors - 
NCOL <- 100 #number of columns 
NROW <- 10000 #nbumber of rows

betas <- rnorm(NCOL) #simulate coefficients for the regression
x <- matrix(data=rnorm(NCOL * NROW), ncol=NCOL, nrow=NROW)

### simulate the dependent variable related to the predictors
z <- x %*% betas + rnorm(NROW, 0, .1)
y <- ifelse(z > 0, 1, 0)

hist(z) #histogram of z
table(y) #distribtuion of the dependent variable

########## Done simulating data ############

### test of RcppEigen functionality
 # sourceCpp("rcpp_eigen_test.cpp")
 # m = matrix(rnorm(100, 0, 1), ncol=10)
 # getEigenValues(m)

sourceCpp("orf2.cpp")
  
orfmod <- online_random_forest(x=x, y=y,
                               numRandomTests=10, 
                               counterThreshold=10, 
                               maxDepth=10,
                               numTrees=100,
                               numEpochs = 10,
                               findTrainError=TRUE,
                               verbose=TRUE)
length(orfmod)
dim(orfmod[[1]])
dim(orfmod[[2]])

colnames(orfmod[[1]]) <- c('nodeNumber','parentNodeNumber','depth','isLeaf','label',
                           'counter','parentCounter','labelStats_size','onlineTests_size',
                           'labelStats_0','labelStats_1',
                           'onlineTests1_feature','onlineTests1_threshold')
orfmod[[1]]
length(orfmod)

dim(orfmod[[1]])
head(orfmod[[1]])
tail(orfmod[[1]])
