########### orf_tst.R ################

### this program is test of the online_random_forest functionality from Rcpp
library(Rcpp)
library(RcppEigen)

sourceCpp("orf.cpp")
source("orf_functions.R")

######### SIMULATE DATA - note: used in analyzing real data this would mean importing a dataset ###########
### simulate some predictors - 
NCOL <- 5 #number of columns 
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


orfmod <- online_random_forest(x=x, y=y,
                               numRandomTests=5, 
                               counterThreshold=1000, 
                               maxDepth=15,
                               numTrees=10,
                               numEpochs = 1,
                               findTrainError=TRUE,
                               verbose=TRUE
                               )
length(orfmod)
names(orfmod)
length(orfmod$forest)
names(orfmod$forest)
dim(orfmod$forest[[1]])
orfmod$forest[[1]]

orfmod$featRange

# colnames(orfmod[[1]]) <- c('nodeNumber','parentNodeNumber','depth','isLeaf','label',
#                            'counter','parentCounter','labelStats_size','onlineTests_size',
#                            'labelStats_0','labelStats_1',
#                            'onlineTests1_feature','onlineTests1_threshold')

### test building a forest from the matrix

orfmod2 <- orf(x=x, y=y, orf=orfmod, trainModel=FALSE)
names(orfmod2)  
dim(orfmod2$forest[[1]])
orfmod2$forest[[1]]

orfmod$oobe
orfmod2$oobe

orfmod$n
orfmod2$n



#with train() function commented out - this code checks that the RF is properly built and then exported
orfmod2$oobe == orfmod$oobe
orfmod2$n == orfmod$n
length(orfmod2$hyperparameters) == length(orfmod$hyperparameters)
for(h in names(orfmod$hyperparameters)) {
  print(h)
  print(orfmod$hyperparameters[[h]] == orfmod2$hyperparameters[[h]])
}

length(orfmod$forest) == length(orfmod2$forest)

dim(orfmod$forest[[1]])
dim(orfmod2$forest[[1]])

length(orfmod$forest[[1]])
sum(orfmod$forest[[1]] == orfmod2$forest[[1]])
orfmod$forest[[1]] == orfmod2$forest[[1]]

### try with training applied 
#sourceCpp("orf.cpp")
orfmod3 <- orf(x=x, y=y, orf=orfmod)

names(orfmod3)  
dim(orfmod2$forest[[1]])
dim(orfmod3$forest[[1]])

orfmod$oobe
orfmod2$oobe
orfmod3$oobe

orfmod$n
orfmod2$n
orfmod3$n

orfmod$forest[[1]]
orfmod3$forest[[1]]

### simulate more data from the same parms
x2 <- matrix(data=rnorm(NCOL * NROW), ncol=NCOL, nrow=NROW)

### simulate the dependent variable related to the predictors
z2 <- x2 %*% betas + rnorm(NROW, 0, .1)
y2 <- ifelse(z2 > 0, 1, 0)

hist(z2) #histogram of z
table(y2) #distribtuion of the dependent variable


orfmod4 <- orf(x=x2, y=y2, orf=orfmod)
orfmod4$n
orfmod4$oobe
orfmod4$oobe/orfmod4$n

orfmod$n
orfmod$oobe
orfmod$oobe/orfmod$n

dim(orfmod4$forest[[1]])
orfmod4$forest[[1]]

#sourceCpp("orf.cpp")

#test predictions

#make predictions on original data using original model
pred1.1 <- predict_orf(x = x, orfModel=orfmod)
length(pred1.1)
dim(pred1.1[[1]])
length(pred1.1[[2]])
head(pred1.1[[1]])
head(pred1.1[[2]])
table(pred1.1[[2]])

table(pred1.1[["prediction"]], y)

#prediction on subsequent data with original model
pred2.1 <- predict_orf(x = x2, orfModel=orfmod)
table(pred2.1[["prediction"]], y2)

#prediction on original data with subsequent models
pred1.3 <- predict_orf(x=x, orfModel = orfmod3)
table(pred1.3[["prediction"]], y)

pred1.4 <- predict_orf(x=x, orfModel = orfmod4)
table(pred1.4[["prediction"]], y)

#prediction on subsequent data from subsequent models
pred2.4 <- predict_orf(x=x2, orfModel = orfmod4)
table(pred2.4[["prediction"]], y2)




##### test with reasonable parms
NCOL <- 100 #number of columns 
NROW <- 10000 #nbumber of rows

betas2 <- runif(NCOL) #simulate coefficients for the regression
x3 <- matrix(data=rnorm(NCOL * NROW), ncol=NCOL, nrow=NROW)

### simulate the dependent variable related to the predictors
z3 <- x3 %*% betas2 + rnorm(NROW, 0, .1)
y3 <- ifelse(z > 0, 1, 0)

orfmod5 <- online_random_forest(x=x3, y=y3,
                               numRandomTests=20, 
                               counterThreshold=1000, 
                               maxDepth=20,
                               numTrees=20,
                               numEpochs = 1,
                               findTrainError=TRUE,
                               verbose=TRUE)


pred3.5 <- predict_orf(x = x3, orfModel=orfmod5)
table(pred3.5[["prediction"]], y3)


### test orf_functions.R

orfmod6 <- init_orf(numClasses=2, numFeatures=ncol(x), numRandomTests = 5, counterThreshold = 1000, maxDepth = 15,
                   numTrees = 10, numEpochs = 1)
summary(orfmod6)

names(orfmod)
names(orfmod6)

orfmod$oobe;orfmod6$oobe
orfmod$n; orfmod6$n
orfmod$numClasses; orfmod6$numClasses
orfmod$featRange; orfmod6$featRange
length(orfmod$hyperparameters); length(orfmod6$hyperparameters)
length(orfmod$forest); length(orfmod6$forest)
dim(orfmod$forest[[1]]); dim(orfmod6$forest[[1]])

#fit model with all data at once
orfmod6_updated <- train_orf(model=orfmod6, x=x, y=y)
length(orfmod$forest); length(orfmod6_updated$forest)
dim(orfmod$forest[[1]]); dim(orfmod6_updated$forest[[1]])
orfmod$featRange; orfmod6_updated$featRange
orfmod6$featRange

#fit with NN data points
NN = nrow(x)
orfmod6_updated <- train_orf(model=orfmod6, x=matrix(x[1:NN,],nrow=NN), y=matrix(y[1:NN], nrow=NN))


#fit a model one datapoint at a time
orfmod6_updated2 <- init_orf(numClasses=2, numFeatures=ncol(x), numRandomTests = 5, counterThreshold = 1000, maxDepth = 15,
                             numTrees = 10, numEpochs = 1)
i <- 1
for(i in 1:nrow(x)) {
  orfmod6_updated2 = train_orf(orfmod6_updated2, matrix(x[i,], nrow=1, byrow = TRUE), matrix(y[i], nrow=1))  
}

summary(orfmod6_updated2)

orfmod6_updated$oobe/orfmod6_updated$n
orfmod6_updated2$oobe/orfmod6_updated2$n

####testing cv method
sourceCpp("orf.cpp")
cv_res <- orf_cv(x=x, y=y, numClasses=2, numRandomTests=10, counterThreshold=1000, maxDepth=10, numTrees=10, numEpochs=1, nfolds=5)
summary(cv_res)  
dim(cv_res$probs); head(cv_res$probs)
dim(cv_res$classes); head(cv_res$classes)

table(cv_res$accurate)
mean(cv_res$accurate)

### demonstrate ability to tune a hyper parameter
rt_list = c(2,10,20)
ct_list = c(50,100,1000)
md_list = c(3,10,20)
nt_list = c(1,10,50,100)

gr <- expand.grid(rt_list, ct_list, md_list, nt_list)
names(gr) <- c('numRandomTests','counterThreshold','maxDepth','numTrees')
dim(gr)
head(gr)

counter = 0
res <- as.data.frame(matrix(NA,nrow=nrow(gr), ncol=ncol(gr)+1))
colnames(res) <- c(names(gr),'perc_acc')
parms <- gr[1,]
for(i in 1:nrow(gr)) {
  parms <- gr[i,]
  print(counter)
  cv_res <- orf_cv(x=x, y=y, numClasses=2, 
                   numRandomTests=parms$numRandomTest,
                   counterThreshold=parms$counterThreshold, 
                   maxDepth=parms$maxDepth, 
                   numTrees=parms$numTrees, 
                   numEpochs=1, 
                   nfolds=5)
  
  res[counter,] <- unlist(c(parms, mean(cv_res$accurate)))
  counter = counter + 1
}

#what are the best parameters?
res[which.max(res$perc_acc),]


### test orf with a multinomial outcome ###
NCOL <- 100 #number of columns 
NROW <- 10000 #nbumber of rows

betas <- rnorm(NCOL) #simulate coefficients for the regression
x <- matrix(data=rnorm(NCOL * NROW), ncol=NCOL, nrow=NROW)

### simulate the dependent variable related to the predictors
z <- x %*% betas + rnorm(NROW, 0, .1)
y <- floor(((z - max(z)) / (max(z) - min(z)) + 1) * 10)
summary(y)
table(y)

sourceCpp("orf.cpp")
source("orf_functions.R")

orfmod7 <- init_orf(numClasses=length(unique(y)), numFeatures=ncol(x), numRandomTests = 50, counterThreshold = 100, maxDepth = 15,
                    numTrees = 100, numEpochs = 1, minFeatRange=rep(0, ncol(x)), maxFeatRange = rep(1, ncol(x)))
summary(orfmod7)
dim(orfmod7$forest[[1]])
#orfmod7$forest
#orfmod7$featRange

orfmod7 <- train_orf(model = orfmod7, x = as.matrix(x), y=as.matrix(y))        

#compare actuals to predicted
porf <- predict_orf(x=as.matrix(x), orfModel = orfmod7)
summary(porf)
head(porf$prediction)
table(porf$prediction)
table(y)
table(y, porf$prediction)


