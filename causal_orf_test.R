########### causal_orf_test.R ################

### this program is test of the online_random_forest functionality from Rcpp
library(Rcpp)
library(RcppEigen)
library(dplyr)
rm(list=ls())
sourceCpp("orf.cpp")
source("orf_functions.R")
logit <- function(p) {p/(1-p)}
logistic <- function(x) {1/(1+exp(-x))}


##########################################################################################################################
#
#  TESTING CAUSAL RANDOM FOREST COMPONENTS
#
##########################################################################################################################

NCOL <- 20 #number of columns
NROW <- 10000 #nbumber of rows

#simulate coefficients for the regression
betas <- rnorm(NCOL)

x <- matrix(data=rnorm(NCOL * NROW), ncol=NCOL, nrow=NROW)

#simulate individual treatment effects that are a function of the first M coefficients
M <- 5
ites <- x[,1:M] %*% (betas[1:M])/M + .5
hist(ites)
mean(ites)

hist(logistic(ites))
mean(logistic(ites))
mean(exp(ites))

#simulate w as the treatment assignment independent of x and ite
w <- ifelse(runif(NROW) > .5, 1, 0)

### simulate the dependent variable related to the predictors
z0 <- x %*% betas
z1 <- z0 + ites
z <- x %*% betas + ites*w

p <- logistic(z)
summary(p)
r <- runif(NROW)
y <- ifelse(r < p, 1, 0)

hist(z) #histogram of z
summary(ites)
table(y) #distribtuion of the dependent variable

#check average treatment effect
tapply(y, w, mean)

cbind(y, w, x) -> itedat
colnames(itedat) <- c("y","w",paste0("x",1:NCOL))
itedat %>%
  as_data_frame(.) %>%
  group_by(w) %>%
  summarise(y=mean(y))

### check ites by x covariates (interactions with x)
itedat %>%
  as_data_frame(.) %>%
  mutate(x1gt0 = ifelse(x1 > 0, 1, 0)) %>%
  group_by(x1gt0, w) %>%
  summarise(y=mean(y))


#see if we can recover the individual treatment effects
#sourceCpp("orf.cpp")

crf <- causal_online_random_forest(x=x, y=y, treat=w, numRandomTests = 10, counterThreshold = 100, trainModel = TRUE,
                                   maxDepth = 25, numTrees = 100, numEpochs = 1, type = "classification", method = "gini",
                                   causal = TRUE, findTrainError = TRUE, verbose = TRUE)

crf$oobe/crf$n

crf_pred <- predict_orf(x = x, orfModel=crf)
length(crf_pred)
names(crf_pred)

#check predictions against actuals
table(crf_pred$prediction, y)
mean(y==crf_pred$prediction)

#check the treatment effect identified
tapply(y, w, mean)
mean(y[w==1]) - mean(y[w==0])

mean(crf_pred$ite[,"1"])

mean(z1 - z0)
hist(z1 - z0)

hist(logistic(crf_pred$ite[,"1"]))

ite_est <- 1 + crf_pred$ite[,"1"]/crf_pred$confidence[,"1"]

plot(ite_est, exp(ites))
abline(lm(exp(ites)~ite_est), col='red', lty=2)

p_0 <- crf_pred$confidence[,1]
p_1 <- crf_pred$confidence[,2]

head(crf_pred$ite)

p_1_treat <- p_1 + crf_pred$ite[,2]

head(cbind(p_1, p_1_treat))
summary(p_1_treat/p_1)
summary(exp(ites))

#look at confidence interval for ites
dim(crf_pred$ite_all[["1"]])
ite_CI <- apply(crf_pred$ite_all[["1"]], 1, quantile, probs=c(.05, .5, .95))
dim(t(ite_CI))
head(t(ite_CI))
summary(t(ite_CI))

### check out the first couple trees in the forest
dim(crf$forest$tree0)
head(colnames(crf$forest$tree0), n=20)
crf$forest$tree0 %>% as_data_frame(.) %>%
  select(nodeNumber, depth, isLeaf, label, treatCounter, controlCounter,
         labelStats_0, labelStats_1, treatLabelStats_1, treatLabelStats_0) %>%
  print(n=20)

dim(crf$forest$tree0)
head(colnames(crf$forest$tree0), n=20)
crf$forest$tree0 %>% as_data_frame(.) %>%
  select(nodeNumber, parentNodeNumber, rightChildNodeNumber, leftChildNodeNumber,
         depth, isLeaf, label, counter, bestTest_feature, bestTest_threshold) %>%
  print(n=20)

#0 -> 1 -> 3 -> 5-> -> 7 -> 12
# 19 > -1.25
# 2 > -1.68
# 10 > -0.656
# 17 > 0.228
# 12 > -2.33


crf$forest$tree0 %>% as_data_frame(.) %>%
  select(nodeNumber, depth, isLeaf,
         label, ite_1, treatCounter, controlCounter,
         labelStats_0, labelStats_1, treatLabelStats_1, treatLabelStats_0, controlLabelStats_1, controlLabelStats_0,
         bestTest_feature, bestTest_threshold
         ) %>%
  filter(nodeNumber == 12) %>% t(.)

10/19 #treatment
2/6 #control

10/19 - 2/6 #ite


#### checking the init and train functions
corf_model <- init_orf(numClasses=2, numFeatures=NCOL, numRandomTests=3, 
                       counterThreshold=100, maxDepth=10, numTrees=100, numEpochs=1, 
                       type = 'classification', method = 'gini', causal=TRUE)
  
# names(corf_model)
# dim(corf_model$forest$tree0)
# colnames(corf_model$forest$tree0)

corf_model <- train_corf(corf_model, x=x, y=y, w=w)
names(corf_model)
corf_model$oobe / corf_model$n
dim(corf_model$forest$tree0)

#try training for a single data point
corf_model <- init_orf(numClasses=2, numFeatures=NCOL, numRandomTests=3, 
                       counterThreshold=100, maxDepth=10, numTrees=100, numEpochs=1, 
                       type = 'classification', method = 'gini', causal=TRUE)

npts <- 1
corf_model <- train_corf(corf_model,
                         x=matrix(x[1:npts,], nrow=npts, byrow=TRUE),
                         y=matrix(y[1:npts], nrow=npts),
                         w=matrix(w[1:npts], nrow=npts), 
                        trainModel = FALSE)

i <- 1
for(i in 1:nrow(x)) {
  print(i)
  corf_model <- train_corf(corf_model, 
                           x=matrix(x[i,], nrow=1, byrow=TRUE), 
                           y=matrix(y[i], nrow=1), 
                           w=matrix(w[i], nrow=1))
}
dim(corf_model$forest$tree0)


#### Mimicing testing with GRF package

### grf causal forest example
#install.packages("grf")
library(grf)
## simulate some data
n = 1000; p = 10
X = matrix(rnorm(n*p), n, p); dim(X)
W = rbinom(n, 1, 0.5)
#Y = pmax(X[,1], 0) * W + X[,2] + pmin(X[,3], 0) + rnorm(n)
Z = logistic(pmax(X[,1], 0) * W + X[,2] + pmin(X[,3], 0) + rnorm(n))
Y = ifelse(Z > .5, 1, 0)

table(Y)
tapply(Y, W, mean)

plot(X[,1],Y,col=ifelse(W==1, 'red','black'))
plot(X[,2],Y)
plot(X[,3],Y)
# Train a causal forest.
c.forest = causal_forest(X, Y, W)
# Predict using the forest.
X.test = matrix(0, 101, p)
X.test[,1] = seq(-2, 2, length.out = 101)
c.pred = predict(c.forest, X.test)
# Estimate the conditional average treatment effect on the full sample (CATE).
average_treatment_effect(c.forest, target.sample = "all")
head(c.pred)
names(c.forest)
head(c.forest$predictions)
head(pmax(X[,1], 0))
plot(c.forest$predictions, X[,1])

# Estimate treatment effects for the test sample.
tau.hat = predict(c.forest, X.test)
plot(X.test[,1], tau.hat$predictions, ylim = c(0,1),
     xlab = "x", ylab = "tau", type = "l")
#lines(X.test[,1], pmax(0, X.test[,1]), col = 2, lty = 2)
lines(X.test[,1], logistic(pmax(0, X.test[,1]))-.5, col = 2, lty = 2)



###try with the online version
oc.forest <- causal_online_random_forest(x=X, y=Y, treat=W, numRandomTests = 3,
                                         counterThreshold = 10, trainModel = TRUE,
                                         maxDepth = 50, numTrees = 2000, numEpochs = 1, 
                                         type = "classification", method = "gini", 
                                         causal = TRUE, findTrainError = TRUE, verbose = TRUE)

#make predictions on test sample
tau.hat.online = predict_orf(x=X.test, orfModel = oc.forest)
head(tau.hat.online$ite)
ite_est.online <- tau.hat.online$ite[,"1"]

plot(X.test[,1], tau.hat$predictions, ylim = c(0,1),
     xlab = "x", ylab = "tau", type = "l")
lines(X.test[,1], ite_est.online, col='green')
lines(X.test[,1], logistic(pmax(0, X.test[,1]))-.5, col = 2, lty = 2)
legend('topright',inset=.01, legend=c('Online Causal','Causal','True'),
       col=c('green','black','red'), lty=c(1,1,2))

cor(tau.hat, tau.hat.online$ite[,2])
cbind(tau.hat, tau.hat.online$ite[,2])

rho <- cor(tau.hat, tau.hat.online$ite[,2])
plot(tau.hat$predictions, tau.hat.online$ite[,2], xlab="CausalRF", ylab="Online CausalRF")
abline(a=0,b=1)
text(x=.05,y=.25,paste("Rho=",round(rho,3)))

head(tau.hat)
head(tau.hat.online$ite[,2])

mean(tau.hat$predictions)
mean(tau.hat.online$ite[,2])

mean(c.forest$predictions)

names(oc.forest)
oc.forest.predictions <- predict_orf(X, oc.forest)
mean(oc.forest.predictions$prediction)
head(oc.forest.predictions$confidence)


