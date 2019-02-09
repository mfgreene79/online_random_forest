########### example_call_orf.R ################

### this program is an example call of the online random forest from simulated data for classification

### load any libraries that we need here ###
#library(ourlibrary)


######### SIMULATE DATA - note: used in analyzing real data this would mean importing a dataset ###########
### simulate some predictors - 
NCOL <- 10 #number of columns 
NROW <- 100 #nbumber of rows

betas <- rnorm(NCOL) #simulate coefficients for the regression
x <- matrix(data=rnorm(NCOL * NROW), ncol=NCOL, nrow=NROW)

### simulate the dependent variable related to the predictors
z <- x %*% betas
y <- ifelse(z > 0, 1, 0)

hist(z) #histogram of z
table(y) #distribtuion of the dependent variable

########## Done simulating data ############


############## Sample Usage 1: run through to update model for all data points at the same time #############

########### Initialize the model object

orf_model = initialize_orf(ntrees=1000, n_features = 3) 
#question: what other parameters are needed to initialize the random forest?


########### pass all data into the model object to fit it
orf_model = fit_orf(model=orf_model, x=X, y=y)
#question: are there other parameters needed for fitting the random forest?
#orf_model should now contain a random forest model with trees and other information

########### we should now be able to do something like predict with it:
orf_predictions = predict(model=orf_model, newdata = X_new) #assuming that we have a new X dataset


############## Sample Usage 2: run through to update model one data point at a time #############

########### Initialize the model object - this doesn't change

orf_model = initialize_orf(ntrees=1000, n_features = 3) 
#question: what other parameters are needed to initialize the random forest?


########### pass all data into the model object to fit it
for(i in 1:nrow(X)) {
  orf_model = fit_orf(model=orf_model, x=X[i,], y=y[i])
}
#question: are there other parameters needed for fitting the random forest?
#orf_model should now contain a random forest model with trees and other information



# sourceCpp("rcpp_eigen_test.cpp")
# m = matrix(rnorm(100, 0, 1), ncol=10)
# getEigenValues(m)

