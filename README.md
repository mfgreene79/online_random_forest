# online_random_forest
Machine learning models for incremental learning.  Implemented in R and C++

This project utilizes incremental machine learning techniques (i.e., online or out-of-core algorithms) to partially fit coefficients given one datapoint at a time.  The project leans on code from Amir Saffiri (https://github.com/amirsaffari/online-multiclass-lpboost).  This project extends the algorithms by making them available in R via the Rcpp and RcppEigen packages.  This implemntation allows for the models to learn one data point at a time and save the models as lists of parameters for persistence to future sessions.

