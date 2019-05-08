# online_random_forest
Machine learning models for incremental learning random forest models.  Implemented as an R package using Rcpp and C++.  Includes versions of an causal random forest that learns incrementally.  

This project utilizes incremental machine learning techniques (i.e., online or out-of-core algorithms) to partially fit models given one datapoint at a time.  The project leverages code from Amir Saffiri (https://github.com/amirsaffari/online-multiclass-lpboost).  This project extends the algorithms by making them available in R via the Rcpp and RcppEigen packages and adds causal versions of the algorithms (see Wager and Athey 2018).  This implemntation allows for the models to learn one data point at a time and save the models as lists of parameters for persistence to future sessions in R.

References:
Athey, Susan, and Guido W. Imbens. "Identification and inference in nonlinear difference‐in‐differences models." Econometrica 74.2 (2006): 431-497.

Athey, Susan, and Guido Imbens. "Recursive partitioning for heterogeneous causal effects." Proceedings of the National Academy of Sciences 113.27 (2016): 7353-7360.

Saffari, Amir, et al. "On-line random forests." 2009 ieee 12th international conference on computer vision workshops, iccv workshops. IEEE, 2009.

Wager, Stefan, and Susan Athey. "Estimation and inference of heterogeneous treatment effects using random forests." Journal of the American Statistical Association 113.523 (2018): 1228-1242.

