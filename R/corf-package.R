#' corf-package: A package for using causal online random forests.
#'
#' The Causal Online Random Forest is an application of the <a
#' href="../grf/index.html">Wager and Athey causal forest</a> to an Online
#' Random Forest able to grow a forest of trees incrementally from streams of
#' data.  The algorithm was adapted from <a
#' href="https://github.com/amirsaffari/online-multiclass-lpboost">Amir
#' Saffari's libraries in C++</a>.  
#'
#' @section CORF functions: 
#'
#' @docType package
#' @name init_orf
#' @name train_orf
#' @name predict_orf
#' @name causal_online_random_forest
#' @name online_random_forest
#' @name causal_orf_cv
#' @name orf_cv
#' @name get_importance