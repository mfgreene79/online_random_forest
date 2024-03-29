% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/orf_functions.R
\name{corf}
\alias{corf}
\title{Causal Online Random Forest}
\usage{
corf(x, y, treat, orfModel, trainModel = TRUE)
}
\arguments{
\item{x}{A matrix of features on which to generate predictions.}

\item{y}{Matrix of classes.  Multiclass classification is supported.  Must be integers and a matrix.}

\item{treat}{Matrix of treatment assignments. Must be values 0 or 1 and a matrix.}

\item{orfModel}{List representing an orf model.}

\item{trainModel}{Should the forest be trained on the new data?
Defaults to TRUE.  Useful for testing.}
}
\description{
Workhorse function used to fit a causal online random forest object.  Use the functions
\code{\link{init_orf}} and \code{\link{train_orf}} or \code{\link{causal_online_random_forest}} 
instead.
}
\seealso{
\code{\link{init_orf}} for initializing the orf object, \code{\link{train_orf}} for training the orf object
}
