\name{tune.control}
\alias{tune.control}
\title{Control Parameters for the Tune Function}
\description{
  Creates an object of class \code{tune.control} to be used with
  the \code{tune} function, containing various control parameters.
}
\usage{
tune.control(random = FALSE, 
      nrepeat = 1, 
      repeat.aggregate = mean,
      sampling = c("cross", "fix", "bootstrap"), 
      sampling.aggregate = mean,
      sampling.dispersion = sd,
      cross = 10, 
      fix = 2/3, 
      nboot = 10, 
      boot.size = 9/10, 
      best.model = TRUE,
      performances = TRUE, 
      error.fun = NULL)
}
\arguments{
  \item{random}{if an integer value is specified, \code{random}
    parameter vectors are drawn from the parameter space.}
  \item{nrepeat}{specifies how often training shall be repeated.}
  \item{repeat.aggregate}{function for aggregating the repeated training results.}
  \item{sampling}{sampling scheme. If \code{sampling = "cross"}, a
    \code{cross}-times cross validation is performed. If \code{sampling
      = "boot"}, \code{nboot} training sets of size \code{boot.size} (part)
    are sampled (with replacement) from the supplied data. If \code{sampling
      = "fix"}, a single split into training/validation set is
    used, the training set containing a \code{fix} part of the supplied
    data. Note that a separate validation set can be supplied via
    \code{validation.x} and \code{validation.y}. It is only used for
    \code{sampling = "boot"} and \code{sampling = "fix"}; in the latter
    case, \code{fix} is set to 1.}
  \item{sampling.aggregate,sampling.dispersion}{functions for aggregating the training
    results on the generated training samples (default: mean and
    standard deviation).}
  \item{cross}{number of partitions for cross-validation.}
  \item{fix}{part of the data used for training in fixed sampling.}
  \item{nboot}{number of bootstrap replications.}
  \item{boot.size}{size of the bootstrap samples.}
  \item{best.model}{if \code{TRUE}, the best model is trained and
    returned (the best parameter set is used for
    training on the complete training set).}
  \item{performances}{if \code{TRUE}, the performance results for all
    parameter combinations are returned.}
  \item{error.fun}{function returning the error measure to be minimized.
    It takes two arguments: a vector of true values and a vector of
    predicted values. If \code{NULL}, the misclassification error is used
    for categorical predictions and the mean squared error for numeric
    predictions.}
}
\value{
  An object of class \code{"tune.control"} containing all the above
  parameters (either the defaults or the user specified values).
}
\author{
  David Meyer\cr
  \email{David.Meyer@R-project.org}
}
\seealso{\code{\link{tune}}}
\keyword{models}
