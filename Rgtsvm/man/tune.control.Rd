\name{tune.control}
\alias{tune.control}
\title{Control parameters for the tune function}
\description{
  Creates an object of class \code{tune.control} to be used with
  the \code{tune.svm} function, containing various control parameters.
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
      rough.cross = 0,
      error.fun = NULL)
}
\arguments{
  \item{random}{\code{FALSE} or an integer value, indicating whether or how many parameter combinations are drawn from the parameter space.}
  \item{nrepeat}{integer value, specifies how many replication of training and prediction shall be repeated, default is 1.}
  \item{repeat.aggregate}{function, used to aggregate the replicates results , default is \code{\link{mean}}.}
  \item{sampling}{string value indicating sampling scheme, three sampling methods are available, default is \code{cross}.  }
  \item{sampling.aggregate}{function to aggregate the test results for each parameter combination, default is \code{\link{mean}}. }
  \item{sampling.dispersion}{function to disperse the test results for each parameter combination, default is \code{\link{sd}}. }
  \item{cross}{integer value, used if \code{sampling = "cross"}, indicating the number of partitions for cross-validation, default is 10.}
  \item{fix}{numeric value, used if \code{sampling = "fix"},  part of the data used for training in fixed sampling., default is 2/3}
  \item{nboot}{integer value, used if  \code{sampling = "boot"}, indicating number of bootstrap replications.}
  \item{boot.size}{numeric value, used if  \code{sampling = "boot"}, indicating size of the bootstrap samples.}
  \item{best.model}{logical value, indicating whether the best model is returned based on the the best parameter set on the complete training set).}
  \item{performances}{logical value, indicating whether the performance results for all parameter combinations are returned.}
  \item{rough.cross}{integer value if \code{sampling = "cross"}, indicating how many training and testing for cross-validation are conducted, it is intended to reduce the computation time by decreasing the cross-validation jobs while maintaining the higher \code{cross} number.}
  \item{error.fun}{function for the error measure, 
    It takes two arguments: a vector of true values and a vector of
    predicted values. If \code{NULL}, the misclassification error is used
    for categorical predictions and the mean squared error for numeric
    predictions.}
}
\details{
  Three options for sampling are available:\cr
  
  1: \code{sampling = "cross"}, a \code{cross}-times cross validation. \cr
  
  2: \code{sampling = "boot"}, sampling with replacement on the \code{nboot} training sets of \code{boot.size} (part). \cr
  
  3: \code{sampling = "fix"}, the dataset is split into a training set of size \code{fix} and validation set. \cr
  
}
\value{
  An object of class \code{"tune.control"} containing all the above parameters (either the defaults or the user specified values).
}
\author{
  David Meyer\cr
  \email{David.Meyer@R-project.org}
}
\seealso{\code{\link{tune.svm}}}
\keyword{tuning}
