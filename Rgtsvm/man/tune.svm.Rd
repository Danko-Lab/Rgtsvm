\name{tune.svm}
\alias{tune.svm}
\alias{best.tune}
\alias{print.tune}
\alias{summary.tune}
\alias{print.summary.tune}
\title{Parameter tuning function using grid search}
\description{
  \code{tune.svm()} tunes hyperparameters of statistical methods using a grid search over supplied parameter ranges.
  \code{best.tune()} returns the best model detected by \code{tune.svm}.
}
\usage{
tune.svm(x, y = NULL, data = NULL, degree = NULL, gamma = NULL,
      coef0 = NULL, cost = NULL, nu = NULL, 
      class.weights = NULL, epsilon = NULL,...)
best.tune(...)
}
\arguments{
  \item{x}{formula object or training matrix.}
  \item{y}{vector indicating the response variable only if \code{train.x} is not a formula object, otherwise ignore.  }
  \item{data}{data frame only if the formula interface is used in \code{train.x}, otherwise ignore. }
  \item{degree}{a numeric vector indicating the tuning values for the parameter \code{degree}.}
  \item{gamma}{a numeric vector indicating the tuning values for the parameter \code{gamma}.}
  \item{coef0}{an numeric vector indicating the tuning values for the parameter \code{coef0}.}
  \item{cost}{a numeric vector indicating the tuning values for the parameter \code{cost}.}
  \item{nu}{a numeric vector indicating the tuning values for the parameter \code{nu}.}
  \item{class.weights}{a numeric vector indicating the tuning values for the parameter \code{class.weights}.}
  \item{epsilon}{a numeric vector indicating the tuning values for the parameter \code{epsilon}.}
  \item{\dots}{Further parameters passed to the training functions, i.e. \code{\link{svm}}, \code{tune}.}
}
\value{
  This function return an object of class \code{tune}, including the components:
  \item{best.parameters}{a 1 x k data frame, k number of parameters.}
  \item{best.performance}{value, best achieved performance.}
  \item{performances}{if requested, a data frame of all parameter combinations along with the corresponding performance results.}
  \item{train.ind}{list of index vectors used for splits into training and validation sets.}
  \item{best.model}{if requested, the model trained on the complete training data using the best parameter combination.}
}
\details{
  To measure performance, classification error and mean squared error can be used for the classificaition and epsilon regression respectively. 
}
\author{
  David Meyer\cr
  \email{David.Meyer@R-project.org}
}
\seealso{\code{\link{tune.control}}, \code{\link{plot.tune}} }

\examples{
  data(iris)

  ## tune `svm' for classification with RBF-kernel (default in svm),
  ## using one split for training/validation set
  obj <- tune.svm(Species~., data = iris, gamma = 2^(-1:1), cost = 2^(2:4))

  summary(obj)
  plot(obj)
}
\keyword{tuning}

