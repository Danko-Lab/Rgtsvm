\name{tune}
\alias{tune}
\alias{best.tune}
\alias{print.tune}
\alias{summary.tune}
\alias{print.summary.tune}
\title{Parameter Tuning of Functions Using Grid Search}
\description{
  This generic function tunes hyperparameters of statistical methods
  using a grid search over supplied parameter ranges.
}
\usage{
tune.svm(method, train.x, train.y = NULL, data = list(), validation.x =
     NULL, validation.y = NULL, ranges = NULL, predict.func = predict,
     tunecontrol = tune.control(), ...)
best.tune(...)
}
\arguments{
  \item{method}{either the function to be tuned, or a character string
    naming such a function.}
  \item{train.x}{either a formula or a matrix of predictors.}
  \item{train.y}{the response variable if \code{train.x} is a predictor
    matrix. Ignored if \code{train.x} is a formula.}
  \item{data}{data, if a formula interface is used. Ignored, if
    predictor matrix and response are supplied directly.}
  \item{validation.x}{an optional validation set. Depending on whether a
    formula interface is used or not, the response can be
    included in \code{validation.x} or separately specified using \code{validation.y}.}
  \item{validation.y}{if no formula interface is used, the response of
    the (optional) validation set.}
  \item{ranges}{a named list of parameter vectors spanning the sampling
    space. The vectors will usually be created by \code{seq}.}
  \item{predict.func}{optional predict function, if the standard \code{predict}
    behavior is inadequate.}
  \item{tunecontrol}{object of class \code{"tune.control"}, as created by the
    function \code{tune.control()}. If omitted, \code{tune.control()}
    gives the defaults.}
  \item{\dots}{Further parameters passed to the training functions.}
}
\value{
  For \code{tune}, an object of class \code{tune}, including the components:
  \item{best.parameters}{a 1 x k data frame, k number of parameters.}
  \item{best.performance}{best achieved performance.}
  \item{performances}{if requested, a data frame of all parameter
    combinations along with the corresponding performance results.}
  \item{train.ind}{list of index vectors used for splits into
    training and validation sets.}
  \item{best.model}{if requested, the model trained on the complete training data
    using the best parameter combination.}
  
  \code{best.tune()} returns the best model detected by \code{tune}.
}
\details{
  As performance measure, the classification error is used
  for classification, and the mean squared error for regression. It is
  possible to specify only one parameter combination (i.e., vectors of
  length 1) to obtain an error estimation of the specified type
  (bootstrap, cross-classification, etc.) on the given data set. For
  convenience, there
  are several \code{tune.foo()} wrappers defined, e.g., for
  \code{nnet()}, \code{randomForest()},
  \code{rpart()}, \code{svm()}, and \code{knn()}.

  Cross-validation randomizes the data set before building the splits
  which---once created---remain constant during the training
  process. The splits can be recovered through the \code{train.ind}
  component of the returned object.
}
\author{
  David Meyer\cr
  \email{David.Meyer@R-project.org}
}
\seealso{\code{\link{tune.control}}, \code{\link{plot.tune}},
  \code{\link{tune.svm}}, \link{tune.wrapper}}
\examples{
  data(iris)
  ## tune `svm' for classification with RBF-kernel (default in svm),
  ## using one split for training/validation set
  
  obj <- tune(svm, Species~., data = iris, 
              ranges = list(gamma = 2^(-1:1), cost = 2^(2:4)),
              tunecontrol = tune.control(sampling = "fix")
             )

  ## alternatively:
  ## obj <- tune.svm(Species~., data = iris, gamma = 2^(-1:1), cost = 2^(2:4))

  summary(obj)
  plot(obj)

  ## tune `knn' using a convenience function; this time with the
  ## conventional interface and bootstrap sampling:
  x <- iris[,-5]
  y <- iris[,5]
  obj2 <- tune.knn(x, y, k = 1:5, tunecontrol = tune.control(sampling = "boot"))
  summary(obj2)
  plot(obj2)

  ## tune `rpart' for regression, using 10-fold cross validation (default)
  data(mtcars)
  obj3 <- tune.rpart(mpg~., data = mtcars, minsplit = c(5,10,15))
  summary(obj3)
  plot(obj3)

  ## simple error estimation for lm using 10-fold cross validation
  tune(lm, mpg~., data = mtcars)
}
\keyword{models}

