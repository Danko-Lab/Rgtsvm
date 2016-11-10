\name{svm}
\alias{svm}
\alias{svm.default}
\alias{svm.formula}
\alias{summary.gtsvm}
\alias{print.summary.gtsvm}
\alias{print.gtsvm}
\title{Training a model of Support Vector Machines by GPU}
\description{
\code{svm} in \emph{Rgtsvm} pakcage is used to train a support vector machine by the C-classfication and epsilon regression method.  A formula interface is provided. 
}
\usage{
\method{svm}{formula}(formula, data = NULL, ..., na.action = na.omit, scale = TRUE);

\method{svm}{default}(x, 
      y           = NULL,
      scale       = TRUE,
      type        = "C-classification",
      kernel      = "radial",
      degree      = 3,
      gamma       = 0.05,
      coef0       = 0,
      cost        = 1,
      class.weights= NULL,
      tolerance   = 0.001,
      epsilon     = 0.1,
      shrinking   = TRUE,
      cross       = 0,
      probability = FALSE,
      fitted      = TRUE,
      rough.cross = 0,
      no.change.x = TRUE,
      verbose     = FALSE,
      ...,
      subset,
      na.action = na.omit)
}
\arguments{
  \item{formula}{a symbolic description of the model to be fit.}
  \item{data}{an optional data frame containing the variables in the model.
          By default the variables are taken from the environment which
          \sQuote{svm} is called from.}
  \item{x}{a data matrix, a vector, or a sparse matrix (object of class
    \code{\link[Matrix]{Matrix}} provided by the \pkg{Matrix} package,
    or of class \code{\link[SparseM]{matrix.csr}}
    provided by the \pkg{SparseM} package, or of class
    \code{\link[slam]{simple_triplet_matrix}} provided by the \pkg{slam}
    package).}
  \item{y}{a response vector with one label for each row/component of
    \code{x}. Can be either a factor (for classification tasks)
    or a numeric vector (for regression).}
  \item{scale}{A logical vector indicating the variables to be
    scaled. If \code{scale} is of length 1, the value is recycled as
    many times as needed.
    Per default, data are scaled internally (both \code{x} and \code{y}
    variables) to zero mean and unit variance. The center and scale
    values are returned and used for later predictions.}
  \item{type}{only \code{C-classification} or \code{eps-regression} available. }
  \item{kernel}{the kernel used in training and predicting. You
    might consider changing some of the following parameters, depending
    on the kernel type.\cr
    \describe{
      \item{linear:}{\eqn{u'v}{u'*v}}
      \item{polynomial:}{\eqn{(\gamma u'v + coef0)^{degree}}{(gamma*u'*v + coef0)^degree}}
      \item{radial basis:}{\eqn{e^(-\gamma |u-v|^2)}{exp(-gamma*|u-v|^2)}}
      \item{sigmoid:}{\eqn{tanh(\gamma u'v + coef0)}{tanh(gamma*u'*v + coef0)}}
      }
    }
  \item{degree}{parameter needed for kernel of type \code{polynomial} (default: 3)}
  \item{gamma}{parameter needed for all kernels except \code{linear}
    (default: 1/(data dimension))}
  \item{coef0}{parameter needed for kernels of type \code{polynomial}
    and \code{sigmoid} (default: 0)}
  \item{cost}{cost of constraints violation (default: 1)---it is the
    \sQuote{C}-constant of the regularization term in the Lagrange formulation.}
  \item{class.weights}{a named vector of weights for the different
    classes, used for asymmetric class sizes. Not all factor levels have
    to be supplied (default weight: 1). All components have to be named.}
  \item{tolerance}{tolerance of termination criterion (default: 0.001)}
  \item{epsilon}{epsilon in the insensitive-loss function (default: 0.1)}
  \item{shrinking}{option whether to use the shrinking-heuristics
    (default: \code{TRUE})}
  \item{cross}{if a integer value k>0 is specified, a k-fold cross
    validation on the training data is performed to assess the quality
    of the model: the accuracy rate for classification and the Mean
    Squared Error for regression}
  \item{fitted}{logical indicating whether the fitted values should be computed
    and included in the model or not (default: \code{TRUE})}
  \item{probability}{logical indicating whether the model should
    allow for probability predictions.}
  \item{rough.cross}{Number which is less than \code{cross}, indicating how many tests are performed for cross-validation. The function will return partial tests for cross-validation rather than all repeated tests in order to reduce the running time. }
  \item{no.change.x}{Logical value indicating whther tne function can change the \code{x} parameter. If \code{x} is a big matrix, it would be save emeory to use this parameter \code{FALSE}. }
  \item{verbose}{logical value indicating whether some alogrithm information(default:FALSE)}
  \item{\dots}{additional parameters for the low level fitting function
    \code{svm.default}}
  \item{subset}{An index vector specifying the cases to be used in the
          training sample.  (NOTE: If given, this argument must be
          named.)}
  \item{na.action}{A function to specify the action to be taken if \code{NA}s are
          found. The default action is \code{na.omit}, which leads to rejection of cases
          with missing values on any required variable. An alternative
	  is \code{na.fail}, which causes an error if \code{NA} cases
	  are found. (NOTE: If given, this argument must be named.)}	
}

\value{
  An object of class \code{"gtsvm"} containing the fitted model, including:
  \item{SV}{The resulting support vectors (possibly scaled).}
  \item{index}{The index of the resulting support vectors in the data
    matrix. Note that this index refers to the preprocessed data (after
    the possible effect of \code{na.omit} and \code{subset})}
  \item{coefs}{The corresponding coefficients times the training labels.}
}
\details{
  \emph{Rgtsvm} internally uses a sparse matrix and regular matrix.
  
  If the predictor variables include factors, the formula interface must be used to get a
  correct model matrix.

  \code{plot.gtsvm} allows a simple graphical visualization of classification models.
}
\references{
  \itemize{
    \item
     Andrew Cotter, Nathan Srebro, Joseph Keshet. "A GPU-Tailored Approach for Training Kernelized SVMs". 17th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2011.

    \item
      Chang, Chih-Chung and Lin, Chih-Jen:\cr
      \emph{LIBSVM: a library for Support Vector Machines}\cr
      \url{http://www.csie.ntu.edu.tw/~cjlin/libsvm}
  }
}
\author{
  Zhong Wang ( R interface & epe-regression in CUDA )  \email{zw355@cornell.edu}\cr
  David Meyer ( R interface in e1071)  \email{David.Meyer@R-project.org} \cr
  Andrew Cotter, Nathan Srebro ,Joseph Keshet ( C/C++ code in CUDA )  \link{http://ttic.uchicago.edu/~cotter/projects/gtsvm/} \cr
}
\seealso{
  \code{\link{predict.gtsvm}}
  \code{plot.gtsvm}
  \code{\link[SparseM]{matrix.csr}} (in package \pkg{SparseM})
}
\examples{

data(iris)
attach(iris)
     
## classification mode
# default with factor response:
model <- svm(Species ~ ., data = iris)
     
# alternatively the traditional interface:
x <- subset(iris, select = -Species)
y <- Species
model <- svm(x, y) 
     
print(model)
summary(model)
     
# test with train data
pred <- predict(model, x)
# (same as:)
pred <- fitted(model)
     
# Check accuracy:
table(pred, y)
     
# compute decision values and probabilities:
pred <- predict(model, x, decision.values = TRUE)
attr(pred, "decision.values")[1:4,]
     
# visualize (classes by color, SV by crosses):
plot(cmdscale(dist(iris[,-5])),
     col = as.integer(iris[,5]),
     pch = c("o","+")[1:150 %in% model$index + 1])
     
}
\keyword{classification}
