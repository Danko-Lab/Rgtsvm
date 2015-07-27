\name{svm}
\alias{svm}
\alias{svm.default}
\alias{svm.formula}
\alias{summary.gtsvm}
\alias{print.summary.gtsvm}
\alias{print.gtsvm}
\title{Support Vector Machines}
\description{
\code{svm} is used to train a support vector machine. It can be used to carry
out general regression and classification (of nu and epsilon-type), as
well as density-estimation. A formula interface is provided. 
}
\usage{
\method{svm}{formula}(formula, data = NULL, ..., subset, na.action =
na.omit, scale = TRUE)

\method{svm}{default}(x, y = NULL, scale = TRUE, type = "C-classification", kernel =
"radial", degree = 3, gamma = if (is.vector(x)) 1 else 1 / ncol(x),
coef0 = 0, cost = 1, tolerance = 0.001, shrinking = TRUE, fitted = TRUE,
..., subset, na.action = na.omit)
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
  \item{type}{\code{svm} can be used as a classification
    machine, as a regression machine, or for novelty detection.
    Depending of whether \code{y} is
    a factor or not, the default setting for \code{type} is \code{C-classification} or \code{eps-regression}, respectively, but may be overwritten by setting an explicit value.\cr
    Valid options are:
    \itemize{
      \item \code{C-classification}
    }
  }
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
  \item{tolerance}{tolerance of termination criterion (default: 0.001)}
  \item{shrinking}{option whether to use the shrinking-heuristics
    (default: \code{TRUE})}
  \item{fitted}{logical indicating whether the fitted values should be computed
    and included in the model or not (default: \code{TRUE})}
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
  An object of class \code{"svm"} containing the fitted model, including:
  \item{SV}{The resulting support vectors (possibly scaled).}
  \item{index}{The index of the resulting support vectors in the data
    matrix. Note that this index refers to the preprocessed data (after
    the possible effect of \code{na.omit} and \code{subset})}
  \item{coefs}{The corresponding coefficients times the training labels.}
}
\details{
  For multiclass-classification with k levels, k>2, \code{libsvm} uses the
  \sQuote{one-against-one}-approach, in which k(k-1)/2 binary classifiers are
  trained; the appropriate class is found by a voting scheme.
  
  \code{libsvm} internally uses a sparse data representation, which is 
  also high-level supported by the package \pkg{SparseM}.
  
  If the predictor variables include factors, the formula interface must be used to get a
  correct model matrix.

  \code{plot.svm} allows a simple graphical
  visualization of classification models.

  The probability model for classification fits a logistic distribution
  using maximum likelihood to the decision values of all binary
  classifiers, and computes the a-posteriori class probabilities for the
  multi-class problem using quadratic optimization. The probabilistic
  regression model assumes (zero-mean) laplace-distributed errors for the
  predictions, and estimates the scale parameter using maximum likelihood.
}
\note{
Data are scaled internally, usually yielding better results.

Parameters of SVM-models usually \emph{must} be tuned to yield sensible results!
}
\references{
  \itemize{
    \item
      Chang, Chih-Chung and Lin, Chih-Jen:\cr
      \emph{LIBSVM: a library for Support Vector Machines}\cr
      \url{http://www.csie.ntu.edu.tw/~cjlin/libsvm}

     \item
      Exact formulations of models, algorithms, etc. can be found in the
      document:\cr
      Chang, Chih-Chung and Lin, Chih-Jen:\cr
      \emph{LIBSVM: a library for Support Vector Machines}\cr
      \url{http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.ps.gz}
    
      \item
      More implementation details and speed benchmarks can be found on:
      Rong-En Fan and Pai-Hsune Chen and Chih-Jen Lin:\cr
      \emph{Working Set Selection Using the Second Order Information for Training SVM}\cr
      \url{http://www.csie.ntu.edu.tw/~cjlin/papers/quadworkset.pdf}
    
  }
}
\author{
  David Meyer (based on C/C++-code by Chih-Chung Chang and Chih-Jen Lin)\cr
  \email{David.Meyer@R-project.org}
}
\seealso{
  \code{\link{predict.svm}}
  \code{\link{plot.svm}}
  \code{\link{tune.svm}}
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
     pch = c("o","+")[1:150 \%in\% model$index + 1])

## try regression mode on two dimensions

# create data
x <- seq(0.1, 5, by = 0.05)
y <- log(x) + rnorm(x, sd = 0.2)

# estimate model and predict input values
m   <- svm(x, y)
new <- predict(m, x)

# visualize
plot(x, y)
points(x, log(x), col = 2)
points(x, new, col = 4)

## density-estimation

# create 2-dim. normal with rho=0:
X <- data.frame(a = rnorm(1000), b = rnorm(1000))
attach(X)

# traditional way:
m <- svm(X, gamma = 0.1)

# formula interface:
m <- svm(~., data = X, gamma = 0.1)
# or:
m <- svm(~ a + b, gamma = 0.1)

# test:
newdata <- data.frame(a = c(0, 4), b = c(0, 4))
predict (m, newdata)

# visualize:
plot(X, col = 1:1000 \%in\% m$index + 1, xlim = c(-5,5), ylim=c(-5,5))
points(newdata, pch = "+", col = 2, cex = 5)

# weights: (example not particularly sensible)
i2 <- iris
levels(i2$Species)[3] <- "versicolor"
summary(i2$Species)
wts <- 100 / table(i2$Species)
wts
m <- svm(Species ~ ., data = i2, class.weights = wts)
}
\keyword{neural}
\keyword{nonlinear}
\keyword{classif}
