\name{svm}
\alias{svm}
\alias{svm.default}
\alias{svm.formula}
\alias{summary.gtsvm}
\alias{print.summary.gtsvm}
\alias{print.gtsvm}
\title{Training a model of SVMs on CUDA-enabled GPU}
\description{
\code{svm} in the \pkg{Rgtsvm} pakcage is used to train a support vector machine by C-classfication or epsilon regression on CUDA-enabled GPU.
}
\usage{
\method{svm}{formula}(formula, data = NULL, ..., subset, na.action = na.omit, scale = TRUE);

\method{svm}{default}(x, 
      y           = NULL,
      scale       = TRUE,
      type        = "C-classification",
      kernel      = "radial",
      degree      = 3,
      gamma       = if (is.vector(x)) 1 else 1/ncol(x),
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
      gpu.id      = NULL,
      maxIter     = NULL,
      verbose     = FALSE,
      ...,
      subset,
      na.action   = na.omit)
}
\arguments{
  \item{formula}{a formula object describing the training model.}
  
  \item{data}{an optional data frame containing the variables in the model. By default the variables are taken from the environment which \sQuote{svm} is called from.}

  \item{x}{a data matrix, or a vector, or a sparse matrix as training data.}

  \item{y}{a factor (for C-classification) or a numeric vector (for eps-regression) specifying the response for each row of \code{x}.}

  \item{scale}{logical value or a logical vector, indicating whether the feature columns are scaled. By default, both \code{x} and \code{y} variables are scaled to zero mean and unit variance. The center and scale values are returned and can be used for scaling new test data.}

  \item{type}{string indicating the SVM method, only \code{C-classification} or \code{eps-regression} available, default is \code{C-classification}}

  \item{kernel}{the kernel function used in training and predicting, four options are available (attached in \code{details}), default is \code{radial}. }

  \item{degree}{integer value indicating parameter value used in kernel of type \code{polynomial}, default is 3.}

  \item{gamma}{numerical value indicating parameter value needed for all kernels except \code{linear}, default is 1/(number of feature vector)). }

  \item{coef0}{numerical value indicating parameter value used in kernels of type \code{polynomial} and \code{sigmoid}, default is 0.}

  \item{cost}{numerical value indicating regularization term in the Lagrange formulation, which is cost of constraints violation, default is 1 .}
  
  \item{class.weights}{a named vector of weights for the different classes, used for asymmetric class sizes. Not all factor levels have to be supplied (default weight: 1). All components have to be named.}

  \item{tolerance}{numerical value indicating the tolerance of termination criterion for the training algorithm, default is 0.001. } 
  
  \item{epsilon}{numerical value indicating epsilon in the insensitive-loss function for the \code{eps-regression} method, default is 0.1. }
  
  \item{shrinking}{logical value indicating whether to use the shrinking-heuristics, default is \code{TRUE}.}
  
  \item{cross}{integer value indicating whether a k-fold cross validation on the training data is performed to assess the quality of the model. Ignored if \code{corss=0}. }

  \item{rough.cross}{integer value which is less than \code{cross}, indicating how many tests are performed for cross-validation. The function will return partial tests for cross-validation rather than all repeated tests in order to reduce the running time. }
  
  \item{fitted}{logical value indicating whether the prediction should be performed and returned in the function calling, default is \code{TRUE}. }

  \item{probability}{logical value indicating whether the model should allow for probability predictions.}
  
  \item{no.change.x}{logical value indicating whther the function can change the \code{x} parameter. It would save CPU memory if this parameter is assigned to \code{FALSE} for the big matrix \code{x}. }
  
  \item{gpu.id}{integer value indicating GPU device ID, starting from 0. No GPU device is changed by default. }

  \item{maxIter}{integer value indicating maximum iteration, default is the number of feature elements (NROW(x)*NCOL(x)) }

  \item{verbose}{logical value indicating whether some alogrithm information is output on R console, default is FALSE. }
  
  \item{\dots}{additional parameters for the low level fitting function \code{svm.default}}
  
  \item{subset}{a vector of index values specifying the cases to be used in the training sample.  (NOTE: If given, this argument must be named.)}

  \item{na.action}{a function to specify the action to be taken if \code{NA}s are found. The default action is \code{na.omit}, which leads to rejection of cases with missing values on any required variable. An alternative is \code{na.fail}, which causes an error if \code{NA} cases are found. (NOTE: If given, this argument must be named.)}	
}

\value{
  This function returns a trained model. It is an object of class \code{"gtsvm"} with following properties:

  \item{total.nSV}{the total number of support vectors.}
  \item{nSV}{the number of support vectors in each group.}
  \item{SV}{the resulting support vectors (possibly scaled).}
  \item{index}{the index of the resulting support vectors in the data matrix. Note that this index refers to the preprocessed data (after the possible effect of \code{na.omit} and \code{subset})}
  \item{coefs}{the corresponding coefficients multiply the training labels.}
  \item{rho}{the negative intercept.}

  \item{fitted}{the prediction values if \code{fitted=TRUE}}
  \item{fitted.accuracy}{The predicted accuracy for \code{C-classification} if \code{fitted=TRUE}.}
  \item{fitted.MSE}{the mean square error for \code{eps-regression} if \code{fitted=TRUE}.}
  \item{fitted.r2}{the R square for \code{eps-regression} if \code{fitted=TRUE}.}
  \item{residuals}{the difference between the true values and prediction for \code{eps-regression} if \code{fitted=TRUE}.}

  \item{MSE}{the mean square errors at each cross-validation test for \code{eps-regression}  if \code{cross>0}.}
  \item{tot.MSE}{the mean square error in cross-validation for \code{eps-regression}  if \code{cross>0}.}
  \item{scorrcoeff}{the R square in cross-validation for \code{eps-regression} if \code{cross>0}.}
  \item{accuracies}{the accuracies at each cross-validation test for \code{C-classification} if \code{cross>0}.}
  \item{tot.accuracy}{the total accuracies in cross-validation for \code{C-classification} if \code{cross>0}.}

  \item{compprob}{logical values if probability is predicted.}
  \item{probA}{numerical value indicating (1) the parameter A of the logistic distributions fitted to the decision values of the binary classifiers 
  (2) the \eqn{\gamma} coefficient of the softmax function for the multi-classification in one-agianst-rest mode. (3) the \eqn{\zeta} parameter of the Laplace distributions for the eps-regression  if \code{probability==TRUE}.}
  \item{probB}{numerical value indicating the parameter B of the logistic distributions fitted to the decision values of the binary classifiers if \code{probability==TRUE}. }
  
}

\details{
  \pkg{Rgtsvm} uses a sparse matrix and regular matrix. The sparse matrix can be defined by the class \code{\link[Matrix]{Matrix}} provided by the \pkg{Matrix} package, 
    or the class \code{\link[SparseM]{matrix.csr}} provided by the \pkg{SparseM} package, 
    or the class \code{\link[slam]{simple_triplet_matrix}} provided by the \pkg{slam} package) \cr
    
   The kernel function has the following parameters, depending on the kernel type.\cr
    \describe{
      \item{linear:}{\eqn{u'v}{u'*v}}
      \item{polynomial:}{\eqn{(\gamma u'v + coef0)^{degree}}{(gamma*u'*v + coef0)^degree}}
      \item{radial basis:}{\eqn{e^(-\gamma |u-v|^2)}{exp(-gamma*|u-v|^2)}}
      \item{sigmoid:}{\eqn{tanh(\gamma u'v + coef0)}{tanh(gamma*u'*v + coef0)}}
      }

  \code{\link{plot.gtsvm}} provides a simple visualization method for the classification model trained by \code{\link{svm}}.
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
  Andrew Cotter, Nathan Srebro ,Joseph Keshet ( C/C++ code in CUDA ) \cr
  \href{http://ttic.uchicago.edu/~cotter/projects/gtsvm/}{http://ttic.uchicago.edu/~cotter/projects/gtsvm/} \cr
}
\seealso{
  \code{\link{predict.gtsvm}}
  \code{\link{plot.gtsvm}}
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
     
# Check accuracy:
table(pred, y)
     
# compute decision values and probabilities:
pred <- predict(model, x, decision.values = TRUE)
attr(pred, "decision.values")[1:4,]
     
# visualize (classes by color, SV by crosses):
plot(cmdscale(dist(iris[,-5])),
     col = as.integer(iris[,5]),
     pch = c("o","+")[1:150 \%in\% model$index + 1])
     
}
\keyword{svm}
