\name{predict.gtsvm}
\alias{predict.gtsvm}
\title{Predict Method for Support Vector Machines}
\description{
  This function predicts values based upon a model trained by \code{svm} in package \emph{Rgtsvm}.
}
\usage{
\method{predict}{gtsvm}(object, newdata, 
    decision.values = FALSE,
    probability = FALSE,
    verbose=FALSE, 
    ..., 
    na.action = na.omit)
}

\arguments{
  \item{object}{Object of class \code{"gtsvm"}, created by \code{svm} in \emph{Rgtsvm} package.}
  \item{newdata}{An object containing the new input data: either a  matrix or a sparse matrix (object of class
     \code{\link[Matrix]{Matrix}} provided by the \pkg{Matrix} package, or of class \code{\link[SparseM]{matrix.csr}}
     provided by the \pkg{SparseM} package, or of class \code{\link[slam]{simple_triplet_matrix}} provided by the \pkg{slam}
     package). A vector will be transformed to a n x 1 matrix.}
  \item{decision.values }{Logical controlling whether the decision values
    of all binary classifiers computed in multiclass classification
    shall be computed and returned.}
  \item{probability}{Logical indicating whether class probabilities
    should be computed and returned. Only possible if the model was
    fitted with the \code{probability} option enabled.}
  \item{verbose}{logical value indicating whether some alogrithm information(default:FALSE)}
  \item{\dots}{Currently not used.}
  \item{na.action}{A function to specify the action to be taken if \sQuote{NA}s are
    found. The default action is \code{na.omit}, which leads to rejection of cases
    with missing values on any required variable. An alternative
    is \code{na.fail}, which causes an error if \code{NA} cases
    are found. (NOTE: If given, this argument must be named.)}	
}
\value{
  A vector of predicted values (for classification: a vector of labels, for density
  estimation: a logical vector). If \code{score} is
  \code{TRUE}, the vector gets a \code{"decision.values"} attribute
  containing a n x c matrix (n number of predicted values, c number of
  classifiers) of all c binary classifiers' decision values.
}
\note{
  If the training set was scaled by \code{svm} in \emph{Rgtsvm}, the
  new data is scaled accordingly using scale and center of
  the training data.
}
\author{
  Zhong Wang ( R interface & epe-regression in CUDA )  \email{zw355@cornell.edu}\cr
  David Meyer ( R interface in e1071)  \email{David.Meyer@R-project.org} \cr
  Andrew Cotter, Nathan Srebro ,Joseph Keshet ( C/C++ code in CUDA )  \link{http://ttic.uchicago.edu/~cotter/projects/gtsvm/} \cr
}
\seealso{
  \code{\link{svm}}
}
\keyword{classification}

