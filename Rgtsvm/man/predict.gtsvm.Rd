\name{predict.gtsvm}
\alias{predict.gtsvm}
\title{Predict method for SVMs on CUDA-enabled GPU}
\description{
  This function performs prediction based on a SVM model trained by \code{svm} in package \emph{Rgtsvm}.
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
  \item{object}{an object of class \code{"gtsvm"} returned by \code{\link{svm}} in \emph{Rgtsvm} package.}
  \item{newdata}{data frame, or matrix, or sparse matrix of the test data. A vector test data must transform to a n x 1 matrix.}
  \item{decision.values }{logical value indicating whether the decision values of binary classfication or multiclass classification shall be returned. Only valid for classfication.}
  \item{probability}{logical value indicating whether class probabilities should be computed and returned.}
  \item{verbose}{logical value indicating whether some alogrithm information is output into the R console, default is FALSE. }
  \item{\dots}{Unused currently.}
  \item{na.action}{a function to specify the action to be taken if \code{NA}s are found. The default action is \code{na.omit}, which leads to rejection of cases with missing values on any required variable. An alternative is \code{na.fail}, which causes an error if \code{NA} cases are found. (NOTE: If given, this argument must be named.}	
}
\value{
  A vector of predicted values or labels are returned.

  If \code{decision.values} is required, the vector has a \code{"decision.values"} attribute containing a decision matrix with number of samples in rows and number of classes and in columns.
  
  If \code{probability} is required, the vector has a \code{"probabilities"} attribute containing a decision matrix with number of samples in rows and number of classes and in columns, only valid for \code{C-classification}.  
   
}
\note{
  The test data can be an object containing the new input data: data frame, or matrix,  or a sparse matrix. 
  The sparse matrix can be defined by the class \code{\link[Matrix]{Matrix}} provided by the \pkg{Matrix} package, 
    or the class \code{\link[SparseM]{matrix.csr}} provided by the \pkg{SparseM} package, 
    or the class \code{\link[slam]{simple_triplet_matrix}} provided by the \pkg{slam} package)

  If the training data was scaled by the \code{\link{svm}} calling, this function shall scale the test data accordingly using scale and center of the training data.

}
\author{
  Zhong Wang ( R interface & eps-regression in CUDA )  \email{zw355@cornell.edu}\cr
  David Meyer ( R interface in e1071)  \email{David.Meyer@R-project.org} \cr
  Andrew Cotter, Nathan Srebro ,Joseph Keshet ( C/C++ code in CUDA )  \cr
  \href{http://ttic.uchicago.edu/~cotter/projects/gtsvm/}{http://ttic.uchicago.edu/~cotter/projects/gtsvm/} \cr
}
\seealso{
  \code{\link{svm}}
}
\keyword{predict}

