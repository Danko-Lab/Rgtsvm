\name{predict.load}
\alias{predict.load}
\title{Load and initialize a training model on CUDA-enabled GPU}
\description{
  This function loads and initializes a SVM model trained by \code{svm} in package \emph{Rgtsvm}. This function works with \code{\link{predict.run}} and \code{\link{predict.unload}} to implement the phased prediction with 3 steps: 
  model loading, multiple predicting call, removing model from GPU. It is suitable for the prediction with huge samples using a big training model. 
}
\usage{
predict.load(object, gpu.id=NULL, verbose=FALSE )
}

\arguments{
  \item{object}{an object of class \code{"gtsvm"} returned by \code{\link{svm}} in \emph{Rgtsvm} package.}
  \item{gpu.id}{a vector indicating multiple GPU device IDs are used for the prediction in parallel.}
  \item{verbose}{logical value indicating whether some alogrithm information is output into the R console, default is FALSE. }
}
\value{
  A slim model with one external pointer returned by the C/C++ functions is returned. The feature vectors and fitting information are removed in order to reduce the memory usage.
  
}
\author{
  Zhong Wang ( R interface & eps-regression in CUDA )  \email{zw355@cornell.edu}\cr
}
\seealso{
  \code{\link{predict.gtsvm}}
}
\keyword{predict}

