\name{predict.unload}
\alias{predict.unload}
\title{Unload SVM object allocated on CUDA-enabled GPU}
\description{
  This function is cleanup step in the phased prediction which is suitable for the multiple predicting calls using a big model.
}
\usage{
predict.unload(object, verbose=FALSE )
}

\arguments{
  \item{object}{an object of class \code{"gtsvm"} returned by \code{\link{predict.load}} in \emph{Rgtsvm} package.}
  \item{verbose}{logical value indicating whether some alogrithm information is output into the R console, default is FALSE. }
}
\value{
  None. The procedure will be interrupted if any errors happen.
   
}
\author{
  Zhong Wang ( R interface & eps-regression in CUDA )  \email{zw355@cornell.edu}\cr
}
\seealso{
  \code{\link{predict.run}}, \code{\link{predict.load}}
}
\keyword{predict}

