\name{predict.run}
\alias{predict.run}
\title{Run prediction using a loaded model on CUDA-enabled GPU}
\description{
  This function performs the phased prediction using 3 steps, which are model loading, prediction based on model, and model unloading. This prediction step can be called by multiple times. 
}
\usage{
predict.run(object, newdata, 
      decision.values = FALSE,
      probability = FALSE,
      verbose=FALSE, 
      ..., 
      na.action = na.omit)
}

\arguments{
  \item{object}{an object of class \code{"gtsvm"} returned by \code{\link{predict.load}} in \emph{Rgtsvm} package.}
  \item{newdata}{data frame, or matrix, or sparse matrix of the test data. A vector test data must transform to a n x 1 matrix.}
  \item{decision.values }{logical value indicating whether the decision values of binary classfication or multiclass classification shall be returned. Only valid for classfication.}
  \item{probability}{logical value indicating whether class probabilities should be computed and returned.}
  \item{verbose}{logical value indicating whether some alogrithm information is output into the R console, default is FALSE. }
  \item{\dots}{Unused currently.}
  \item{na.action}{a function to specify the action to be taken if \code{NA}s are found. The default action is \code{na.omit}, which leads to rejection of cases with missing values on any required variable. An alternative is \code{na.fail}, which causes an error if \code{NA} cases are found. (NOTE: If given, this argument must be named.}	
}
\value{
  A vector of predicted values or labels are returned. The structue is same as \code{\link{predict.gtsvm}}.

  If \code{decision.values} is required, the vector has a \code{"decision.values"} attribute containing a decision matrix with number of samples in rows and number of classes and in columns.
  
  If \code{probability} is required, the vector has a \code{"probabilities"} attribute containing a decision matrix with number of samples in rows and number of classes and in columns, only valid for \code{C-classification}.  
   
}
\note{
  See \code{\link{predict.gtsvm}}
}
\author{
  Zhong Wang ( R interface & eps-regression in CUDA )  \email{zw355@cornell.edu}\cr
}
\examples{

data(iris)
attach(iris)
     
## classification training
model <- svm(Species ~ ., data = iris)

object <- predict.load(model);
r <- predict.run(object, newdata = iris, decision.values = TRUE )
predict.unload(object);

show(r);
  
}
\seealso{
  \code{\link{predict.load}, \link{predict.unload}}
}
\keyword{predict}

