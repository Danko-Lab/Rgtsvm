\name{predict.gtsvm}
\alias{predict.gtsvm}
\title{Predict Method for Support Vector Machines}
\description{
  This function predicts values based upon a model trained by \code{svm}.
}
\usage{
\method{predict}{gtsvm}(object, newdata, score = FALSE, ..., na.action = na.omit)
}

\arguments{
  \item{object}{Object of class \code{"svm"}, created by \code{svm}.}
  \item{newdata}{An object containing the new input data: either a
  matrix or a sparse matrix (object of class
    \code{\link[Matrix]{Matrix}} provided by the \pkg{Matrix} package,
    or of class \code{\link[SparseM]{matrix.csr}}
    provided by the \pkg{SparseM} package, or of class
    \code{\link[slam]{simple_triplet_matrix}} provided by the \pkg{slam}
    package). A vector will
    be transformed to a n x 1 matrix.}
  \item{score}{Logical controlling whether the decision values
    of all binary classifiers computed in multiclass classification
    shall be computed and returned.}
  \item{na.action}{A function to specify the action to be taken if \sQuote{NA}s are
    found. The default action is \code{na.omit}, which leads to rejection of cases
    with missing values on any required variable. An alternative
    is \code{na.fail}, which causes an error if \code{NA} cases
    are found. (NOTE: If given, this argument must be named.)}	
  \item{\dots}{Currently not used.}
}
\value{
  A vector of predicted values (for classification: a vector of labels, for density
  estimation: a logical vector). If \code{decision.value} is
  \code{TRUE}, the vector gets a \code{"decision.values"} attribute
  containing a n x c matrix (n number of predicted values, c number of
  classifiers) of all c binary classifiers' decision values. There are k
  * (k - 1) / 2 classifiers (k number of classes). The colnames of
  the matrix indicate the labels of the two classes. If \code{probability} is
  \code{TRUE}, the vector gets a \code{"probabilities"} attribute
  containing a n x k matrix (n number of predicted values, k number of
  classes) of the class probabilities.
}
\note{
  If the training set was scaled by \code{svm} (done by default), the
  new data is scaled accordingly using scale and center of
  the training data.
}
\author{
  David Meyer (based on C++-code by Chih-Chung Chang and Chih-Jen Lin)\cr
  \email{David.Meyer@R-project.org}
}
\seealso{
  \code{\link{svm}}
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
model <- svm(x, y, probability = TRUE) 

print(model)
summary(model)

# test with train data
pred <- predict(model, x)
# (same as:)
pred <- fitted(model)

# compute decision values and probabilites
pred <- predict(model, x, decision.values = TRUE, probability = TRUE)
attr(pred, "decision.values")[1:4,]
attr(pred, "probabilities")[1:4,]

## try regression mode on two dimensions

# create data
x <- seq(0.1, 5, by = 0.05)
y <- log(x) + rnorm(x, sd = 0.2)

# estimate model and predict input values
m   <- svm(x, y)
new <- predict(m, x)

# visualize
plot   (x, y)
points (x, log(x), col = 2)
points (x, new, col = 4)

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
}
\keyword{neural}
\keyword{nonlinear}
\keyword{classif}

