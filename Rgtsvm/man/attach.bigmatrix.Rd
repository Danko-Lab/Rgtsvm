\name{attach.bigmatrix}
\alias{attach.bigmatrix}
\title{
Wrapping a big matrix into a reference class
}
\description{
This function wraps a big matrix into a reference class in order to avoid multiple variable copying when variables are passed into deep calls in R. It seems to become pointer calling in C language.
}
\usage{
attach.bigmatrix(data)
}
\arguments{
  \item{data}{ matrix object }
}
\value{
Return a reference class with the name "BigMatrix.refer". It can be used in \code{\link{svm}} and \code{\link{predict}} calling in Rgtsvm.
}

\seealso{
\code{\link{load.bigmatrix}}
}
\examples{
library(mvtnorm);
size=5000;
dimension=100;

covar.mat <- matrix(runif(dimension*dimension),nrow=dimension);
covar.mat <- t(covar.mat) %*% ( covar.mat);
covar.mat <- round( (covar.mat + t(covar.mat))/2, 4);

zero <- rmvnorm(size, mean=c(1:dimension), sigma= covar.mat);
one  <- rmvnorm(size, mean=c(1:dimension)-5, sigma= covar.mat);
x <- rbind(zero,one);
y <- c(rep(0,nrow(zero)),rep(1,nrow(one)));

i.all <- sample(1:(2*size));
i.training <- i.all[(1:round(2*size*0.8))];
i.test <- i.all[-c(1:round(2*size*0.8))];

bigm.x <- attach.bigmatrix( data = x[ i.training,]);
model.gpu <- svm(bigm.x,y[ i.training ],type="C-classification");

y.pred <- predict(model.gpu,x[i.test,]);
cat("accuracy", sum(y.pred==y[i.test])/length(i.test),"\n");


}
\keyword{ data loading }
