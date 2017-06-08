\name{attach.bigmatrix}
\alias{attach.bigmatrix}
\title{
Wrapping a big matrix into a reference class
}
\description{
This function wrapps a big matrix into a reference class in order to avoid multiple variable coping when variables are passed into deep calls in R. It seems to become pointer calling in C language.
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

library(MASS);
size=5000;
dimension=100;
covar.mat <- matrix(runif(dimension*dimension),nrow=dimension);
covar.mat <- t(covar.mat)%*% covar.mat;
zero <- mvrnorm(size,mu=c(1:dimension),Sigma= covar.mat);
one  <- mvrnorm(size,mu=c(1:dimension)-5,Sigma= covar.mat);
x <- rbind(zero,one);
y <- c(rep(0,nrow(zero)),rep(1,nrow(one)));

i.all <- 1:(2*size);
i.training <- sample(i.all, length(i.all)*0.8);
i.test <- i.all [! i.all %in% i.training ];

bigm.x <- attach.bigmatrix( data = x[ i.training,]);
model.gpu <- svm(bigm.x,y[ i.training ],type="C-classification");

y.pred <- predict(model.gpu,x[i.test,]);
cat("accuracy", sum(y.pred==y[i.test])/length(i.test),"\n");


}
\keyword{ data loading }
