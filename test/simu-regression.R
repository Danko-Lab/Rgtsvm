library(Rgtsvm);
library(MASS);
set.seed(1);
size=20000;
dimension=100;

covar.mat0 <- matrix(runif(dimension*dimension),nrow=dimension);
covar.mat0 <- t(covar.mat0)%*% covar.mat0;

covar.mat1 <- matrix(runif(dimension*dimension),nrow=dimension);
covar.mat1 <- t(covar.mat1)%*% covar.mat1;

zero <- mvrnorm(size,mu=c(1:dimension),Sigma= covar.mat0);
one  <- mvrnorm(size,mu=c(1:dimension)-10,Sigma= covar.mat1);

zero.d <- (zero-matrix(rep(c(1:100), size),nrow=size,byrow=T));
zero.d <- apply(zero.d,1,FUN=function(x)sum(x^2))+rnorm(n=size,sd=5000);

one.d <- (one-matrix(rep(c(1:100), size),nrow=size,byrow=T));
one.d <- apply(one.d,1,FUN=function(x)sum(x^2))+rnorm(n=size,sd=5000);

x <- rbind(zero,one);
y <- c(zero.d, one.d);

i.all <- 1:(2*size);
i.training <- sample(i.all, length(i.all)*0.8);
i.test <- i.all [! i.all %in% i.training ];

model.gpu <- svm(x[i.training,],y[ i.training ],type="eps-regression");
y.pred <- predict( model.gpu, x[i.test,] );
cat("correlation=", cor( y.pred, y[i.test]),"\n");
