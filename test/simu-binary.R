library(Rgtsvm);
library(MASS);
set.seed(1);
size=50000;
dimension=100;

covar.mat <- matrix( runif(dimension*dimension),nrow=dimension);
covar.mat <- t(covar.mat)%*% covar.mat;

zero<- mvrnorm(size,mu=c(1:dimension),Sigma=covar.mat);
one <- mvrnorm(size,mu=c(1:dimension)-5,Sigma=covar.mat);

x <- rbind(zero,one);
y <- c(rep(0,nrow(zero)),rep(1,nrow(one)));

i.all <- 1:(2*size);
i.training <- sample(i.all, length(i.all)*0.8);
i.test <- i.all [! i.all %in% i.training];

model.gpu <- svm(x[i.training,],y[i.training],type="C-classification", probability=T);
y.pred <-predict( model.gpu, x[i.test,] , probability=T);
cat("accuracy", sum(y.pred == y[i.test])/length(i.test),"\n");

#accuracy=0.8997
