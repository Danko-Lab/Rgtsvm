library(Rgtsvm);
library(MASS);

####multi-class classification example
size=3000
dimension=2


covar.mat0<-matrix(runif(dimension*dimension),nrow=dimension)
covar.mat0<-t(covar.mat0)%*% covar.mat0
covar.mat1<-matrix(runif(dimension*dimension),nrow=dimension)
covar.mat1<-t(covar.mat1)%*% covar.mat1
covar.mat2<-matrix(runif(dimension* dimension),nrow= dimension)
covar.mat2<-t(covar.mat2)%*% covar.mat2

zero<-mvrnorm(size,mu=c(1: dimension),Sigma= covar.mat0)
one<-mvrnorm(size,mu=c(1: dimension)-100,Sigma= covar.mat1)
two<-mvrnorm(size,mu=c(1: dimension)-200,Sigma= covar.mat2)

x<-rbind(zero,one,two)
y<-c(rep(0,nrow(zero)),rep(1,nrow(one)),rep(2,nrow(two)))


all.idx<-1:(3*size)
training.idx<-sample(all.idx, length(all.idx)*0.8)
test.idx<-all.idx[! all.idx %in% training.idx]


model.gpu<-svm(x[training.idx,],y[training.idx],type="C-classification", probability=TRUE)
predicted.y<-predict(model.gpu,x[test.idx,], probability=TRUE)
cat("accuracy", sum(predicted.y==y[test.idx])/length(test.idx),"\n")


for (gamma in 10^c(-2:5))
{
  model.gpu<-Rgtsvm::svm(x[training.idx,],y[training.idx],type="C-classification",gamma= gamma)
  predicted.y<-predict(model.gpu,x[test.idx,])
  cat("accuracy", sum(predicted.y==y[test.idx])/length(test.idx),"\n")

}

for (gamma in 10^c(-2:5)){
  model.gpu<-e1071::svm(x[training.idx,],y[training.idx],type="C-classification",gamma= gamma)
  predicted.y<-predict(model.gpu,x[test.idx,])
  cat("accuracy", sum(predicted.y==y[test.idx])/length(test.idx),"\n")
}

system.time(
model.gpu<-Rgtsvm::svm(x[training.idx,],y[training.idx],type="C-classification")
)

table(model.gpu$fitted==y[training.idx])

system.time( model.cpu<-e1071::svm(x,y,type="C-classification") )
