library(Rgtsvm);

load("ELF1_trainData.rdata"); # 99.5% (*)23 sec
#load("MAZ_svmdata.rdata")    # 99.4%
#load("389k_svmdata.rdata")   # 96.0% (*)20 min
#load("680k_svrdata.rdata")   # 97.1% (*)63 min 
#trainAllStatus[trainAllStatus==0] <- -1;

data <- data.frame(Y=trainAllStatus, trainAll);

#gt.tune <- tune.svm( Y~., data=data, sampling = "cross", gamma = 2^c(-4, -2, -1, 0,4), cost = 2^c(-2, -1, 0, 1, 2), cross=10 );
gt.tune <- tune.svm( Y~., data=data, sampling = "cross", gamma = 2^c(-2, -1, 0,4), cost = 2^c(-1, 0, 1), cross=10 );

save( gt.tune, file="gt.tune.rdata");

plot( gt.tune, transform.x = log2, transform.y = log2)
plot( gt.tune, type = "perspective", theta = 120, phi = 45)

if(0)
{
	gt.model <- svm( Y~., data, gamma=0.05, tolerance=0.01, type="C-classification", scale = TRUE, fitted=TRUE, cross=5);

	save(pt1, data, gt.model, file="test1-gt1.rdata");

	plot(gt.model, data, formula = Y~.);
}

library(e1071);
data(iris)

obj <- tune.svm( Species~., data = iris, gamma = 2^(-1:1), cost = 2^(2:4), sampling = "cross", cross=10 );
