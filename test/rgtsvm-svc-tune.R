library(Rgtsvm);

load("ELF1_trainData.rdata"); # 99.5% (*)23 sec
data <- data.frame(Y=trainAllStatus, trainAll);

### Example 1 
gt.tune <- tune.svm( Y~., data=data, gamma = 2^c(-6, -8, -10), cost = 2^c(-1, 1, 2, 3), verbose=F, tunecontrol=tune.control(sampling = "cross", cross = 4 ) );
plot( gt.tune, transform.x = log2, transform.y = log2)
plot( gt.tune, type = "perspective", theta = 120, phi = 45)

gt.model <- svm( Y~., data=data, type="C-classification", gamma=0.00390625, cost=10, rough.cross=4 );


### Example 2 
gt.tune <- tune.svm( Y~., data=data, gamma = 2^c(-6, -8, -10), cost = 2^c(1, 2, 3,4, 5), verbose=F, tunecontrol=tune.control(sampling = "cross", cross = 10,rough.cross = 2) );
plot( gt.tune, transform.x = log2, transform.y = log2)
plot( gt.tune, type = "perspective", theta = 120, phi = 45)

### Example 3 
if(0)
{
	library(e1071);
	data(iris)

	obj <- tune.svm( Species~., data = iris, gamma = 2^(-1:1), cost = 2^(2:4), sampling = "cross", cross=10 );
}
