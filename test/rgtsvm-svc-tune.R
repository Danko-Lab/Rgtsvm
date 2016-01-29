library(Rgtsvm);

load("ELF1_trainData.rdata"); # 99.5% (*)23 sec
#load("MAZ_svmdata.rdata")    # 99.4%
#load("389k_svmdata.rdata")   # 96.0% (*)20 min
#load("680k_svrdata.rdata")   # 97.1% (*)63 min 
#trainAllStatus[trainAllStatus==0] <- -1;

data <- data.frame(Y=trainAllStatus, trainAll);

gt.tune <- tune.svm( Y~., data=data, sampling = "fix", gamma = 2^c(-8,-4, -2, -1, 0,4), cost = 2^c(-8,-4,-2,0, 1, 2) );

plot( gt.tune, transform.x = log2, transform.y = log2)
plot( gt.tune, type = "perspective", theta = 120, phi = 45)


gt.model <- svm( Y~., data, gamma=0.05, tolerance=0.01, type="C-classification", scale = TRUE, fitted=TRUE, cross=5);

save(pt1, data, gt.model, file="test1-gt1.rdata");

plot(gt.model, data, formula = Y~.);

