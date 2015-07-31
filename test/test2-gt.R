library(Rgtsvm);
 
#wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/glass.scale
# normal matrix 
#mat <- as.matrix( load.svmlight("http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/glass.scale") );
# sparse matrix 
#mat <- load.svmlight("http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/glass.scale") ;

mat <- load.svmlight("mnist")
 
gt.model <- svm( x=mat[,-1], y=mat[,1], gamma=0.05, tolerance=0.01, type="C-classification", scale = FALSE, fitted=TRUE);
save(gt.model, file="test2-gt.rdata");
cat("correction=", gt.model$correct, "\n");
 
gt.predit <- predict( gt.model, mat[,-1], score=TRUE );
save(gt.model, gt.predit,  file="test2-gt.rdata");

head(gt.model$fitted, n=200);

summary(gt.predit);
