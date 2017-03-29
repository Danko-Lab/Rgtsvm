library(Rgtsvm);
mat <- load.svmlight("mnist")

#wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/glass.scale
# normal matrix
#mat <- as.matrix( load.svmlight("http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/glass.scale") );
# sparse matrix
#mat <- load.svmlight("http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/glass.scale") ;

mat <- load.svmlight("mnist")
mat <- load.svmlight("epsilon_normalized")
mat <- load.svmlight("glass.scale", .loadbyC=TRUE)

gt.model <- svm( x=mat[,-1], y=mat[,1], gamma=0.05, tolerance=0.01, type="C-classification", scale = FALSE, fitted=TRUE);
save(gt.model, file="test2-gt.rdata");
cat("correction=", gt.model$correct, "\n");

gt.predit <- predict( gt.model, mat[,-1], score=FALSE );
save( gt.model, gt.predit, y=mat[,1], file="test2-gt.rdata");

correct = length( which(gt.predit == mat[,1]) )/nrow(mat)
cat("correct=", correct, "\n");

summary(gt.predit);
