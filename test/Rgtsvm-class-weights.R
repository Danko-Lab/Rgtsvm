library(Rgtsvm);

file.rdata <- "ELF1_trainData.rdata"; # 99.5% (*)23 sec
load( file.rdata ); 

make_unbalanced_data<-function(ratio)
{
	idx1<- which(trainAllStatus==0)
	idx2<- which(trainAllStatus==1)

	idx1.1 <- sample(1:length(idx1))[1:round(ratio*length(idx1))];
	minor.idx1 <- idx1[ idx1.1 ]
	major.idx1 <- idx1[ -idx1.1 ]
	idx2.1 <- sample(1:length(idx2))[1:round(ratio*length(idx2))];
	minor.idx2 <- idx2[ idx2.1 ]
	major.idx2 <- idx2[ -idx2.1 ]

	### Example 1
	train.idx <- c(minor.idx1, major.idx2 );
	test.idx <- c(major.idx1, minor.idx2 );
	
	return(list(train.idx=train.idx, test.idx=test.idx));
}

test.weights <- function(idx, class.weights=c(1,1))
{
	#gt.model1 <- svm( trainAll[idx$train.idx,], trainAllStatus[idx$train.idx], gamma=0.05, tolerance=0.01, type="C-classification", scale = FALSE, fitted=TRUE);
	gt.model1 <- svm( trainAll[idx$train.idx,], trainAllStatus[idx$train.idx], gamma=0.05, tolerance=0.01, type="C-classification",class.weights=class.weights, scale = FALSE, fitted=TRUE);

	gt.predict1 <- predict( gt.model1, trainAll[idx$train.idx,], decision.values = TRUE );
	accuracy1 <- length(which(trainAllStatus[idx$train.idx]==gt.predict1))/length(trainAllStatus[idx$train.idx]);
	cat("accuracy=", accuracy1, "\n");

	gt.predict2 <- predict( gt.model1, trainAll[idx$test.idx,], decision.values = TRUE );
	accuracy2 <- length(which(trainAllStatus[idx$test.idx]==gt.predict2))/length(trainAllStatus[idx$test.idx]);
	cat("accuracy=", accuracy2, "\n");

	return(c(accuracy1, accuracy2));
}

r <- make_unbalanced_data( 0.1)

x1<-test.weights(r );
x2<-test.weights(r, c(10,1));
x3<-test.weights(r, c(1,0.1));
x4<-test.weights(r, c(10,0.1));
x5<-test.weights(r, c(10,0.01));
x6<-test.weights(r, c(100,0.01));

## > x1
## [1] 0.9991989 0.3363709
## > x2
## [1] 0.9998999 0.3870419
## > x3
## [1] 0.9535349 0.6366914
## > x4
## [1] 0.9406169 0.7022832
##> x5
## [1] 0.1000401 0.8999599
## > x6
## [1] 0.1000401 0.8999599


r <- make_unbalanced_data( 0.3)

x1<-test.weights(r );
x2<-test.weights(r, c(10,1));
x3<-test.weights(r, c(1,0.1));
x4<-test.weights(r, c(10,0.1));
x5<-test.weights(r, c(10,0.01));
x6<-test.weights(r, c(100,0.01));

# > x1
# [1] 0.9978971 0.7212097
# > x2
# [1] 0.9981975 0.7441418
# > x3
# [1] 0.5674945 0.7910074
# > x4
# [1] 0.5650911 0.7896054
# > x5
# [1] 0.30002 0.69998
# > x6
# [1] 0.30002 0.69998
