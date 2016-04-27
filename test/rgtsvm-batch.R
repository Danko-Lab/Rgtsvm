library(Rgtsvm)

##eps-regreesion test
if(0)
{
	load("SVR_training.rdata");

	#gt.train   <- svm( training_x_y[,-c(121)], training_x_y[,121], gamma=0.00833, type="eps-regression", scale=TRUE, fitted=FALSE);
	#save(gt.train, file="SVR_training_svr.rdata");

	load("SVR_training_svr.rdata");

	ptm <- proc.time();
	gt.test    <- predict(gt.train, training_x_y[,-c(121)] );
	t.elapsed <- proc.time() - ptm;
	show(t.elapsed);


	newdata <- training_x_y[,-121];
	file.rds <- "batch-test.RDS";
	saveRDS(newdata,file.rds);

	ptm <- proc.time();
	gt.test    <- predict.batch(gt.train, c(file.rds,file.rds,file.rds) );
	t.elapsed <- proc.time() - ptm;
	show(t.elapsed);
}


if(0)
{
	load("SVC_ELF1_train.rdata");

	#gt.model1 <- svm( trainAll, trainAllStatus, gamma=0.05, tolerance=0.01, type="C-classification",class.weights=c(0.5, 200), scale = FALSE, fitted=TRUE);
	#save(gt.model1, file="SVR_training_svc.rdata");
	load("SVR_training_svc.rdata");

	ptm <- proc.time();
	gt.test <- predict( gt.model1, trainAll, decision.values = TRUE );
	t.elapsed <- proc.time() - ptm;
	show(t.elapsed);

	file.rds <- "batch-test.RDS";
	saveRDS(trainAll, file.rds);

	ptm <- proc.time();
	gt.test2    <- predict.batch(gt.model1, c(file.rds,file.rds,file.rds) );
	t.elapsed <- proc.time() - ptm;
	show(t.elapsed);
}
