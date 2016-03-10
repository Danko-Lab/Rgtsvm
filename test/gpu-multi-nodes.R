library(parallel)
mc <- 2
run1 <- function(cost) {
	
	cat(cost, "\n");
    library(Rgtsvm)
    load("ELF1_trainData.rdata"); 
    gt.model <- svm( trainAll, trainAllStatus, type="C-classification", cross=10, gamma=0.00390625, cost=cost );
    #return(gt.model);
    return(gt.model$tot.accuracy);
    
}

 
ptm <- proc.time();
cl <- makeCluster(mc)
library(Rgtsvm) 
x <- parLapply(cl, seq_len(mc)*10, run1);
stopCluster(cl)

t.elapsed <- proc.time() - ptm;      
t.elapsed
