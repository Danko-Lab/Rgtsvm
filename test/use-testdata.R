library(Rgtsvm);
library(SparseM)

file1 <- "ELF1_trainData.rdata"); # 99.5% (*)23 sec
file2 <- "/work/03350/tg826494/Rgtsvm/test/MAZ_svmdata.rdata")    # 99.4%
file3 <- "/work/03350/tg826494/Rgtsvm/test/389k_svmdata.rdata")   # 96.0% (*)20 min
file4 <- "/work/03350/tg826494/Rgtsvm/test/680k_svrdata.rdata")   # 97.1% (*)63 min 
file5 <- "/work/03350/tg826494/Rgtsvm/test/400k_trainingvectors.rdata") 
file6 <- "/work/03350/tg826494/Rgtsvm/test/MAFK_chipseqTesting_k562_gray.rdata") 
