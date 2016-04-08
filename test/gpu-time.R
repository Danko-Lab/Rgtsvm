


search.gtmodel <- function(dir.f, exclude.dir= NULL)
{
	get.gtsvm.info<-function(var)
	{
cat( NROW(var$fitted), NCOL(var$SV), var$t.elapsed[3], var$fitted.MSE, var$fitted.r2, "\n");
		data <- c(NROW(var$fitted), NCOL(var$SV), var$t.elapsed[3], var$fitted.MSE, var$fitted.r2);
		return(list(data=data, type.name=var$type.name));
	}

	gtsvmdata <- c();
	gtsvmname <- c();
	gtsvmtype <- c();
	fnames = list.files(dir.f, pattern = glob2rx("*.rdata"), full.names = TRUE, recursive = TRUE)
	for(rdata in fnames)
	{
		var.olds <- ls();
cat("Loading", rdata,"\n");
		if (!is.null(exclude.dir) && dirname(rdata) %in% exclude.dir ) {cat("Skip\n"); next;}

		r <- try( load(rdata) );
		if(class(r)=="try-error") next;

		var.news <- setdiff(ls(), var.olds)
		for(var.name in var.news)
		{
cat("check vairable:", var.name, "\n");
			var <- eval(parse(text=var.name));
			if(class(var)!="gtsvm") next;
			var.info <- try( get.gtsvm.info(var) );
			if(class(var.info)!="try-error")
			{
				gtsvmname <- c(gtsvmname, rdata);
				gtsvmtype <- c(gtsvmtype, var.info$type.name);
				gtsvmdata <- rbind(gtsvmdata, var.info$data);
			}
		}

		rm(var.news);
		gc();
	}

	return(data.frame( gtsvmname, gtsvmtype, gtsvmdata ));

}

exclude.dir <-c("/work/03350/tg826494/Rgtsvm/dreg/gm12878",
	"/work/03350/tg826494/Rgtsvm/dreg/k562",
	"/work/03350/tg826494/Rgtsvm/dreg/UMU",
	"/work/03350/tg826494/Rgtsvm/dreg/UMU_57",
	"/work/03350/tg826494/Rgtsvm/dreg/UMU88",
	"/work/03350/tg826494/Rgtsvm/dreg/UMU_II_III_VII" );


r1 <-  search.gtmodel (getwd(), exclude.dir=exclude.dir );

write.table( r1, file="gpu-time.tab", quote=F, row.names=F, col.names=F);
