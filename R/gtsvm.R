trim.space <- function (x) gsub("^\\s+|\\s+$", "", x)

gtsvmtrain.classfication.call<-function(y, x, param, final.result=FALSE, verbose=TRUE, ignoreNoProgress=FALSE)
{
	y.org <- y;
	y.idx <- c();
	for( y0 in sort(unique(y) ) )
		y.idx <- c( y.idx, which( y == y0 ) );
	y <- y[y.idx];
	x <- x[y.idx,];
	
	y0 <- y;
	
    if( param$nclass==2 )
		y <- as.integer( c(-1, 1)[y] );
	
    err <- empty_string <- paste(rep(" ", 255), collapse = "")
 	ptm <- proc.time();

	nr <- nrow(x);
    maxIter <- nr * 100;
    
    # GTSVM only support the un-biased classification for mulri-class.
    if (param$nclass>2) param$biased<-FALSE;
   
	if( sys.nframe()> 6  && as.character( as.list(sys.call(-5))[[1]])=="tune" ) 
		ignoreNoProgress <- TRUE;
	
    cret <- .C ("gtsvmtrain_classfication",
                ## data
                as.double  (if (param$sparse) x@ra else x),
                as.integer (nrow(x)), 
                as.integer (ncol(x)),

				## for multiclass, the values are from 0 to nclass-1. but for binary, the values are 1 and -1
				## the data from R starts from 1
                if(param$nclass>2 ) as.double(y)-1 else as.double(y),
                
                ## sparse index info
                as.integer64 (if (param$sparse) (x@ia)-1 else 0), #offset values start from 0
                as.integer64 (if (param$sparse) (x@ja)-1 else 0), #index values start from 1
                as.integer (param$sparse),

                ## parameters
                as.integer (param$kernel),
                as.integer (param$nclass),
                ## kernelParameter 3
                as.integer (param$degree),
                ## kernelParameter 1
                as.double  (param$gamma),
                ## kernelParameter 2
				as.double  (param$coef0),
                ## regularization
				as.double  (param$cost),
                as.double  (param$tolerance),
                ## as.integer (param$shrinking),
                ## as.integer (param$probability),
                as.integer (param$fitted),
                as.integer (param$biased),
                as.integer (maxIter),

                ## results
                nclasses = integer  (1),
                ## nr of support vectors
                nr       = integer  (1), 			
                ## the index of support vectors
                index    = integer  (nr),
				## the labels of classes
                labels   = integer  (param$nclass),
                ## the support vectors of each classes
                nSV      = integer  (param$nclass),
                rho      = double   (param$nclass * (param$nclass - 1) / 2),
                ## sigma    = double   (1),
                ## probA    = double   (param$nclass * (param$nclass - 1) / 2),
                ## probB    = double   (param$nclass * (param$nclass - 1) / 2),
                predict  = double   (nr),
		        totalIter= integer  (1),

                coefs    = double   (nr * param$nclass ),

				as.integer(ignoreNoProgress),
                as.integer(verbose),
                error    = err,
                PACKAGE  = "Rgtsvm");

    t.elapsed <- proc.time() - ptm;      

    #if ( cret$error != empty_string )
    #    stop(paste(cret$error, "!", sep=""))
    
    if ( trim.space(cret$error) != "" )
        stop(paste(cret$error, "!", sep=""))
	
	cret$t.elapsed <- t.elapsed; 

	if(final.result)
    {
    	cret$index  <- cret$index[1:cret$nr]
    	gtsvm.class <- ifelse( cret$nclasses==2, 1, cret$nclasses );

		cret <- list (
            ## parameter
            kernel    = param$kernel,
            degree    = param$degree,
            gamma     = param$gamma,
            coef0     = param$coef0,
            cost      = param$cost,		
			sparse    = param$sparse,
			
			#number of classes
			nclasses  = cret$nclasses,  
			# total number of sv
			tot.nSV   = cret$nr, 		
			# number of SV in diff. classes
			nSV       = cret$nSV[1:cret$nclasses], 
			# labels of the SVs.
			labels    = cret$label[1:cret$nclasses], 
			SV        = if (param$sparse) SparseM::t(SparseM::t(x[cret$index,])) else t(t(x[cret$index,])), #copy of SV
			y.SV      = y0[cret$index],
			# indexes of sv in x
			index     = cret$index,  
			##constants in decision functions
			rho       = cret$rho[1:(cret$nclasses * (cret$nclasses - 1) / 2)],
            ## coefficiants of sv
			coefs     = matrix( cret$coefs[1:(gtsvm.class * cret$nr)], ncol = gtsvm.class ),
			totalIter = cret$totalIter,
			t.elapsed = cret$t.elapsed );
    }
    
    return(cret);    
}


gtsvmpredict.classfication.call<-function( x, x.sparse, obj.train, param=list(decision.values=FALSE, probability = FALSE), verbose=TRUE )
{
    if (ncol(obj.train$SV) != ncol(x))
        stop ("test data does not match model !")

	ptm <- proc.time()
    err <- empty_string <- paste(rep(" ", 255), collapse = "")
	
	y.train <- obj.train$y.SV
    if( obj.train$nclass==2 )
		y.train <- as.integer( c(-1, 1)[y.train] );

    cret <- .C ("gtsvmpredict_classfication",
               as.integer (param$decision.values),
               as.integer (param$probability),

               ## model
               as.integer (obj.train$sparse),
               as.double  (if (obj.train$sparse) obj.train$SV@ra else obj.train$SV ),
               as.integer (nrow(obj.train$SV)), 
               as.integer (ncol(obj.train$SV)),
               as.integer64 (if (obj.train$sparse) obj.train$SV@ia-1 else 0),
               as.integer64 (if (obj.train$sparse) obj.train$SV@ja-1 else 0),

               as.integer (obj.train$nclasses),
               as.integer (obj.train$tot.nSV),
               if(obj.train$nclasses > 2 ) as.double(y.train)-1.0 else as.double(y.train),
			   
			   ## to-do list	
               as.double  (obj.train$rho),
               as.double  (as.vector(obj.train$coefs)),

               ## parameter
               as.integer (obj.train$kernel),
               as.integer (obj.train$degree),
               as.double  (obj.train$gamma),
               as.double  (obj.train$coef0),
               as.double  (obj.train$cost),

               ## test matrix
               as.integer (x.sparse),
               as.double  (if (x.sparse) x@ra else x ),
               as.integer (nrow(x)),
               as.integer64 (if (x.sparse) x@ia-1 else 0),
               as.integer64 (if (x.sparse) x@ja-1 else 0),

               ## decision-values
               ret = double( nrow(x) ),
               dec = double( nrow(x) * obj.train$nclasses ),
               prob = double( nrow(x) * obj.train$nclasses ),

               as.integer(verbose),
               error   = err,
               PACKAGE = "Rgtsvm");

    ##if ( cret$error != empty_string )
    ##    stop(paste(cret$error, "!", sep=""))
    
    if ( trim.space(cret$error) != "" )
        stop(paste(cret$error, "!", sep=""))

	cret$t.elapsed <- proc.time() - ptm;

	if( obj.train$nclasses > 2 ) cret$ret <- cret$ret + 1;

	return(cret);
}

gtsvmtrain.regression.call<-function(y1, x1, param, final.result=FALSE, verbose=TRUE, ignoreNoProgress=FALSE)
{
	x <- rbind(x1, x1);
	y <- c(y1, y1);
	
    err <- empty_string <- paste(rep(" ", 255), collapse = "")
 	ptm <- proc.time();

	nr <- nrow(x);
    maxIter <- nr * 100;

	if( sys.nframe()> 6  && as.character( as.list(sys.call(-5))[[1]])=="tune" ) 
		ignoreNoProgress <- TRUE;

 	cret <- .C ("gtsvmtrain_epsregression",
                ## data
                as.double  (if (param$sparse) x@ra else x),
                as.integer (nrow(x)), 
                as.integer (ncol(x)),
                as.double  (y),
                ## sparse index info
                as.integer64 (if (param$sparse) (x@ia)-1 else 0), #offset values start from 0
                as.integer64 (if (param$sparse) (x@ja)-1 else 0), #index values start from 1
                as.integer (param$sparse),

                ## parameters
                as.integer (param$kernel),
                ## kernelParameter 3
                as.integer (param$degree),
                ## kernelParameter 1
                as.double  (param$gamma),
                ## kernelParameter 2
				as.double  (param$coef0),
                ## regularization
				as.double  (param$cost),
				
                as.double  (param$tolerance),
                as.double  (param$epsilon),
                as.integer (param$shrinking),
                as.integer (param$probability),
                as.integer (param$fitted),
                as.integer (maxIter),

                ## results
		        totalIter= integer  (1),
                ## nr of support vectors
                nr       = integer  (1), 			
                ## the index of support vectors
                index    = integer  (nr),
				## the labels of classes
                labels   = integer  (param$nclass),
                ## the support vectors of each classes
                nSV      = integer  (param$nclass),
                rho      = double   (1),
                sigma    = double   (1),
                probA    = double   (param$nclass * (param$nclass - 1) / 2),
                probB    = double   (param$nclass * (param$nclass - 1) / 2),
                predict  = double   (nr),
                coefs    = double(nr),

				as.integer(ignoreNoProgress),
                as.integer(verbose),
                error    = err,
                PACKAGE  = "Rgtsvm");
          
    t.elapsed <- proc.time() - ptm;      

    ##if ( cret$error != empty_string )
    ##    stop(paste(cret$error, "!", sep=""))

    if ( trim.space(cret$error) != "" )
        stop(paste(cret$error, "!", sep=""))
  
	
	cret$t.elapsed <- t.elapsed; 
	cret$nclasses  <- 2;
	cret$nSV       <- c(0,0);
	cret$labels    <- c(0,0);
	
	if(final.result)
    {
    	cret$index  <- cret$index[1:cret$nr]
    	gtsvm.class <- 1;
    	
		cret <- list (
            ## parameter
            kernel    = param$kernel,
            degree    = param$degree,
            gamma     = param$gamma,
            coef0     = param$coef0,
            cost      = param$cost,		
			sparse    = param$sparse,
			
			#number of classes
			nclasses  = 1,  
			# total number of sv
			tot.nSV   = cret$nr, 		
			# number of SV in diff. classes
			nSV       = cret$nSV, 
			# labels of the SVs.
			labels    = cret$labels, 
			SV        = if (param$sparse) SparseM::t(SparseM::t(x[cret$index,])) else t(t(x[cret$index,])), #copy of SV
			y.SV      = y1[cret$index],
			# indexes of sv in x
			index     = cret$index,  
			##constants in decision functions
			rho       = cret$rho,
            ## coefficiants of sv
			coefs     = matrix( cret$coefs[1:(gtsvm.class * cret$nr)], ncol = gtsvm.class ),
			totalIter = cret$totalIter,
			t.elapsed = cret$t.elapsed );
    }
    
    return(cret);    
}


gtsvmpredict.regression.call<-function( x, x.sparse, obj.train, param=list(decision.values=FALSE, probability = FALSE), verbose=TRUE )
{
    if (ncol(obj.train$SV) != ncol(x))
        stop ("test data does not match model !")

	ptm <- proc.time()
    err <- empty_string <- paste(rep(" ", 255), collapse = "")

	y.train <- c();
	#y.train <- obj.train$y.SV
    #if( obj.train$nclass == 2 && obj.train$type == C_CLASSFICATION)
	#	y.train <- as.integer( c(-1, 1)[y.train] );

    cret <- .C ("gtsvmpredict_epsregression",
               as.integer (param$decision.values),
               as.integer (param$probability),

               ## model
               as.integer (obj.train$sparse),
               as.double  (if (obj.train$sparse) obj.train$SV@ra else obj.train$SV ),
               as.integer (nrow(obj.train$SV)), 
               as.integer (ncol(obj.train$SV)),
               as.integer64 (if (obj.train$sparse) obj.train$SV@ia-1 else 0),
               as.integer64 (if (obj.train$sparse) obj.train$SV@ja-1 else 0),
               as.integer (obj.train$tot.nSV),
               as.double  (y.train),
			   
			   ## to-do list	
               as.double  (obj.train$rho),
               as.double  (as.vector(obj.train$coefs)),

               ## parameter
               as.integer (obj.train$kernel),
               as.integer (obj.train$degree),
               as.double  (obj.train$gamma),
               as.double  (obj.train$coef0),
               as.double  (obj.train$cost),

               ## test matrix
               as.integer (x.sparse),
               as.double  (if (x.sparse) x@ra else x ),
               as.integer (nrow(x)),
               as.integer64 (if (x.sparse) x@ia-1 else 0),
               as.integer64 (if (x.sparse) x@ja-1 else 0),

               ## decision-values
               ret = double( nrow(x) ),
               dec = double( nrow(x)  ),
               prob = double( nrow(x) * obj.train$nclasses ),

               as.integer(verbose),
               error   = err,
               PACKAGE = "Rgtsvm");

    #if ( cret$error != empty_string )
    #    stop(paste(cret$error, "!", sep=""))

    if ( trim.space(cret$error) != "" )
        stop(paste(cret$error, "!", sep=""))
    
	cret$t.elapsed <- proc.time() - ptm;
	
	return(cret);
}