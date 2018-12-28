##
##	Copyright (C) 2017  Zhong Wang
##
##	This program is free software: you can redistribute it and/or modify
##	it under the terms of the GNU General Public License as published by
##	the Free Software Foundation, either version 3 of the License, or
##	(at your option) any later version.
##
##	This program is distributed in the hope that it will be useful,
##	but WITHOUT ANY WARRANTY; without even the implied warranty of
##	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##	GNU General Public License for more details.
##
##	You should have received a copy of the GNU General Public License
##	along with this program.  If not, see <http://www.gnu.org/licenses/>.
##

trim.space <- function (x) gsub("^\\s+|\\s+$", "", x)

gtsvmtrain.classfication.call<-function(y, x, param, final.result=FALSE, verbose=FALSE, ignoreNoProgress=FALSE)
{
    if( inherits(x, "BigMatrix.refer") ) bigm.push(x);

    y.org <- y;
    y.idx <- c();
    for( y0 in sort(unique(y) ) )
    {
        if (verbose) cat("* Lable=", y0, length(which( y == y0 ) ), "\n");
        y.idx <- c( y.idx, which( y == y0 ) );
    }

    y <- y[y.idx];

    if ( inherits(x, "BigMatrix.refer") )
        bigm.subset(x, rows=y.idx)
    else
        x <- x[y.idx,];

    y0 <- y;
    if( param$nclass==2 )
        y <- as.integer( c(-1, 1)[y] )

    # GTSVM only support the un-biased classification for mulri-class.
    if (param$nclass>2) param$biased<-FALSE;

    if( sys.nframe()> 6)
    {
        func.call <- try( as.character( as.list(sys.call(-5))[[1]]) )
        if( func.call == "tune" ) ignoreNoProgress <- TRUE;
    }

    ptm <- proc.time();
    nr  <- NROW(x);
    if(is.null(param$maxIter))
    	param$maxIter <- NROW(x)*ifelse( NCOL(x)<=100, 100, NCOL(x) );

    if(is.null(param$class.weights))
        param$class.weights <- rep(1, param$nclass);

    cret <- .Call( "gtsvmtrain_classfication",
                ## data
                as.integer (param$sparse),
                as.double  (if (param$sparse) x@ra else x),
                as.integer64 (if (param$sparse) (x@ia)-1 else 0), #offset values start from 0
                as.integer64 (if (param$sparse) (x@ja)-1 else 0), #index values start from 1
                as.integer (NROW(x)),
                as.integer (NCOL(x)),
                as.integer (if ( class(x)=="BigMatrix.refer") bigm.internal.nrow( x ) else NROW(x)),
                as.integer (if ( class(x)=="BigMatrix.refer") bigm.internal.ncol( x ) else NCOL(x)),
                as.integer (if ( class(x)=="BigMatrix.refer") bigm.row.index( x )-1 else c(1:NROW(x))-1 ),
                as.integer (if ( class(x)=="BigMatrix.refer") bigm.col.index( x )-1 else c(1:NCOL(x))-1 ),
                ## sparse index info

                ## for multiclass, the values are from 0 to nclass-1. but for binary, the values are 1 and -1
                ## the data from R starts from 1
                if(param$nclass>2 ) as.double(y)-1 else as.double(y),

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
                as.double  (param$class.weights),
                as.double  (param$tolerance),
                as.integer (param$fitted),
                as.integer (param$biased),
                ## maxIter = NROW(x)*100;
                as.integer (param$maxIter ),
                as.integer (ignoreNoProgress),

## #This part will be returned in SEXP object.
##                ## results
##                totalIter= integer  (1),
##                ## the total number of classes
##                nclasses = integer  (1),
##                ## nr of support vectors
##                nr       = integer  (1),
##                ## the index of support vectors
##                index    = integer  (nr),
##                ## the labels of classes
##                labels   = integer  (param$nclass),
##                ## the support vectors of each classes
##                nSV      = integer  (param$nclass),
##                rho      = double   (param$nclass * (param$nclass - 1) / 2),
##                coefs    = double   (nr * param$nclass ),
##                predict  = double   (NROW(y)),
##                decision = double   (NROW(y)*param$nclass),
##                error    = as.integer(1),
## (End)

                as.integer(param$nclass),
                as.integer(verbose),
                PACKAGE  = "Rgtsvm");

    t.elapsed <- proc.time() - ptm;

    if ( cret$error!=0 ) stop("Error in GPU process.")

    cret$t.elapsed <- t.elapsed;

    gtsvm.class <- ifelse( cret$nclasses==2, 1, cret$nclasses );

    if( final.result )
    {
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
            nr        = cret$nr,
            tot.nSV   = cret$nr,

            index     = cret$index,
            nSV       = cret$nSV,
            rho       = cret$rho,
            labels    = cret$labels,
            coefs     = cret$coefs,

            totalIter = cret$totalIter,
            t.elapsed = cret$t.elapsed );
    }

    #index of SV points
    cret$index <- cret$index[1:cret$nr];
    # number of SV in diff. classes
    cret$nSV   <- cret$nSV[1:cret$nclasses];

    #cret$SV    <- if (param$sparse) SparseM::t(SparseM::t(x[cret$index,])) else x[cret$index,,drop=F];
    cret$SV <- x[cret$index [1:cret$nr], ];

    ##constants in decision functions
    cret$rho   <- cret$rho[1:(cret$nclasses * (cret$nclasses - 1) / 2)];
    # labels of the SVs.
    cret$labels<- cret$label[1:cret$nclasses];

    ## coefficiants of sv
    if( param$nclass==2 )
        cret$coefs <- cret$coefs[1:cret$nr] * as.integer( y [ cret$index ] )
    else
        cret$coefs  <- matrix( cret$coefs[1:(gtsvm.class * cret$nr)], ncol = gtsvm.class );

    if( inherits(x, "BigMatrix.refer") ) bigm.pop(x);

    return(cret);
}


gtsvmpredict.classfication.call<-function( x, x.sparse, obj.train, param=list(decision.values=FALSE, probability = FALSE), verbose=FALSE )
{
    if (ncol(obj.train$SV) != ncol(x))
        stop ("test data does not match model !")

    ptm <- proc.time();
    if( is.null(obj.train$pointer ) )
	    cret <- .Call ("gtsvmpredict_classfication",
               as.integer (param$decision.values),
               as.integer (param$probability),

               ## model
               as.integer (obj.train$sparse),
               as.double  (if (obj.train$sparse) obj.train$SV@ra else obj.train$SV ),
               as.integer64 (if (obj.train$sparse) obj.train$SV@ia-1 else 0),
               as.integer64 (if (obj.train$sparse) obj.train$SV@ja-1 else 0),
               as.integer (NROW(obj.train$SV)),
               as.integer (NCOL(obj.train$SV)),
               as.integer (c(1:NROW(obj.train$SV))-1 ),
               as.integer (c(1:NCOL(obj.train$SV))-1 ),

               as.integer (obj.train$nclasses),
               as.integer (obj.train$tot.nSV),
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
               as.integer64 (if (x.sparse) x@ia-1 else 0),
               as.integer64 (if (x.sparse) x@ja-1 else 0),
               as.integer (NROW(x)),
               as.integer (if ( class(x)=="BigMatrix.refer") bigm.internal.nrow( x ) else NROW(x)),
               as.integer (if ( class(x)=="BigMatrix.refer") bigm.internal.ncol( x ) else NCOL(x)),
               as.integer (if ( inherits(x, "BigMatrix.refer") ) bigm.row.index( x )-1 else c(1:NROW(x))-1 ),
               as.integer (if ( inherits(x, "BigMatrix.refer") ) bigm.col.index( x )-1 else c(1:NCOL(x))-1 ),

## #This part will be returned in SEXP object.
##               ## decision-values
##               ret = double( NROW(x) ),
##               dec = double( NROW(x) * obj.train$nclasses ),
##               prob = double( NROW(x) * obj.train$nclasses ),
##               error = as.integer(1),
## (End)

               as.integer(verbose),
               PACKAGE = "Rgtsvm")
	else
	    cret <- .Call ("gtsvmpredict_classfication_direct",
               as.integer (param$decision.values),
               as.integer (param$probability),

               ## model
               obj.train$pointer,

               ## test matrix
               as.integer (x.sparse),
               as.double  (if (x.sparse) x@ra else x ),
               as.integer64 (if (x.sparse) x@ia-1 else 0),
               as.integer64 (if (x.sparse) x@ja-1 else 0),
               as.integer (NROW(x)),
               as.integer (if ( class(x)=="BigMatrix.refer") bigm.internal.nrow( x ) else NROW(x)),
               as.integer (if ( class(x)=="BigMatrix.refer") bigm.internal.ncol( x ) else NCOL(x)),
               as.integer (if ( inherits(x, "BigMatrix.refer") ) bigm.row.index( x )-1 else c(1:NROW(x))-1 ),
               as.integer (if ( inherits(x, "BigMatrix.refer") ) bigm.col.index( x )-1 else c(1:NCOL(x))-1 ),

               as.integer(verbose),
               PACKAGE = "Rgtsvm");


    if ( cret$error!=0 ) stop("Error in GPU process.")

    cret$t.elapsed <- proc.time() - ptm;

    if( obj.train$nclasses > 2 ) cret$ret <- cret$ret + 1;

    return(cret);
}

gtsvmtrain.regression.call<-function(y1, x, param, final.result=FALSE, verbose=FALSE, ignoreNoProgress=FALSE)
{
    if ( inherits(x, "BigMatrix.refer") )
    {
        bigm.push(x);
        bigm.rbindcopy(x);
    }
    else
        x <- rbind(x, x);

    y <- c(y1, y1);

    nr <- nrow(x);
    if(is.null(param$maxIter))
    	param$maxIter <- NROW(x)*ifelse( NCOL(x)<=100, 100, NCOL(x) );

    if( sys.nframe()> 6  && as.character( as.list(sys.call(-5))[[1]])=="tune" )
        ignoreNoProgress <- TRUE;

     ptm <- proc.time();

     cret <- .Call ("gtsvmtrain_epsregression",
                ## X data
                as.integer (param$sparse),
                as.double  (if (param$sparse) x@ra else x),
                ## sparse index info
                as.integer64 (if (param$sparse) (x@ia)-1 else 0), #offset values start from 0
                as.integer64 (if (param$sparse) (x@ja)-1 else 0), #index values start from 1
                as.integer (NROW(x)),
                as.integer (NCOL(x)),
                as.integer (if ( class(x)=="BigMatrix.refer") bigm.internal.nrow( x ) else NROW(x)),
                as.integer (if ( class(x)=="BigMatrix.refer") bigm.internal.ncol( x ) else NCOL(x)),
                as.integer (if ( class(x)=="BigMatrix.refer") bigm.row.index( x )-1 else c(1:NROW(x))-1 ),
                as.integer (if ( class(x)=="BigMatrix.refer") bigm.col.index( x )-1 else c(1:NCOL(x))-1 ),
                ## Y data
                as.double  (y),

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
                as.integer (param$fitted),
                as.integer (param$maxIter),
                as.integer (ignoreNoProgress),

## #This part will be returned in SEXP object.
##                ## results
##                totalIter= integer  (1),
##                ## nr of support vectors
##                nr       = integer  (1),
##                ## the index of support vectors
##                index    = integer  (nr),
##                ## the labels of classes
##                labels   = integer  (param$nclass),
##                ## the support vectors of each classes
##                nSV      = integer  (param$nclass),
##                rho      = double   (1),
##                ## alpha values
##                coefs    = double(nr),
##                ## prdict labels for the fitted option
##                predict  = double   (nr),
##                error = as.integer(1),
## (End)
                as.integer(verbose),
                PACKAGE  = "Rgtsvm");

    t.elapsed <- proc.time() - ptm;

    if ( cret$error!=0 ) stop("Error in GPU process.")

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
            # indexes of sv in x
            index     = cret$index,
            ##constants in decision functions
            rho       = cret$rho,
            ## coefficiants of sv
            coefs     = matrix( cret$coefs[1:(gtsvm.class * cret$nr)], ncol = gtsvm.class ),
            totalIter = cret$totalIter,
            t.elapsed = cret$t.elapsed );
    }

    #cret$SV <- if (param$sparse) SparseM::t(SparseM::t(x[cret$index,])) else x[cret$index,,drop=F]; #copy of SV
    cret$SV <- x[cret$index[ 1:cret$nr], ];

    if ( inherits(x, "BigMatrix.refer") ) bigm.pop(x);

    return(cret);
}


gtsvmpredict.regression.call<-function( x, x.sparse, obj.train, param=list(decision.values=FALSE, probability = FALSE), verbose=FALSE )
{
    if (ncol(obj.train$SV) != ncol(x))
        stop ("test data does not match model !")

    ptm <- proc.time();

   if( is.null(obj.train$pointer ) )
	    cret <- .Call ("gtsvmpredict_epsregression",
                   as.integer (param$decision.values),
                   as.integer (param$probability),

                   ## model
                   as.integer (obj.train$sparse),
                   as.double (if (obj.train$sparse) obj.train$SV@ra else obj.train$SV ),
                   as.integer64 (if (obj.train$sparse) obj.train$SV@ia-1 else 0),
                   as.integer64 (if (obj.train$sparse) obj.train$SV@ja-1 else 0),
                   as.integer (NROW(obj.train$SV)),
                   as.integer (NCOL(obj.train$SV)),
                   as.integer (c(1:NROW(obj.train$SV))-1 ),
                   as.integer (c(1:NCOL(obj.train$SV))-1 ),
                   as.integer (obj.train$tot.nSV),

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
                   as.integer64 (if (x.sparse) x@ia-1 else 0),
                   as.integer64 (if (x.sparse) x@ja-1 else 0),
                   as.integer (nrow(x)),
                   as.integer (if ( class(x)=="BigMatrix.refer") bigm.internal.nrow( x ) else NROW(x)),
                   as.integer (if ( class(x)=="BigMatrix.refer") bigm.internal.ncol( x ) else NCOL(x)),
                   as.integer (if ( class(x)=="BigMatrix.refer") bigm.row.index( x )-1 else c(1:NROW(x))-1 ),
                   as.integer (if ( class(x)=="BigMatrix.refer") bigm.col.index( x )-1 else c(1:NCOL(x))-1 ),

## #This part will be returned in SEXP object.
##                   ## decision-values
##                   ret = double( nrow(x) ),
##                   dec = double( nrow(x)  ),
##                   prob = double( nrow(x) * obj.train$nclasses ),
##                   error = as.integer(1),
## (End)
                   as.integer(verbose),
                   PACKAGE = "Rgtsvm")
    else
	    cret <- .Call ("gtsvmpredict_epsregression_direct",
                   as.integer (param$decision.values),
                   as.integer (param$probability),

                   ## model
                   obj.train$pointer,

                   ## test matrix
                   as.integer (x.sparse),
                   as.double  (if (x.sparse) x@ra else x ),
                   as.integer64 (if (x.sparse) x@ia-1 else 0),
                   as.integer64 (if (x.sparse) x@ja-1 else 0),
                   as.integer (nrow(x)),
                   as.integer (if ( class(x)=="BigMatrix.refer") bigm.internal.nrow( x ) else NROW(x)),
                   as.integer (if ( class(x)=="BigMatrix.refer") bigm.internal.ncol( x ) else NCOL(x)),
                   as.integer (if ( class(x)=="BigMatrix.refer") bigm.row.index( x )-1 else c(1:NROW(x))-1 ),
                   as.integer (if ( class(x)=="BigMatrix.refer") bigm.col.index( x )-1 else c(1:NCOL(x))-1 ),

                   as.integer(verbose),
                   PACKAGE = "Rgtsvm");


    if ( cret$error!=0 ) stop("Error in GPU process.")

    cret$t.elapsed <- proc.time() - ptm;

    return(cret);
}


gtsvmpredict.regression.batch.call<-function( file.rds, x.count, obj.train, param=list(decision.values=FALSE, probability = FALSE), verbose=FALSE )
{
    ptm <- proc.time();

    cret <- .C ("gtsvmpredict_epsregression_batch",
                   as.integer (param$decision.values),
                   as.integer (param$probability),

                   ## model
                   as.integer (obj.train$sparse),
                   as.double  (if (obj.train$sparse) obj.train$SV@ra else obj.train$SV ),
                   as.integer64 (if (obj.train$sparse) obj.train$SV@ia-1 else 0),
                   as.integer64 (if (obj.train$sparse) obj.train$SV@ja-1 else 0),
                   as.integer (NROW(obj.train$SV)),
                   as.integer (NCOL(obj.train$SV)),
                   as.integer (c(1:NROW(obj.train$SV))-1 ),
                   as.integer (c(1:NCOL(obj.train$SV))-1 ),
                   as.integer (obj.train$tot.nSV),

                   ## model ( rho and coefs )
                   as.double  (obj.train$rho),
                   as.double  (as.vector(obj.train$coefs)),

                   ## parameter
                   as.integer (obj.train$kernel),
                   as.integer (obj.train$degree),
                   as.double  (obj.train$gamma),
                   as.double  (obj.train$coef0),
                   as.double  (obj.train$cost),
                   as.double  (obj.train$x.scale$"scaled:center"),
                   as.double  (obj.train$x.scale$"scaled:scale"),

                   ## test matrix
                   as.character ( file.rds ),
                   as.integer ( length(file.rds) ),

                   ## decision-values
                   ret = double( x.count ),
                   dec = double( x.count ),
                   prob = double( x.count * obj.train$nclasses ),

                   as.integer(verbose),
                   error = as.integer(1),
                   #DUP     = TRUE,
                   PACKAGE = "Rgtsvm");

    if ( cret$error!=0 ) stop("Error in GPU process.")

    cret$t.elapsed <- proc.time() - ptm;

    return(cret);
}

gtsvmpredict.classfication.batch.call<-function( file.rds, x.count, obj.train, param=list(decision.values=FALSE, probability = FALSE), verbose=FALSE )
{
    ptm <- proc.time();
    cret <- .C ("gtsvmpredict_classfication_batch",
                   as.integer (param$decision.values),
                   as.integer (param$probability),

                   ## model
                   as.integer (obj.train$sparse),
                   as.double  (if (obj.train$sparse) obj.train$SV@ra else obj.train$SV ),
                   as.integer64 (if (obj.train$sparse) obj.train$SV@ia-1 else 0),
                   as.integer64 (if (obj.train$sparse) obj.train$SV@ja-1 else 0),
                   as.integer (NROW(obj.train$SV)),
                   as.integer (NCOL(obj.train$SV)),
                   as.integer (c(1:NROW(obj.train$SV))-1 ),
                   as.integer (c(1:NCOL(obj.train$SV))-1 ),
                   as.integer (obj.train$tot.nSV),

                   ## model ( rho and coefs )
                   as.double  (obj.train$rho),
                   as.double  (as.vector(obj.train$coefs)),
                   as.integer (obj.train$nclasses),

                   ## parameter
                   as.integer (obj.train$kernel),
                   as.integer (obj.train$degree),
                   as.double  (obj.train$gamma),
                   as.double  (obj.train$coef0),
                   as.double  (obj.train$cost),
                   as.double  (obj.train$x.scale$"scaled:center"),
                   as.double  (obj.train$x.scale$"scaled:scale"),

                   ## test matrix
                   as.character ( file.rds ),
                   as.integer ( length(file.rds) ),

                   ## decision-values
                   ret = double( x.count ),
                   dec = double( x.count  ),
                   prob = double( x.count * obj.train$nclasses ),

                   as.integer(verbose),
                   error = as.integer(1),
                   #DUP     = TRUE,
                   PACKAGE = "Rgtsvm");

    if ( cret$error!=0 ) stop("Error in GPU process.")

    cret$t.elapsed <- proc.time() - ptm;

    if( obj.train$nclasses > 2 ) cret$ret <- cret$ret + 1;

    return(cret);
}

gtsvmpredict.loadsvm <- function( obj.train, deviceId = -1, verbose=FALSE)
{
    ptm <- proc.time();

    cret <- .Call ("gtsvmpredict_loadsvm",
               ## model
               as.integer (obj.train$sparse),
               as.double  (if (obj.train$sparse) obj.train$SV@ra else obj.train$SV ),
               as.integer64 (if (obj.train$sparse) obj.train$SV@ia-1 else 0),
               as.integer64 (if (obj.train$sparse) obj.train$SV@ja-1 else 0),
               as.integer (NROW(obj.train$SV)),
               as.integer (NCOL(obj.train$SV)),
               as.integer (c(1:NROW(obj.train$SV))-1 ),
               as.integer (c(1:NCOL(obj.train$SV))-1 ),

               as.integer (if (obj.train$type == EPSILON_SVR) 0 else obj.train$nclasses),
               as.integer (obj.train$tot.nSV),
               as.double  (obj.train$rho),
               as.double  (as.vector(obj.train$coefs)),

               ## parameter
               as.integer (obj.train$kernel),
               as.integer (obj.train$degree),
               as.double  (obj.train$gamma),
               as.double  (obj.train$coef0),
               as.double  (obj.train$cost),

               as.integer( deviceId),
               as.integer(verbose),
               PACKAGE = "Rgtsvm")

    if ( cret$error!=0 )
    {
		if(cret$error==-1)
    		stop("Error in device initialization.")
    	else
    		stop("Error in GPU computation.")
	}

    cret$t.elapsed <- proc.time() - ptm;
	return(cret);
}

gtsvmpredict.unloadsvm <- function ( obj.train, verbose=FALSE)
{
   if( is.null(obj.train$pointer ) )
        stop ("Model is not loaded into GPU node.!")

	cret <- .Call ("gtsvmpredict_unloadsvm",
               obj.train$pointer,
               as.integer(verbose),
               PACKAGE = "Rgtsvm")

    if ( cret$error!=0 ) stop("Error in GPU process.")

	return(cret);
}
