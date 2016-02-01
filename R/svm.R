## Rgtsvm package,  Zhong Wang<zw355@cornell.edu>
##
## This script is a modification of the svm.R file
## from the e1071 package (version 1.6-6)
## https://cran.r-project.org/web/packages/e1071/index.html
##
## The copyright information for the e1071 package is as follows:
## Copyright (C) 2015 David Meyer <David.Meyer at R-project.org> et al.
## Licensed under GPL-2
##

C_CLASSFICATION <- 0;	
EPSILON_SVR <- 3;	


svm <- function (x, ...)
    UseMethod ("svm")

svm.formula <- function (formula, data = NULL, ..., subset, na.action = na.omit, scale = TRUE)
{
    call <- match.call()
    if (!inherits(formula, "formula"))
        stop("method is only for formula objects")
    m <- match.call(expand.dots = FALSE)
    if (identical(class(eval.parent(m$data)), "matrix"))
        m$data <- as.data.frame(eval.parent(m$data))
    m$... <- NULL
    m$scale <- NULL
    m[[1]] <- as.name("model.frame")
    m$na.action <- na.action
    m <- eval(m, parent.frame())
    Terms <- attr(m, "terms")
    attr(Terms, "intercept") <- 0
    x <- model.matrix(Terms, m)
    y <- model.extract(m, "response")
    attr(x, "na.action") <- attr(y, "na.action") <- attr(m, "na.action")
    if (length(scale) == 1)
        scale <- rep(scale, ncol(x))
    if (any(scale)) {
        remove <- unique(c(which(labels(Terms) %in%
                                 names(attr(x, "contrasts"))),
                           which(!scale)
                           )
                         )
        scale <- !attr(x, "assign") %in% remove
    }
    ret <- svm.default (x, y, scale = scale, ..., na.action = na.action)
    ret$call <- call
    ret$call[[1]] <- as.name("svm")
    ret$terms <- Terms
    if (!is.null(attr(m, "na.action")))
        ret$na.action <- attr(m, "na.action")
    class(ret) <- c("svm.formula", class(ret))
    return (ret)
}

svm.default <- function (x,
          y           = NULL,
          scale       = TRUE,
          type        = "C-classification",
          kernel      = "radial",
          degree      = 3,
          gamma       = if (is.vector(x)) 1 else 1 / ncol(x),
          coef0       = 0,
          cost        = 1,
          tolerance   = 0.001,
          epsilon     = 0.1,
          shrinking   = TRUE,
          fitted      = TRUE,
          cross       = 0,
          probability = FALSE,
          ...,
          subset,
          na.action = na.omit)
{
    library(bit64);

    if(inherits(x, "Matrix")) {
        library("SparseM")
        library("Matrix")
        x <- as(x, "matrix.csr")
    }
    
    if(inherits(x, "simple_triplet_matrix")) {
        library("SparseM")
        ind <- order(x$i, x$j)
        x <- new("matrix.csr",
                 ra = x$v[ind],
                 ja = x$j[ind],
                 ia = as.integer(cumsum(c(1, tabulate(x$i[ind])))),
                 dimension = c(x$nrow, x$ncol))
    }
    
    if (sparse <- inherits(x, "matrix.csr"))
    {
    	library("SparseM")
	}
	
    ## NULL parameters?
    if(is.null(degree)) stop(sQuote("degree"), " must not be NULL!")
    if(is.null(gamma)) stop(sQuote("gamma"), " must not be NULL!")
    if(is.null(coef0)) stop(sQuote("coef0"), " must not be NULL!")
    if(is.null(cost)) stop(sQuote("cost"), " must not be NULL!")
    if(is.null(epsilon)) stop(sQuote("epsilon"), " must not be NULL!")
    if(is.null(tolerance)) stop(sQuote("tolerance"), " must not be NULL!")

    x.scale <- y.scale <- NULL
    formula <- inherits(x, "svm.formula")

    ## only support C-classification
    if (is.null(type)) 
        type <- ifelse ( is.factor(y), "C-classification", "eps-regression");

	type.name <- type;
	type <- C_CLASSFICATION;
    	
    if (type.name != "C-classification" && type.name != "eps-regression") 
    	stop("Rgtsvm only support C-classification and eps-regression!")
	else
		if(type.name == "eps-regression") type <- EPSILON_SVR;
		
	## kernel type 
    kernel <- pmatch(kernel, c("linear",
                               "polynomial",
                               "radial",
                               "sigmoid"), 99) - 1
    if (kernel > 10) stop("wrong kernel specification!")

    nac <- attr(x, "na.action")

    ## scaling, and NA handling
    if (sparse) {
        scale <- rep(FALSE, ncol(x))
        if(!is.null(y)) na.fail(y)
        x <- SparseM::t(SparseM::t(x)) ## make shure that col-indices are sorted
    } else {
        x <- as.matrix(x)

        ## na-handling for matrices
        if (!formula) {
            if (!missing(subset)) {
                x <- x[subset,]
                y <- y[subset]
                if (!is.null(xhold))
                    xhold <- as.matrix(xhold)[subset,]
            }
        
            if (is.null(y))
                x <- na.action(x)
            else {
                df <- na.action(data.frame(y, x))
                y <- df[,1]
                x <- as.matrix(df[,-1])
                nac <-
                    attr(x, "na.action") <-
                        attr(y, "na.action") <-
                            attr(df, "na.action")
            }
        }

        ## scaling
        if (length(scale) == 1)
            scale <- rep(scale, ncol(x))
        if (any(scale)) {
            co <- !apply(x[,scale, drop = FALSE], 2, var)
            if (any(co)) {
                warning(paste("Variable(s)",
                              paste(sQuote(colnames(x[,scale,
                                                      drop = FALSE])[co]),
                                    sep="", collapse=" and "),
                              "constant. Cannot scale data.")
                        )
                scale <- rep(FALSE, ncol(x))
            } else {
                xtmp <- scale(x[,scale])
                x[,scale] <- xtmp
                x.scale <- attributes(xtmp)[c("scaled:center","scaled:scale")]
				# scale Y for regression
                if (is.numeric(y) && (type>2)) {
                    y <- scale(y)
                    y.scale <- attributes(y)[c("scaled:center","scaled:scale")]
                    y <- as.vector(y)
                }
            }
        }
    }

    ## further parameter checks
    nr <- nrow(x)
    if (cross > nr)
        stop(sQuote("cross"), " cannot exceed the number of observations!")

    if ( length(y) != nr )
        stop("x and y don't match.")

    if (!is.vector(y) && !is.factor (y) )
        stop("y must be a vector or a factor.")
	
	#### C/C++ part of rgtTrain requires the Y is sorted by -1 and 1
	#### y.idx will be used later! 

	y.org <- y;
	y.idx <- c();
	for( y0 in sort(unique(y) ) )
		y.idx <- c( y.idx, which( y == y0 ) );
	y <- y[y.idx];
	x <- x[y.idx,];

    lev <- NULL
    ## in case of classification: transform factors into integers
    
    if( type < EPSILON_SVR )
    {
		if (is.factor(y)) {
			lev <- levels(y)
		} else {
			if(any(as.integer(y) != y))
			   stop("dependent variable has to be of factor or integer type for classification mode.")
			y <- as.factor(y)
			lev <- levels(y)
		}
	}
	
    nclass <- length(lev);
	
    if (is.null(type)) stop("type argument must not be NULL!")
    if (is.null(kernel)) stop("kernel argument must not be NULL!")
    if (is.null(degree)) stop("degree argument must not be NULL!")
    if (is.null(gamma)) stop("gamma argument must not be NULL!")
    if (is.null(coef0)) stop("coef0 seed argument must not be NULL!")
    if (is.null(cost)) stop("cost argument must not be NULL!")
    if (is.null(tolerance)) stop("tolerance argument must not be NULL!")
    if (is.null(epsilon)) stop("epsilon argument must not be NULL!")
    if (is.null(shrinking)) stop("shrinking argument must not be NULL!")
    if (is.null(cross)) stop("cross argument must not be NULL!")
    if (is.null(sparse)) stop("sparse argument must not be NULL!")
    if (is.null(probability)) stop("probability argument must not be NULL!")
	
	biased <- ifelse(nclass<=2, TRUE, FALSE);

	param <- list(type=type, type.name = type.name, kernel=kernel, degree=degree, gamma=gamma, 
			 coef0=coef0, cost=cost, tolerance=tolerance, epsilon=epsilon, 
			 shrinking=shrinking, cross=cross, sparse=sparse, probability=probability,
			 biased = biased, fitted=fitted, nclass=nclass );
	
	if( type == C_CLASSFICATION)
		cret <- gtsvmtrain.classfication.call( y, x, param );
	if( type == EPSILON_SVR)
		cret <- gtsvmtrain.regression.call( y, x, param );

    cret$index  <- cret$index[1:cret$nr]
    gtsvm.class <- ifelse( cret$nclasses==2, 1, cret$nclasses );

    ret <- list (
                 call      = match.call(),
                 type.name = type.name,
                 type      = type,
                 kernel    = kernel,
                 cost      = cost,
                 degree    = degree,
                 gamma     = gamma,
                 coef0     = coef0,
				 
				 tolerance = tolerance,
                 epsilon   = epsilon,
                 sparse    = sparse,
                 scaled    = scale,
                 x.scale   = x.scale,
                 y.scale   = y.scale,
                 biased    = biased,

				 #number of classes
                 nclasses  = cret$nclasses,  
                 levels    = lev,
                 # total number of sv
                 tot.nSV   = cret$nr, 		

                 # number of SV in diff. classes
                 nSV       = cret$nSV[1:cret$nclasses], 
                 
                 # labels of the SVs.
                 labels    = cret$label[1:cret$nclasses], 
                 
                 SV        = if (sparse) SparseM::t(SparseM::t(x[cret$index,])) else x[cret$index,], #copy of SV
                 y.SV      = y[cret$index],
                 # indexes of sv in x
                 index     = cret$index,  
				 	
                 ##constants in decision functions
                 rho       = cret$rho[1:(cret$nclasses * (cret$nclasses - 1) / 2)],

                 ## coefficiants of sv
                 coefs    = matrix( cret$coefs[1:(gtsvm.class * cret$nr)], ncol = gtsvm.class ),

		         totalIter = cret$totalIter,
		         t.elapsed = cret$t.elapsed,
                 na.action = nac );

    if (cross > 0)
	{
		cross.ret <- cross_validation( y, x, param );
		
        if ( type > 2) {
            scale.factor     <- if (any(scale)) crossprod(y.scale$"scaled:scale") else 1;
            ret$MSE          <- cross.ret$cresults * scale.factor;
            ret$tot.MSE      <- cross.ret$ctotal1  * scale.factor;
            ret$scorrcoeff   <- cross.ret$ctotal2;
        } else {
            ret$accuracies   <- cross.ret$cresults;
            ret$tot.accuracy <- cross.ret$ctotal1;
        }
	}
	
	class (ret) <- "gtsvm"

    if (fitted) {
    	if(type == C_CLASSFICATION)
    	{
			org.idx <- sort.int(y.idx, index.return=T)$ix;
			ret$fitted <- as.factor(cret$predict[org.idx]) ;
			levels( ret$fitted ) <- lev;

			ret$decision.values <- attr(ret$fitted, "decision.values")
			attr(ret$fitted, "decision.values") <- NULL;

			ret$fitted.accuracy <- length( which( ret$fitted == y.org ) )/length(y)
       	}
       	else
       	{
			org.idx <- sort.int(y.idx, index.return=T)$ix;

			ret$decision.values <- matrix( cret$predict[org.idx], ncol=1) ;
			ret$fitted <- cret$predict[org.idx];
			y1 <- y[org.idx]
			
			if(!is.null(y.scale))
			{
				ret$fitted <- ret$fitted * y.scale$"scaled:scale" + y.scale$"scaled:center";
				y1 <- y1 * y.scale$"scaled:scale" + y.scale$"scaled:center";
			}
			
	    	y1.v <- ret$fitted;
	    	ret$fitted.MSE <- sum(( y1 - y1.v )^2, na.rm=T)/length(y1);
		    ret$fitted.r2  <- ( length(y1)* sum(y1.v*y1, na.rm=T) - sum(y1.v, na.rm=T)*sum(y1, na.rm=T) )^2 / ( length(y1)*sum(y1.v^2, na.rm=T) - (sum(y1.v, na.rm=T))^2) / ( length(y1)*sum(y1^2, na.rm=T)- (sum(y1, na.rm=T))^2);
		    ret$residuals <- (y1 - y1.v)
       	}
    }

    ret
}

# Cross-Validation-routine from svm-train in R code
cross_validation <- function ( y, x, param )
{
	idx.shuffle <- sample(1:NROW(y));
	y <- y[idx.shuffle ];
	x <- x[idx.shuffle,];
	y.v <- rep(NA, length(y));
	
	breaks <- round(seq(1, length(y), length.out=param$cross+1));
	cresults <- c();
	
	for(i in 1:(length(breaks)-1))
	{
		idx.cross <- c(breaks[i]:breaks[i+1]);
		if( param$type == C_CLASSFICATION)
		{
			sret <- gtsvmtrain.classfication.call( y[ -idx.cross ], x[ -idx.cross,], param, final.result=TRUE, verbose=FALSE, ignoreNoProgress=TRUE );
			pret <- gtsvmpredict.classfication.call( x[ idx.cross,], param$sparse, sret, verbose=FALSE );

			if( sret$nclasses==2 )
				y.v [idx.cross] <- levels(y)[as.factor(pret$ret)]
			else
			{
				ret2 <- matrix( pret$dec[ 1:(length(idx.cross)*param$nclass) ], 
								nrow = length(idx.cross), ncol= param$nclass  );
				y.v [idx.cross] <- apply(ret2, 1, which.max);
			}

			cresults[i] = 100.0 * sum( as.integer(y.v [idx.cross] == y[idx.cross]) )/length(idx.cross);
		}
		
		if( param$type == EPSILON_SVR)
		{
			sret <- gtsvmtrain.regression.call( y[ -idx.cross ], x[ -idx.cross,], param, final.result=TRUE, verbose=FALSE, ignoreNoProgress=TRUE );
			pret <- gtsvmpredict.regression.call( x[ idx.cross,], param$sparse, sret, verbose=FALSE);
			
			y.v [idx.cross] <- pret$ret;
			cresults[i] = sum(( y.v[idx.cross] - y[idx.cross])^2 )/length(idx.cross)
		}
	}

	if(param$type == EPSILON_SVR )
	{
	    # MSE
	    ctotal1 <- sum((y.v-y)^2)/length(y);
	    # R2
	    ctotal2 <- (length(y)* sum(y.v*y) - sum(y.v)*sum(y) )^2 / ( length(y)*sum(y.v^2) - (sum(y.v))^2) / ( length(y)*sum(y^2)- (sum(y))^2);
	}
	else
	{
		ctotal1 <- 100.0 * sum(y.v==y)/length(y);
	    ctotal2 <- NA;
	}    
	    
	return(list(ctotal1=ctotal1, ctotal2=ctotal2, cresults=cresults));
}


predict.gtsvm <- function (object, newdata,
          decision.values = FALSE,
          probability = FALSE,
          ...,
          na.action = na.omit)
{
    library(bit64);

    if (missing(newdata))
        return(fitted(object))

    if (object$tot.nSV < 1)
        stop("Model is empty!")

    if(inherits(newdata, "Matrix")) {
        library("SparseM")
        library("Matrix")
        newdata <- as(newdata, "matrix.csr")
    }
    if(inherits(newdata, "simple_triplet_matrix")) {
       library("SparseM")
       ind <- order(newdata$i, newdata$j)
       newdata <- new("matrix.csr",
                      ra = newdata$v[ind],
                      ja = newdata$j[ind],
                      ia = as.integer(cumsum(c(1, tabulate(newdata$i[ind])))),
                      dimension = c(newdata$nrow, newdata$ncol))
   }

    sparse <- inherits(newdata, "matrix.csr")
    if (object$sparse || sparse)
        library("SparseM")

    act <- NULL
    if ((is.vector(newdata) && is.atomic(newdata)))
        newdata <- t(t(newdata))
        
    if (sparse)
        newdata <- SparseM::t(SparseM::t(newdata))
        
    preprocessed <- !is.null(attr(newdata, "na.action"))
    rowns <- if (!is.null(rownames(newdata))) 
		    	rownames(newdata)
		    else
		        1:nrow(newdata);
        
    if (!object$sparse) {
        if (inherits(object, "svm.formula")) {
            if(is.null(colnames(newdata)))
                colnames(newdata) <- colnames(object$SV)
            newdata <- na.action(newdata)
            act <- attr(newdata, "na.action")
            newdata <- model.matrix(delete.response(terms(object)),
                                    as.data.frame(newdata))
        } else {
            newdata <- na.action(as.matrix(newdata))
            act <- attr(newdata, "na.action")
        }
    }

    if (!is.null(act) && !preprocessed)
        rowns <- rowns[-act];

    if (any(object$scaled))
        newdata[,object$scaled] <-
            scale(newdata[,object$scaled, drop = FALSE],
                  center = object$x.scale$"scaled:center",
                  scale  = object$x.scale$"scaled:scale")

    if (ncol(object$SV) != ncol(newdata))
        stop ("test data does not match model !")
	
	param <- list( decision.values = decision.values, probability = probability );
	
	# Call C/C++ interface to do predict
	if(object$type == C_CLASSFICATION)
		ret <- gtsvmpredict.classfication.call( newdata, sparse, object, param)
	else if(object$type == EPSILON_SVR)
		ret <- gtsvmpredict.regression.call( newdata, sparse, object, param)
	else
		stop("only 'C-classification' and 'eps-regression' are implemented in this package!");
		
	ret2 <- ret$ret;
	if (is.character(object$levels)) # classification: return factors
    {
    	ret2 <- as.factor(ret$ret) ;
        levels( ret2 ) <- object$levels;
	}
    else if (any(object$scaled) && !is.null(object$y.scale)) # return raw values, possibly scaled back
        ret2 <- ret$ret * object$y.scale$"scaled:scale" + object$y.scale$"scaled:center"
	
    names(ret2) <- rowns
    ret2 <- napredict(act, ret2);

    if (decision.values) 
    {
        colns = c()
        for (i in 1:(object$nclasses - 1))
            for (j in (i + 1):object$nclasses)
                colns <- c(colns, 
                           paste(object$levels[object$labels[i]],
                                 "/", object$levels[object$labels[j]],
                                 sep = ""))
        attr(ret2, "decision.values") <-
            napredict(act, matrix(ret$dec, nrow = nrow(newdata), byrow = TRUE, dimnames = list(rowns, colns) ) )
    }

    if (probability && object$type < 2) 
    {
        if (!object$compprob)
            warning("SVM has not been trained using `probability = TRUE`, probabilities not available for predictions.")
        else
            attr(ret2, "probabilities") <-
                napredict(act, matrix(ret$prob, nrow = nrow(newdata), byrow = TRUE,
                			 		  dimnames = list(rowns, object$levels[object$labels]) ) )
    }

    ret2
}

print.gtsvm <- function (x, ...)
{
    cat("\nCall:", deparse(x$call, 0.8 * getOption("width")), "\n", sep="\n")
    cat("Parameters:\n")
    cat("   SVM-Type: ", x$type.name, "\n")
    cat(" SVM-Kernel: ", c("linear",
                           "polynomial",
                           "radial",
                           "sigmoid")[x$kernel+1], "\n")

    cat("       cost: ", x$cost, "\n")
    if (x$kernel==1)
        cat("     degree: ", x$degree, "\n")
    cat("      gamma: ", x$gamma, "\n")
    if (x$kernel==1 || x$kernel==3)
        cat("     coef.0: ", x$coef0, "\n")

    cat("    tolerance: ", x$tolerance, "\n")
	cat(" time elapsed: ", x$t.elapsed[3], "\n\n");
	
    cat("\nNumber of Support Vectors: ", x$tot.nSV)
    cat("\n\n")

}

summary.gtsvm <- function(object, ...)
    structure(object, class="summary.gtsvm")

print.summary.gtsvm <- function (x, ...)
{
    print.gtsvm(x)

    cat(" (", x$nSV, ")\n\n")
    cat("\nNumber of Classes: ", x$nclasses, "\n\n")
    cat("Levels:", if(is.numeric(x$levels)) "(as integer)", "\n", x$levels)

    cat("\n\n")
}

scale.data.frame <- function(x, center = TRUE, scale = TRUE)
{
    i <- sapply(x, is.numeric)
    if (ncol(x[, i, drop = FALSE])) {
        x[, i] <- tmp <- scale.default(x[, i, drop = FALSE], na.omit(center), na.omit(scale))
        if(center || !is.logical(center))
            attr(x, "scaled:center")[i] <- attr(tmp, "scaled:center")
        if(scale || !is.logical(scale))
            attr(x, "scaled:scale")[i]  <- attr(tmp, "scaled:scale")
    }
    x
}

plot.gtsvm <- function(x, data, formula = NULL, fill = TRUE,
         grid = 50, slice = list(), symbolPalette = palette(),
         svSymbol = "x", dataSymbol = "o", ...)
{
        if (is.null(formula) && ncol(data) == 3) {
            formula <- formula(delete.response(terms(x)))
            formula[2:3] <- formula[[2]][2:3]
        }
        
        if (is.null(formula))
            stop("missing formula.")
        
        if (fill) 
        {
            sub <- model.frame(formula, data)
            xr <- seq(min(sub[, 2]), max(sub[, 2]), length = grid)
            yr <- seq(min(sub[, 1]), max(sub[, 1]), length = grid)
            l <- length(slice)
            
            if (l < ncol(data) - 3) 
            {
                slnames <- names(slice)
                slice <- c(slice, rep(list(0), ncol(data) - 3 - l))
                names <- labels(delete.response(terms(x)))
                names(slice) <- c(slnames, names[!names %in% c(colnames(sub), slnames)])
            }
            
            for (i in names(which(sapply(data, is.factor))))
                if (!is.factor(slice[[i]])) {
                    levs <- levels(data[[i]])
                    lev <- if (is.character(slice[[i]])) slice[[i]] else levs[1]
                    fac <- factor(lev, levels = levs)
                    if (is.na(fac))
                        stop(paste("Level", dQuote(lev), "could not be found in factor", sQuote(i)))
                    slice[[i]] <- fac
                }

            lis <- c(list(yr), list(xr), slice);
            names(lis)[1:2] <- colnames(sub);
            new <- expand.grid(lis)[, labels(terms(x))];
            preds <- predict(x, new);

            filled.contour(xr, yr,
                           matrix(as.numeric(preds),
                                  nrow = length(xr), byrow = TRUE),
                           plot.axes = {
                               axis(1)
                               axis(2)
                               colind <- as.numeric(model.response(model.frame(x, data)))
                               dat1 <- data[-x$index,]
                               dat2 <- data[x$index,]
                               coltmp1 <- symbolPalette[colind[-x$index]]
                               coltmp2 <- symbolPalette[colind[x$index]]
                               points(formula, data = dat1, pch = dataSymbol, col = coltmp1)
                               points(formula, data = dat2, pch = svSymbol, col = coltmp2)
                           },
                           levels = 1:(length(levels(preds)) + 1),
                           key.axes = axis(4, 1:(length(levels(preds))) + 0.5,
                           labels = levels(preds),
                           las = 3),
                           plot.title = title(main = "SVM classification plot",
                           xlab = names(lis)[2], ylab = names(lis)[1]), ...);
        }
        else {
            plot(formula, data = data, type = "n", ...)
            colind <- as.numeric(model.response(model.frame(x, data)))
            
            dat1 <- data[-x$index,]
            dat2 <- data[x$index,]
            coltmp1 <- symbolPalette[colind[-x$index]]
            coltmp2 <- symbolPalette[colind[x$index]]
            points(formula, data = dat1, pch = dataSymbol, col = coltmp1)
            points(formula, data = dat2, pch = svSymbol, col = coltmp2)
            
            invisible()
        }
}

load.svmlight = function( filename ) 
{
	require(Matrix)
  	content = readLines( filename )
  	num_lines = length( content )
  	tomakemat = cbind(1:num_lines, -1, unlist(lapply(1:num_lines, function(i){ strsplit( content[i], ' ' )[[1]][1]} )))

  	# loop over lines
  	makemat = rbind(tomakemat,
		do.call(rbind, 
			lapply(1:num_lines, function(i){
				# split by spaces, remove lines
				line = as.vector( strsplit( content[i], ' ' )[[1]])
				cbind(i, t(simplify2array(strsplit(line[-1], ':'))))   
		})))
	
	class(makemat) = "numeric"
	
	
	yx = sparseMatrix(i = makemat[,1], j = makemat[,2]+2, x = makemat[,3])
	return( yx );
}
