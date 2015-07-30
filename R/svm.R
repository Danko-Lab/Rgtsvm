## Rgtsvm package,  Zhong Wang<zw355@cornell.edu>
##
## This scipt is a modification of the svm.R file
## from the e1071 package (version 1.6-6)
## https://cran.r-project.org/web/packages/e1071/index.html
##
## The copyright information for the e1071 package is as follows:
## Copyright (C) 2015 David Meyer <David.Meyer at R-project.org> et al.
## Licensed under GPL-2
##

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
          class.weights = NULL,
          tolerance   = 0.001,
          epsilon     = 0.1,
          shrinking   = TRUE,
          fitted      = TRUE,
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

    xhold   <- if (fitted) x else NA
    x.scale <- y.scale <- NULL
    formula <- inherits(x, "svm.formula")

    ## only support C-classification
    if (is.null(type)) type <- "C-classification";
    type <- pmatch(type, c("C-classification",
                           "nu-classification",
                           "one-classification",
                           "eps-regression",
                           "nu-regression"), 99) - 1
    if (type > 0) 
    	stop("Rgtsvm onpy support C-classification!")

	## kernel type 
    kernel <- pmatch(kernel, c("linear",
                               "polynomial",
                               "radial",
                               "sigmoid"), 99) - 1
    if (kernel > 10) stop("wrong kernel specification!")

    nac <- attr(x, "na.action")

    ## scaling, subsetting, and NA handling
    if (sparse) {
        scale <- rep(FALSE, ncol(x))
        if(!is.null(y)) na.fail(y)
        x <- SparseM::t(SparseM::t(x)) ## make shure that col-indices are sorted
    } else {
        x <- as.matrix(x)

        ## subsetting and na-handling for matrices
        if (!formula) {
            if (!missing(subset)) x <- x[subset,]
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
                if (is.numeric(y) && (type > 2)) {
                    y <- scale(y)
                    y.scale <- attributes(y)[c("scaled:center","scaled:scale")]
                    y <- as.vector(y)
                }
            }
        }
    }

    ## further parameter checks
    nr <- nrow(x)

    if (!is.vector(y) && !is.factor (y) && type != 2)
        stop("y must be a vector or a factor.")
    if ( length(y) != nr )
        stop("x and y don't match.")

    lev <- NULL
    weightlabels <- NULL

    ## in case of classification: transform factors into integers
    if (is.factor(y)) {
        lev <- levels(y)
        y.new <- as.integer(y)
        if (!is.null(class.weights)) {
            if (is.null(names(class.weights)))
                stop("Weights have to be specified along with their according level names !")
            weightlabels <- match (names(class.weights), lev)
            if (any(is.na(weightlabels)))
                stop("At least one level name is missing or misspelled.")
         }
    } else {
        if (type < 3) {
            if(any(as.integer(y) != y))
                  stop("dependent variable has to be of factor or integer type for classification mode.")
              y.new <- as.factor(y)
              lev <- levels(y.new)
        } 
        else 
          	lev <- unique(y)
    }
	
    nclass <- 2
    if (type < 2) nclass <- length(lev);
    
	y.idx <- c();
	for( y0 in unique(y) )
		y.idx <- c( y.idx, which( y == y0 ) );
	y <- y[y.idx];
	x <- x[y.idx,];

    if (is.null(type)) stop("type argument must not be NULL!")
    if (is.null(kernel)) stop("kernel argument must not be NULL!")
    if (is.null(degree)) stop("degree argument must not be NULL!")
    if (is.null(gamma)) stop("gamma argument must not be NULL!")
    if (is.null(coef0)) stop("coef0 seed argument must not be NULL!")
    if (is.null(cost)) stop("cost argument must not be NULL!")
    if (is.null(tolerance)) stop("tolerance argument must not be NULL!")
    if (is.null(epsilon)) stop("epsilon argument must not be NULL!")
    if (is.null(shrinking)) stop("shrinking argument must not be NULL!")
    if (is.null(sparse)) stop("sparse argument must not be NULL!")

    err <- empty_string <- paste(rep(" ", 255), collapse = "")
    
    maxIter <- nr * 100;
	ptm <- proc.time()

    cret <- .C ("gtsvmtrain",
                ## data
                as.double  (if (sparse) x@ra else x),
                as.integer (nr), 
                as.integer(ncol(x)),
                as.double  (y),
                ## sparse index info
                as.integer64 (if (sparse) (x@ia)-1 else 0), #offset values start from 0
                as.integer64 (if (sparse) (x@ja)-1 else 0),   #index values start from 1

                ## parameters
                as.integer (sparse),
                as.integer (kernel),
                as.integer (nclass),
                ## kernelParameter 3
                as.integer (degree),
                ## kernelParameter 1
                as.double  (gamma),
                ## kernelParameter 2
				as.double  (coef0),
                ## regularization
				as.double  (cost),
                as.double  (tolerance),
                as.integer (shrinking),
                as.integer (maxIter),

                ## results
                nclasses = integer  (1),
                nr       = integer  (1), # nr of support vectors
                index    = integer  (nr),
                labels   = integer  (nclass),
                nSV      = integer  (nclass),
                rho      = double   (nclass * (nclass - 1) / 2),
                
                trainingAlphas             = double   (nr * nclass ),
                trainingResponses          = double   (nr * nclass ),
                trainingNormsSquared       = double   (nr),
                trainingKernelNormsSquared = double   (nr),

		        totalIter= integer  (1),
                error    = err,
                PACKAGE  = "Rgtsvm");
                
	show(proc.time() - ptm);

    if (cret$error != empty_string)
        stop(paste(cret$error, "!", sep=""))

    cret$index  <- cret$index[1:cret$nr]
    gtsvm.class <- ifelse(cret$nclasses==2, 1, cret$nclasses );

    ret <- list (
                 call     = match.call(),
                 type     = type,
                 kernel   = kernel,
                 cost     = cost,
                 degree   = degree,
                 gamma    = gamma,
                 coef0    = coef0,
				 
                 tolerance= tolerance,
                 sparse   = sparse,
                 scaled   = scale,
                 x.scale  = x.scale,
                 y.scale  = y.scale,

				 #number of classes
                 nclasses = cret$nclasses,  
                 levels   = lev,
                 # total number of sv
                 tot.nSV  = cret$nr, 		

                 # number of SV in diff. classes
                 nSV      = cret$nSV[1:cret$nclasses], 
                 
                 # labels of the SVs.
                 labels   = cret$label[1:cret$nclasses], 
                 
                 SV       = if (sparse) SparseM::t(SparseM::t(x[cret$index,])) else t(t(x[cret$index,])), #copy of SV
                 # indexes of sv in x
                 index    = cret$index,  

                 ##constants in decision functions
                 rho      = cret$rho[1:(cret$nclasses * (cret$nclasses - 1) / 2)],

                 ## coefficiants of sv
                 trainingAlphas    = matrix( cret$trainingAlphas[1:(gtsvm.class * cret$nr)], ncol = gtsvm.class ),
                 trainingResponses = matrix( cret$trainingResponses[1:(gtsvm.class * cret$nr)], ncol = gtsvm.class ),
                 trainingNormsSquared = cret$trainingNormsSquared[ 1:cret$nr ],
                 trainingKernelNormsSquared = cret$trainingKernelNormsSquared[ 1:cret$nr],

		         totalIter = cret$totalIter,
		         time      = proc.time() - ptm,
                 na.action = nac );

    class (ret) <- "gtsvm"

    if (fitted) {
        ret$fitted <- na.action(predict(ret, xhold,
                                        decision.values = TRUE))
        ret$decision.values <- attr(ret$fitted, "decision.values")
        attr(ret$fitted, "decision.values") <- NULL;
        if(cret$nclasses==2)
        	ret$correct <- length( which( ( ret$fitted * y )==1))/length(y)
        else	
        	ret$correct <- c();
    }

    ret
}

predict.gtsvm <- function (object, newdata,
          decision.values = FALSE,
          score = FALSE,
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
        1:nrow(newdata)
        
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
        rowns <- rowns[-act]

    if (any(object$scaled))
        newdata[,object$scaled] <-
            scale(newdata[,object$scaled, drop = FALSE],
                  center = object$x.scale$"scaled:center",
                  scale  = object$x.scale$"scaled:scale"
                  )

    if (ncol(object$SV) != ncol(newdata))
        stop ("test data does not match model !")

    err <- empty_string <- paste(rep(" ", 255), collapse = "")
	
	y.fake <- c();
	for( i in 1:length(object$labels) )
		y.fake <- c(y.fake, rep( as.numeric(object$labels[i]), object$nSV[i]));

	ptm <- proc.time()

    ret <- .C ("gtsvmpredict",
               as.integer (decision.values),
               as.integer (score),

               ## model
               as.integer (object$sparse),
               as.double  (if (object$sparse) object$SV@ra else object$SV ),
               as.integer (nrow(object$SV)), 
               as.integer (ncol(object$SV)),
               as.integer64 (if (object$sparse) object$SV@ia-1 else 0),
               as.integer64 (if (object$sparse) object$SV@ja-1 else 0),

               as.integer (object$nclasses),
               as.integer (object$tot.nSV),
               as.double (y.fake),

               as.double  (as.vector(object$trainingAlphas)),
               as.double  (as.vector(object$trainingResponses)),
               as.double  (as.vector(object$trainingNormsSquared)),
               as.double  (as.vector(object$trainingKernelNormsSquared)),

               ## parameter
               as.integer (object$kernel),
               as.integer (object$degree),
               as.double  (object$gamma),
               as.double  (object$coef0),
               as.double  (object$cost),

               ## test matrix
               as.integer (sparse),
               as.double  (if (sparse) newdata@ra else newdata ),
               as.integer (nrow(newdata)),
               as.integer64 (if (sparse) newdata@ia-1 else 0),
               as.integer64 (if (sparse) newdata@ja-1 else 0),

               ## decision-values
               ret = double( nrow(newdata) ),
               dec = double( nrow(newdata) * object$nclasses ),
               prob = double( nrow(newdata) * object$nclasses ),

               error    = err,
               PACKAGE = "Rgtsvm");

	show(proc.time() - ptm);

    gtsvm.class <- ifelse(object$nclasses==2, 1, object$nclasses );
	ret2 <- matrix( ret$dec[ 1:(nrow(newdata)*gtsvm.class) ], nrow = nrow(newdata), ncol= gtsvm.class  );
	
	if( !score )
	{
		ret2 <- if ( is.character(object$levels) && length(object$levels)> 2 ) # classification: return factors
			factor (object$levels[ret$ret], levels = object$levels)
		else if (any(object$scaled) && !is.null(object$y.scale)) # return raw values, possibly scaled back
			ret$ret * object$y.scale$"scaled:scale" + object$y.scale$"scaled:center"
		else
			ret$ret

	    #names(ret2) <- rowns
	    #ret2 <- napredict(act, ret2)
	}
	

    ret2
}

print.gtsvm <- function (x, ...)
{
    cat("\nCall:", deparse(x$call, 0.8 * getOption("width")), "\n", sep="\n")
    cat("Parameters:\n")
    cat("   SVM-Type: ", c("C-classification",
                           "nu-classification",
                           "one-classification",
                           "eps-regression",
                           "nu-regression")[x$type+1], "\n")
    cat(" SVM-Kernel: ", c("linear",
                           "polynomial",
                           "radial",
                           "sigmoid")[x$kernel+1], "\n")
    if (x$type==0 || x$type==3 || x$type==4)
        cat("       cost: ", x$cost, "\n")
    if (x$kernel==1)
        cat("     degree: ", x$degree, "\n")
    cat("      gamma: ", x$gamma, "\n")
    if (x$kernel==1 || x$kernel==3)
        cat("     coef.0: ", x$coef0, "\n")
    if (x$type==1 || x$type==2 || x$type==4)
        cat("         nu: ", x$nu, "\n")
    if (x$type==3) {
        cat("    epsilon: ", x$epsilon, "\n\n")
        if (x$compprob)
            cat("Sigma: ", x$sigma, "\n\n")
    }

    cat("\nNumber of Support Vectors: ", x$tot.nSV)
    cat("\n\n")

}

summary.gtsvm <- function(object, ...)
    structure(object, class="summary.gtsvm")

print.summary.gtsvm <- function (x, ...)
{
    print.gtsvm(x)
    if (x$type<2) {
        cat(" (", x$nSV, ")\n\n")
        cat("\nNumber of Classes: ", x$nclasses, "\n\n")
        cat("Levels:", if(is.numeric(x$levels)) "(as integer)", "\n", x$levels)
    }
    cat("\n\n")
    if (x$type==2) cat("\nNumber of Classes: 1\n\n\n")

    if ("MSE" %in% names(x)) {
        cat(length (x$MSE), "-fold cross-validation on training data:\n\n", sep="")
        cat("Total Mean Squared Error:", x$tot.MSE, "\n")
        cat("Squared Correlation Coefficient:", x$scorrcoef, "\n")
        cat("Mean Squared Errors:\n", x$MSE, "\n\n")
    }
    if ("accuracies" %in% names(x)) {
        cat(length (x$accuracies), "-fold cross-validation on training data:\n\n", sep="")
        cat("Total Accuracy:", x$tot.accuracy, "\n")
        cat("Single Accuracies:\n", x$accuracies, "\n\n")
    }
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
    if (x$type < 3) 
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
}

load.svmlight <- function( file.svmlight )
{
	con <- file(file.svmlight, "rt")

	lines <- readLines( con )

	vfec <- c();
	lab  <- c();

	for( j in 1:length(lines)) 
	{
		str <- strsplit( lines[j], " " )

		sss <- strsplit( str [[1]] [-1], ":")
		lab <- c( lab, str[[1]][1] )

		for (i in 1:length(sss))
			vfec <- rbind( vfec,  c( j, sss[[i]]) );
	}

	row.n<- max( as.numeric(vfec[,1]))
	col.n<- max( as.numeric(vfec[,2]))

	Mat <- matrix( data= NA, nrow = row.n, ncol = col.n)

	v0 <- matrix( as.numeric(vfec), ncol=3)

	for(k in 1:nrow(v0))
		Mat[v0[k,1], v0[k,2]] <- v0[k,3]

	M.new <- data.frame(lab, Mat)
	
	return( M.new );
}
