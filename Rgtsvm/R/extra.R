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

load.svmlight = function( filename, .loadbyC=TRUE )
{
    if( !.loadbyC )
    {
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

    }
    else
    {
        makemat <-.Call("get_svmlight", as.character(filename) );
        if(is.null(makemat))
            return(NULL);
    }

    class(makemat) = "numeric";
    requireNamespace("Matrix");
    yx = Matrix::sparseMatrix(i = makemat[,1], j = makemat[,2]+2, x = makemat[,3]);
    return( yx );
}



# improved list of objects
# author: Dirk Eddelbuettel
# reference: http://stackoverflow.com/questions/1358003/tricks-to-manage-the-available-memory-in-an-r-session

.ls.objects <- function (pos = 1, envir=NULL, pattern, order.by,
                        decreasing=FALSE, head=FALSE, n=5) {

    napply <- function(names, fn, missing=NA) sapply(names, function(x){
        ret <- suppressWarnings( try(fn( if(is.null(envir)) get(x, pos = pos) else get(x, envir=envir) ), TRUE) );
        if (class(ret)=="try-error") return(missing);
        ret;
        });

    if(is.null(envir))
        names <- ls( pos = pos, pattern = pattern)
    else
        names <- ls( envir = envir )

    obj.class <- napply(names, function(x) as.character(class(x))[1], "NA")
    obj.mode <- napply(names, mode)
    obj.type <- ifelse(is.na(obj.class), obj.mode, obj.class)
    obj.prettysize <- napply(names, function(x)       {
                           capture.output(format(utils::object.size(x), units = "auto")) } )
    obj.size <- napply(names, object.size )
    obj.dim <- t(napply(names, function(x)
                        as.numeric(dim(x))[1:2], c(NA,NA) ) );
    vec <- is.na(obj.dim)[, 1] & (obj.type != "function")
    obj.dim[vec, 1] <- napply(names, length)[vec]
    out <- data.frame(obj.type, obj.size, obj.prettysize, obj.dim)
    names(out) <- c("Type", "Size", "PrettySize", "Rows", "Columns")

    if (!missing(order.by))
        out <- out[order(out[[order.by]], decreasing=decreasing), ]

    if (head)
        out <- head(out, n)
    out
}

# shorthand
lsos <- function(..., n=10) {
    .ls.objects(..., order.by="Size", decreasing=TRUE, head=TRUE, n=n)
}

