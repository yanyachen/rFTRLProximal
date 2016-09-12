#' @title Subsetting dgCMatrix
#'
#' @description
#' Returns subsets of a \code{dgCMatrix}.
#'
#' @param x a transposed \code{dgCMatrix}.
#' @param i logical expression indicating elements or rows to keep.
#' @return A \code{dgCMatrix} containing the subset of rows that are selected.
#' @export

slice <- function(x, i) {
  x_slice <- x[,i]
  attr(x_slice, "mapping") <- attr(x, "mapping")
  return(x_slice)
}
