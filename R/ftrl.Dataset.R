#' Dimensions of ftrl.Dataset
#'
#' Returns a vector of numbers of rows and of columns in an \code{ftrl.Dataset}.
#' @param x Object of class \code{ftrl.Dataset}
#'
#' @details
#' Note: since \code{nrow} and \code{ncol} internally use \code{dim}, they can also
#' be directly used with an \code{ftrl.Dataset} object.
#'
#' @export
dim.ftrl.Dataset <- function(x) {
  c(ncol(x$X), nrow(x$X))
}

#' Print ftrl.Dataset
#'
#' Print information about ftrl.Dataset
#' Currently it displays dimensions and presence of feature mapping.
#'
#' @param x a object of \code{ftrl.Dataset}.
#' @param verbose whether to print mapping
#' @param ... not currently used
#'
#' @examples
#' library(data.table)
#' library(FeatureHashing)
#' data(ipinyou)
#' f <- ~ 0 + BidID + IP + City + AdExchange + BiddingPrice + PayingPrice
#' m.train <- FTRLProx_Model_Matrix(f, ipinyou.train[, -"IsClick", with = FALSE],
#'                                  label = as.numeric(ipinyou.train$IsClick))
#'
#' m.train
#' print(m.train, verbose = TRUE)
#'
#' @method print ftrl.Dataset
#' @export
print.ftrl.Dataset <- function(x, verbose = FALSE, ...) {
  cat("ftrl.Dataset  dim:", nrow(x), "x", ncol(x), " label: ", ifelse(!is.null(x$y), "yes", "no"), "\n")
  if (verbose) {
    print(x$Mapping)
  }
}

#' @title Subsetting ftrl.Dataset
#'
#' @description
#' Returns subsets of a \code{ftrl.Dataset}.
#'
#' @param data an object of class "ftrl.Dataset"
#' @param i logical expression indicating elements or rows to keep.
#' @param ... other parameters (currently not used)
#' @return A object of \code{ftrl.Dataset} containing the subset of rows that are selected.
#' @rdname slice.ftrl.Dataset
#' @export
slice <- function(data, ...) UseMethod("slice")

#' @rdname slice.ftrl.Dataset
#' @export
slice.ftrl.Dataset <- function(data, i, ...) {
  # Check
  if (class(data) != "ftrl.Dataset") {
    stop("slice: first argument must be ftrl.Dataset")
  }
  # Slicing
  X_slice <- data$X[,i]
  if (!is.null(data$y)) {
    y_slice <- data$y[i]
  } else {
    y_slice <- NULL
  }
  # Generating FTRLProx Dataset
  data_slice <- list(X = X_slice, y = y_slice, Mapping = data$Mapping)
  class(data_slice) <- "ftrl.Dataset"
  # Return FTRLProx Dataset
  return(data_slice)
}
