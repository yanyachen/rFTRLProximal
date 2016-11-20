#' @title Construct Sparse Design Matrix for FTRL-Proximal Model
#'
#' @description
#' Construct a sparse model or "design" matrix, form a formula and data frame for FTRL-Proximal Algorithm.
#' A wrapper of \code{sparse.model.matrix} function in the Matrix package.
#' Please always use this function to generate sparse matrix for training and prediction.
#'
#' @param formula a model formula.
#' @param data a \code{data.frame} or \code{data.table}. The original data.
#' @param ... further arguments passed to \code{sparse.model.matrix}.
#' @return an object of class "dgCMatrix"
#' @examples
#' library(data.table)
#' library(FeatureHashing)
#' data(ipinyou)
#' f <- ~ 0 + BidID + IP + City + AdExchange + BiddingPrice + PayingPrice
#' m.train <- FTRLProx_Model_Matrix(f, ipinyou.train[, -"IsClick", with = FALSE])
#' hash.mapping(m.train)
#' @export

FTRLProx_Model_Matrix <- function(formula = ~ ., data, ...) {
  # Sparse Design Matrix
  x <- Matrix::sparse.model.matrix(object = formula, data = data, transpose = TRUE, ...)
  # Create Mapping
  Mapping <- (seq_len(nrow(x)) - 1) %>%
    as.list(.) %>%
    magrittr::set_names(., value = rownames(x)) %>%
    as.environment(.)
  attr(x, "mapping") <- Mapping
  # Return Sparse Design Matrix
  return(x)
}
