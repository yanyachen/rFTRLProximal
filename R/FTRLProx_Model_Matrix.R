#' @title Construct Sparse Design Matrix for FTRL-Proximal Model
#'
#' @description
#' Construct a sparse model or "design" matrix, form a formula and data frame for FTRL-Proximal Algorithm.
#' A wrapper of \code{sparse.model.matrix} function in the Matrix package.
#' Please always use this function to generate sparse matrix for training and prediction.
#'
#' @param formula a model formula.
#' @param data a \code{data.frame} or \code{data.table}. The original data.
#' @param label a vector containing labels.
#' @param ... further arguments passed to \code{sparse.model.matrix}.
#' @return constructed dataset, an object of class "ftrl.Dataset"
#' @examples
#' library(data.table)
#' library(FeatureHashing)
#' data(ipinyou)
#' f <- ~ 0 + BidID + IP + City + AdExchange + BiddingPrice + PayingPrice
#' m.train <- FTRLProx_Model_Matrix(f, ipinyou.train[, -"IsClick", with = FALSE],
#'                                  label = as.numeric(ipinyou.train$IsClick))
#' @export

FTRLProx_Model_Matrix <- function(formula = ~ ., data, label = NULL, ...) {
  # Sparse Design Matrix
  X <- Matrix::sparse.model.matrix(object = formula, data = data, transpose = TRUE, row.names = TRUE, ...)
  # Create Mapping
  Mapping <- data.table::data.table(Index = seq_len(nrow(X)),
                                    Feature = rownames(X))
  # Generating FTRLProx Dataset
  FTRLProx_Dataset <- list(X = X, y = label, Mapping = Mapping)
  class(FTRLProx_Dataset) <- "ftrl.Dataset"
  # Return FTRLProx Dataset
  return(FTRLProx_Dataset)
}
