#' @title Transform Sparse Design Matrix to Assigned Schema
#'
#' @description
#' Transform a sparse "design" matrix to assigned schema for FTRL-Proximal Algorithm.
#'
#' @param data a object of \code{ftrl.Dataset}.
#' @param feature_name the feature name of design matrix
#' @return constructed dataset, an object of class "ftrl.Dataset"
#' @examples
#' library(data.table)
#' library(FeatureHashing)
#' data(ipinyou)
#' f <- ~ 0 + BidID + IP + City + AdExchange + BiddingPrice + PayingPrice
#' m.train <- FTRLProx_Model_Matrix(f, ipinyou.train[, -"IsClick", with = FALSE])
#' m.test <- FTRLProx_Model_Matrix(f, ipinyou.test[, -"IsClick", with = FALSE])
#' identical(rownames(m.train$X), rownames(m.test$X))
#' m.test <- FTRLProx_Matrix_Match(m.test, rownames(m.train$X))
#' identical(rownames(m.train$X), rownames(m.test$X))
#' m.test$Mapping
#' @export

FTRLProx_Matrix_Match <- function(data, feature_name) {
  # Sparse Design Matrix
  p_add <- setdiff(feature_name, rownames(data$X))
  if (length(p_add) > 0) {
    X <- rbind(data$X,
               Matrix::Matrix(0, nrow = length(p_add), ncol = ncol(data$X),
                              dimnames = list(p_add, NULL),
                              sparse = TRUE))
  }
  X <- X[match(feature_name, rownames(X)), , drop = FALSE]
  # Create Mapping
  Mapping <- data.table::data.table(Index = seq_len(nrow(X)),
                                    Feature = rownames(X))
  # Generating FTRLProx Dataset
  FTRLProx_Dataset <- list(X = X, y = data$y, Mapping = Mapping)
  class(FTRLProx_Dataset) <- "ftrl.Dataset"
  # Return FTRLProx Dataset
  return(FTRLProx_Dataset)
}
