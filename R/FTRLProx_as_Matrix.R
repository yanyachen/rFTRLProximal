#' @title Transform Data Frame to Sparse Design Matrix for FTRL-Proximal Model
#'
#' @description
#' Transform a data frame to sparse design matrix for FTRL-Proximal Algorithm.
#' Please always use this function to generate sparse matrix for training and prediction.
#'
#' @param data a \code{data.frame} or \code{data.table}. The original data.
#' @param label a vector containing labels.
#' @return constructed dataset, an object of class "ftrl.Dataset"
#' @examples
#' library(data.table)
#' library(FeatureHashing)
#' data(ipinyou)
#' m.train <- FTRLProx_as_Matrix(ipinyou.train[, c("BiddingPrice", "PayingPrice"), with = FALSE],
#'                               label = as.numeric(ipinyou.train$IsClick))
#' @export

FTRLProx_as_Matrix <- function(data, label = NULL) {
  # Sparse Design Matrix
  X <- data %>%
    as.matrix(.) %>%
    Matrix::Matrix(., sparse = TRUE) %>%
    Matrix::t(.)
  # Create Mapping
  Mapping <- data.table::data.table(Index = seq_len(ncol(data)),
                                    Feature = names(data))
  # Generating FTRLProx Dataset
  FTRLProx_Dataset <- list(X = X, y = label, Mapping = Mapping)
  class(FTRLProx_Dataset) <- "ftrl.Dataset"
  # Return FTRLProx Dataset
  return(FTRLProx_Dataset)
}
