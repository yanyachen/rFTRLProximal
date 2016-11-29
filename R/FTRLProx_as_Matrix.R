#' @title Transform Data Frame to Sparse Design Matrix for FTRL-Proximal Model
#'
#' @description
#' Transform a data frame to sparse "design" matrix for FTRL-Proximal Algorithm.
#' Please always use this function to generate sparse matrix for training and prediction.
#'
#' @param data a \code{data.frame} or \code{data.table}. The original data.
#' @return an object of class "dgCMatrix"
#' @examples
#' library(data.table)
#' library(FeatureHashing)
#' data(ipinyou)
#' m.train <- FTRLProx_as_Matrix(ipinyou.train[, c("BiddingPrice", "PayingPrice"), with = FALSE])
#' hash.mapping(m.train)
#' @export

FTRLProx_as_Matrix <- function(data) {
  # Sparse Design Matrix
  x <- data %>%
    as.matrix(.) %>%
    Matrix::Matrix(., sparse = TRUE) %>%
    Matrix::t(.)
  # Create Mapping
  Mapping <- (seq_len(nrow(x)) - 1) %>%
    as.list(.) %>%
    magrittr::set_names(., value = rownames(x)) %>%
    as.environment(.)
  attr(x, "mapping") <- Mapping
  # Return Sparse Design Matrix
  return(x)
}
