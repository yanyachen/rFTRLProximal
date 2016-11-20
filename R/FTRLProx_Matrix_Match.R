#' @title Transform Sparse Design Matrix to Assigned Scheme
#'
#' @description
#' Transform a sparse "design" matrix to assigned scheme for FTRL-Proximal Algorithm.
#'
#' @param x a \code{data.frame} or \code{data.table}. The original data.
#' @param default_name the feature name of design matrix
#' @return an object of class "dgCMatrix"
#' @examples
#' library(data.table)
#' library(FeatureHashing)
#' data(ipinyou)
#' f <- ~ 0 + BidID + IP + City + AdExchange + BiddingPrice + PayingPrice
#' m.train <- FTRLProx_Model_Matrix(f, ipinyou.train[, -"IsClick", with = FALSE])
#' m.train <- FTRLProx_Matrix_Match(m.train,
#'                                  c("AdExchange2", "AdExchange3", "BiddingPrice", "PayingPrice"))
#' hash.mapping(m.train)
#' @export

FTRLProx_Matrix_Match <- function(x, default_name) {
  # Sparse Design Matrix
  p_add <- setdiff(default_name, rownames(x))
  if (length(p_add) > 0) {
    x <- rbind(x,
               Matrix::Matrix(0, nrow = length(p_add), ncol = ncol(x),
                              dimnames = list(p_add, NULL),
                              sparse = TRUE))
  }
  x <- x[match(default_name, rownames(x)), , drop = FALSE]
  # Create Mapping
  Mapping <- (seq_len(nrow(x)) - 1) %>%
    as.list(.) %>%
    magrittr::set_names(., value = rownames(x)) %>%
    as.environment(.)
  attr(x, "mapping") <- Mapping
  # Return Sparse Design Matrix
  return(x)
}
