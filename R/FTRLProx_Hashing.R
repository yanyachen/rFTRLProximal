#' @title Feature Hashing for FTRL-Proximal Model
#'
#' @description
#' Feature Hashing for FTRL-Proximal Algorithm. A wrapper of \code{hashed.model.matrix} function in the FeatureHashing package.
#' Please always use this function to generate sparse matrix for training and prediction.
#'
#' @param formula formula or a character vector of column names (will be expanded to a formula)
#' @param data a \code{data.frame} or \code{data.table}. The original data.
#' @param hash.size positive integer. The hash size of feature hashing.
#' @param signed.hash logical value. Indicating if the hashed value is multipled by random sign.
#'   This will reduce the impact of collision. Disable it will enhance the speed.
#' @param verbose logical value. Indicating if the progress bar is displayed or not.
#' @return an object of class "dgCMatrix"
#' @examples
#' library(data.table)
#' library(FeatureHashing)
#' data(ipinyou)
#' m.train <- FTRLProx_Hashing(~ 0 + ., ipinyou.train[, -"IsClick", with = FALSE],
#'                             hash.size = 2^13, signed.hash = FALSE, verbose = TRUE)
#' m.test <- FTRLProx_Hashing(~ 0 + ., ipinyou.test[,-"IsClick", with = FALSE],
#'                            hash.size = 2^13, signed.hash = FALSE, verbose = TRUE)
#' @export

FTRLProx_Hashing <- function(formula = ~ ., data, hash.size, signed.hash, verbose = TRUE) {
  # Feature Hashing
  if (verbose == TRUE) {
    message("Feature Hashing:")
  }
  x <- FeatureHashing::hashed.model.matrix(formula = formula,
                                           data = data, hash.size = hash.size, signed.hash = signed.hash,
                                           transpose = TRUE, create.mapping = TRUE, is.dgCMatrix = TRUE,
                                           progress = verbose)
  # Return Design Sparse Matrix
  return(x)
}
