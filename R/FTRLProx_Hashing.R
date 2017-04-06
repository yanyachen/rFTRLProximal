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
#' @param label a vector containing labels.
#' @return constructed dataset, an object of class "ftrl.Dataset"
#' @examples
#' library(data.table)
#' library(FeatureHashing)
#' data(ipinyou)
#' m.train <- FTRLProx_Hashing(~ 0 + ., ipinyou.train[, -"IsClick", with = FALSE],
#'                             hash.size = 2^13, signed.hash = FALSE, verbose = TRUE,
#'                             label = as.numeric(ipinyou.train$IsClick))
#' m.test <- FTRLProx_Hashing(~ 0 + ., ipinyou.test[,-"IsClick", with = FALSE],
#'                            hash.size = 2^13, signed.hash = FALSE, verbose = TRUE,
#'                            label = as.numeric(ipinyou.test$IsClick))
#' @importFrom magrittr %>%
#' @export

FTRLProx_Hashing <- function(formula = ~ ., data, hash.size, signed.hash, verbose = TRUE,
                             label = NULL) {
  # Check
  if (!is.null(label)) {
    stopifnot(nrow(data) == length(label))
  }
  # Feature Hashing
  X <- FeatureHashing::hashed.model.matrix(formula = formula,
                                           data = data, hash.size = hash.size, signed.hash = signed.hash,
                                           transpose = TRUE, create.mapping = TRUE, is.dgCMatrix = TRUE,
                                           progress = verbose)
  Mapping <- FeatureHashing::hash.mapping(X) %>%
    Mapping_DT_Gen(.)
  # Generating FTRLProx Dataset
  FTRLProx_Dataset <- list(X = X, y = label, Mapping = Mapping)
  class(FTRLProx_Dataset) <- "ftrl.Dataset"
  # Return FTRLProx Dataset
  return(FTRLProx_Dataset)
}
