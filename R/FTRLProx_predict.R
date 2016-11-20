#' @title FTRL-Proximal Linear Model Predicting
#'
#' @description
#' An advanced interface for FTRL-Proximal online learning model predicting.
#'
#' @param model a FTRL-Proximal linear model object.
#' @param newx a transposed \code{dgCMatrix}.
#' @return an vector of linear model predicted values
#' @examples
#' library(data.table)
#' library(FeatureHashing)
#' library(MLmetrics)
#' data(ipinyou)
#' m.train <- FTRLProx_Hashing(~ 0 + ., ipinyou.train[, -"IsClick", with = FALSE],
#'                             hash.size = 2^13, signed.hash = FALSE, verbose = TRUE)
#' m.test <- FTRLProx_Hashing(~ 0 + ., ipinyou.test[,-"IsClick", with = FALSE],
#'                            hash.size = 2^13, signed.hash = FALSE, verbose = TRUE)
#' ftrl_model <- FTRLProx_train(m.train, y = as.numeric(ipinyou.train$IsClick), family = "binomial",
#'                              params = list(alpha = 0.01, beta = 0.1, l1 = 1.0, l2 = 1.0),
#'                              epoch = 10, verbose = TRUE)
#' pred_ftrl <- FTRLProx_predict(ftrl_model, newx = m.test)
#' AUC(pred_ftrl, as.numeric(ipinyou.test$IsClick))
#' @importFrom magrittr %>%
#' @importFrom foreach %do%
#' @export

FTRLProx_predict <- function(model, newx) {
  # Prediction
  if (is.null(model$weight_mat) == TRUE) {
    FTRLProx_predict_spMatrix(newx, model$weight, model$family) %>%
      as.numeric(.)
  } else {
    foreach::foreach(i = seq_len(ncol(model$weight_mat)), .combine = "cbind") %do% {
      FTRLProx_predict_spMatrix(newx, model$weight_mat[,i], model$family) %>%
        as.numeric(.)
    } %>%
      Matrix::rowMeans(as.matrix(.))
  }
}
utils::globalVariables(c("."))
