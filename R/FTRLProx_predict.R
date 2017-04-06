#' @title FTRL-Proximal Linear Model Predicting
#'
#' @description
#' An advanced interface for FTRL-Proximal online learning model predicting.
#'
#' @param model a FTRL-Proximal linear model object.
#' @param data a object of class \code{ftrl.Dataset}.
#' @return an vector of linear model predicted values
#' @examples
#' library(data.table)
#' library(FeatureHashing)
#' library(MLmetrics)
#' data(ipinyou)
#' m.train <- FTRLProx_Hashing(~ 0 + ., ipinyou.train[, -"IsClick", with = FALSE],
#'                             hash.size = 2^13, signed.hash = FALSE, verbose = TRUE,
#'                             label = as.numeric(ipinyou.train$IsClick))
#' m.test <- FTRLProx_Hashing(~ 0 + ., ipinyou.test[,-"IsClick", with = FALSE],
#'                            hash.size = 2^13, signed.hash = FALSE, verbose = TRUE,
#'                            label = as.numeric(ipinyou.test$IsClick))
#' ftrl_model <- FTRLProx_train(data = m.train, model = NULL,
#'                              family = "binomial",
#'                              params = list(alpha = 0.01, beta = 0.1,
#'                                            l1 = 0.1, l2 = 0.1, dropout = 0), epoch = 50,
#'                              watchlist = list(test = m.test),
#'                              eval = AUC, patience = 5, maximize = TRUE,
#'                              nthread = 1, verbose = TRUE)
#' pred_ftrl <- FTRLProx_predict(ftrl_model, m.test)
#' AUC(pred_ftrl, as.numeric(ipinyou.test$IsClick))
#' @importFrom magrittr %>%
#' @export

FTRLProx_predict <- function(model, data) {
  # Prediction
  predict_spMatrix(data$X, model$state$w, model$family) %>%
    as.double(.)
}
utils::globalVariables(c("."))
