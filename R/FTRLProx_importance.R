#' @title FTRL-Proximal Linear Model Feature Importance Function
#'
#' @description
#' FTRLProx_importance showes the most important features of FTRL-Proximal Linear Model.
#'
#' @param model a FTRL-Proximal linear model object.
#' @return A \code{data.table} of the features used in the model with their weight
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
#' FTRLProx_importance(ftrl_model)
#' @importFrom magrittr %>% %T>%
#' @importFrom data.table :=
#' @export

FTRLProx_importance <- function(model) {
  # Feature Importance Generation
  data.table::data.table(Index = seq_len(nrow(model$state$w)),
                         Weight = as.double(model$state$w)) %>%
    merge(., model$mapping, by = "Index") %>%
    magrittr::extract(., j = c("Feature", "Weight")) %>%
    magrittr::extract(., i = order(abs(Weight), decreasing = TRUE))
}
utils::globalVariables(c("Weight"))
