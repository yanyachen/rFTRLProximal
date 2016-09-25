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
#' data(ipinyou)
#' m.train <- FTRLProx_Hashing(~ 0 + ., ipinyou.train[, -"IsClick", with = FALSE],
#'                             hash.size = 2^13, signed.hash = FALSE, verbose = TRUE)
#' ftrl_model <- FTRLProx_train(m.train, y = as.numeric(ipinyou.train$IsClick),
#'                              family = "binomial",
#'                              params = list(alpha = 0.01, beta = 0.1, l1 = 1.0, l2 = 1.0),
#'                              epoch = 10, verbose = TRUE)
#' FTRLProx_importance(ftrl_model)
#' @importFrom stats na.omit
#' @importFrom magrittr %>% %T>%
#' @importFrom data.table :=
#' @export

FTRLProx_importance <- function(model) {
  # Solve Hashing Collision
  Mapping_DT_Gen <- function(Mapping) {
    data.table::data.table(Index = Mapping,
                           Feature = names(Mapping)) %>%
      magrittr::extract(., j = .(Feature = paste(get("Feature"), collapse = "+")), by = "Index")
  }
  Mapping_DT <- Mapping_DT_Gen(model$mapping)
  # Feature Importance Generation
  data.table::data.table(Index = seq_along(model$weight),
                         Weight = as.numeric(model$weight)) %T>%
    magrittr::extract(., j = "Feature" := match(get("Index"), Mapping_DT$Index) %>%
                        magrittr::extract(Mapping_DT$Feature, .), with = FALSE) %T>%
    magrittr::extract(., j = "Abs_Weight" := abs(get("Weight"))) %T>%
    data.table::setorderv(., cols = c("Abs_Weight", "Feature"), order = c(-1, +1)) %>%
    magrittr::extract(., j = c("Feature", "Weight"), with = FALSE) %>%
    na.omit(.)
}
utils::globalVariables(c("Index"))
