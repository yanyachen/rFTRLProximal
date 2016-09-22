#' @title FTRL-Proximal Linear Model Feature Importance Function
#'
#' @description
#' FTRLProx.importance showes the most important features of FTRL-Proximal Linear Model.
#'
#' @param model a FTRL-Proximal linear model object.
#' @return A \code{data.table} of the features used in the model with their weight
#' @examples
#' library(data.table)
#' library(FeatureHashing)
#' data(ipinyou)
#' m.train <- FTRLProx_Hashing(~ 0 + ., ipinyou.train[, -"IsClick", with = FALSE],
#'                             hash.size = 2^10, signed.hash = FALSE, verbose = TRUE)
#' ftrl_model <- FTRLProx_train(m.train, y = as.numeric(ipinyou.train$IsClick),
#'                              family = "binomial",
#'                              params = list(alpha = 0.01, beta = 0.1, l1 = 1.0, l2 = 1.0),
#'                              epoch = 5, verbose = TRUE)
#' FTRLProx.importance(ftrl_model)
#' @importFrom stats na.omit
#' @importFrom magrittr %>% %T>%
#' @importFrom data.table :=
#' @export

FTRLProx.importance <- function(model) {
  # Solve Hashing Collision
  Mapping_DT_Gen <- function(Mapping) {
    Mapping_DT <- data.table::data.table(Index = unique(Mapping),
                                         Feature_Name = character(1))
    Mapping_Name <- names(Mapping)
    for (i in seq_len(nrow(Mapping_DT))) {
      data.table::set(Mapping_DT, i, "Feature_Name",
                      value = paste(Mapping_Name[Mapping == Mapping_DT[i, Index]], collapse = " + "))
    }
    return(Mapping_DT)
  }
  Mapping_DT <- Mapping_DT_Gen(model$mapping)
  # Feature Importance Generation
  data.table::data.table(ID = seq_along(model$weight),
                         Weight = model$weight) %T>%
    magrittr::extract(., j = "Feature" := match(get("ID"), Mapping_DT$Index) %>%
                        magrittr::extract(Mapping_DT$Feature_Name, .), with = FALSE) %T>%
    magrittr::extract(., j = "Abs_Weight" := abs(get("Weight"))) %T>%
    data.table::setorderv(., cols = c("Abs_Weight", "Feature"), order = c(-1, +1)) %>%
    magrittr::extract(., j = c("Feature", "Weight"), with = FALSE) %>%
    na.omit(.)
}
utils::globalVariables(c("Index"))
