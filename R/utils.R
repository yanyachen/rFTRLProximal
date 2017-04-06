#' Create Feature Mapping DT
#' This is used for creating feature mapping from features hashing
#'
#' @importFrom magrittr %>%
#' @keywords internal
Mapping_DT_Gen <- function(Mapping_Vec) {
  data.table::data.table(Index = Mapping_Vec,
                         Feature = names(Mapping_Vec)) %>%
    magrittr::extract(., j = .(Feature = paste(get("Feature"), collapse = "+")), by = "Index") %>%
    magrittr::extract(., i = order(Index, decreasing = FALSE))
}
utils::globalVariables(c("Index"))

#' Update Feature Mapping DT
#' This is used for updating feature mapping from features hashing
#'
#' @importFrom magrittr %>% %T>%
#' @keywords internal
Mapping_DT_Update <- function(Mapping_Model, Mapping_Data) {
  merge(Mapping_Model, Mapping_Data, by = "Index", all = TRUE) %T>%
    magrittr::extract(., i = is.na(Feature.x), j = Feature := Feature.y) %T>%
    magrittr::extract(., i = is.na(Feature.y), j = Feature := Feature.x) %T>%
    magrittr::extract(., i = !is.na(Feature.x) & !is.na(Feature.y), j = Feature := paste(Feature.x, Feature.y, sep = "+")) %>%
    magrittr::extract(., j = c("Index", "Feature"), with = FALSE)
}
utils::globalVariables(c("Feature.x", "Feature.y", "Feature"))

#' Performance Printing Function
#' This is used for printing model performance of each round
#'
#' @importFrom magrittr %>%
#' @keywords internal
Perf_Print <- function(Round, Name, Value) {
  Round_Part <- paste("[", Round, "]", "\t", sep = "", collapse = "")
  Perf_Part <- paste(Name, ":", format(Value, digits = 6), " \t ", sep = "", collapse = "")
  paste(Round_Part, Perf_Part, "\n", sep = "", collapse = "") %>% cat(.)
}
