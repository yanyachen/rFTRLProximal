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
