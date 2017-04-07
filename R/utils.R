#' Create Feature Mapping DT
#' This is used for creating feature mapping from features hashing
#'
#' @importFrom magrittr %>% %T>%
#' @keywords internal
Mapping_DT_Gen <- function(Mapping_Vec) {
  data.table::data.table(Index = Mapping_Vec,
                         Feature = names(Mapping_Vec)) %>%
    unique(., by = c("Index", "Feature")) %T>%
    data.table::setorderv(., cols = c("Index", "Feature"), order = c(+1L, +1L)) %>%
    magrittr::extract(., j = .(Feature = paste(get("Feature"), collapse = "+")), by = "Index") %>%
    magrittr::extract(., i = order(Index, decreasing = FALSE))
}
utils::globalVariables(c("Index"))

#' Decompose Feature Mapping DT
#' This is used for decomposing feature mapping from features hashing
#'
#' @importFrom magrittr %>%
#' @keywords internal
Mapping_DT_Decompose <- function(Mapping) {
  Mapping_Feature_Index <- Mapping$Index
  Mapping_Feature_List <- strsplit(Mapping$Feature, split = "+", fixed = TRUE)
  rep(Mapping_Feature_Index,
      vapply(Mapping_Feature_List, length, integer(1))) %>%
    magrittr::set_names(., unlist(Mapping_Feature_List))
}

#' Update Feature Mapping DT
#' This is used for updating feature mapping from features hashing
#'
#' @importFrom magrittr %>%
#' @keywords internal
Mapping_DT_Update <- function(Mapping_Model, Mapping_Data) {
  Mapping_Vec <- c(Mapping_DT_Decompose(Mapping_Model),
                   Mapping_DT_Decompose(Mapping_Data)) %>%
    Mapping_DT_Gen(.)
}

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
