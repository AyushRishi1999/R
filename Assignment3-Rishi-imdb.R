library(skimr)
library(recipes)
library(stringr)
library(tidyverse)
library(h2o)
library(dplyr)
library(caret)
library(skimr)
library(recipes)

df <- read_csv("data/imdb_top_2000_movies.csv",na = c("",NA,"-1"))

# Getting Structured view of data-set columns
summary(df)
glimpse(df)

# Check for NA in the entire data-set
any_na <- any(is.na(df))
any_na

#Renaming Column names to replace " " with "_"
names(df) <- gsub(" ", "_", names(df))

# Remove non-numeric characters from the Release_Year column
df$Release_Year <- str_replace_all(df$Release_Year, "\\D", "")

# Extract the last four digits and set is as double
df$Release_Year <- substr(df$Release_Year, nchar(df$Release_Year) - 3, nchar(df$Release_Year))
df$Release_Year <- as.double(df$Release_Year)

# Remove the "$" and "M" from the Gross column and convert to double
df$Gross <- as.double(gsub("\\$|M", "", df$Gross))

# Segregating Data into x and y train
x_train_tbl <- df |> select(-"IMDB_Rating")
y_train_tbl <- df |> select("IMDB_Rating")

# Checking Datatypes for x and y train
x_train_tbl_skim = partition(skim(x_train_tbl))
names(x_train_tbl_skim)
y_train_tbl_skim = partition(skim(y_train_tbl))
names(y_train_tbl_skim)

yield_data_skim = partition(skim(x_train_tbl))
names(x_train_tbl)
yield_data_skim$character
yield_data_skim$numeric |> tibble()

# Converting Character data to Factor for x train
string_2_factor_names <- yield_data_skim$character$skim_variable
string_2_factor_names

rec_obj_x <- recipe(~ ., data = x_train_tbl) |>
  step_string2factor(all_of(string_2_factor_names)) |>
  step_impute_median(all_numeric()) |> # missing values in numeric columns
  step_impute_mode(all_nominal()) |> # missing values in factor columns
  prep()

x_train_processed_tbl <- bake(rec_obj_x, x_train_tbl)
x_train_processed_tbl

rec_obj <- recipe(~ ., data = y_train_tbl) %>%
  step_dummy(all_nominal(), -all_outcomes())
rec_obj <- prep(rec_obj)
y_train_processed_tbl <- bake(rec_obj, new_data = y_train_tbl)
y_train_processed_tbl

# Writing processed data set into csv format
df_processed <- cbind(x_train_processed_tbl, y_train_processed_tbl)
write.csv(df_processed, "Assignment3-Rishi-imdb.csv", row.names = FALSE)

# Initiate h2o and get cluster information
h2o.init(nthreads = -1)
h2o.clusterInfo()

data_h2o <- as.h2o(bind_cols(y_train_tbl, x_train_processed_tbl))

splits <- h2o.splitFrame(data = data_h2o, seed = 1234,
                         ratios = c(0.6 , 0.2)) 
train_h2o <- splits[[1]]
valid_h2o <- splits[[2]]

# Training the model
y <- "IMDB_Rating"
x <- setdiff(names(train_h2o), y)
m1 <- h2o.deeplearning(
  model_id = "Assignment3-Rishi-imdb.h2o",
  x = x,
  y = y,
  training_frame = train_h2o,
  validation_frame = valid_h2o,
  overwrite_with_best_model = F,
  hidden = c(128,128,128),
  epochs = 30
)

# Saving the h2o model
h2o.saveModel(object = m1,
              path = getwd(),
              force = TRUE)

h2o.shutdown(prompt = F)

