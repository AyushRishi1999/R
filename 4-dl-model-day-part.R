if (!require('tidyverse')) install.packages('tidyverse', quiet = TRUE); library('tidyverse')
if (!require('skimr')) install.packages('skimr', quiet = TRUE); library('skimr')
if (!require('h2o')) install.packages('h2o', quiet = TRUE); library('h2o')
if (!require('DALEXtra')) install.packages('DALEXtra', quiet = TRUE); library('DALEXtra')

processed_youtube_df <- readRDS("processed_youtube_ml.rds")

h2o.init(min_mem_size = "2G",nthreads=-1)
data_h2o_no_destination <- as.h2o(processed_youtube_df)

# Partition the data into training, validation and test sets
splits <- h2o.splitFrame(data = data_h2o_no_destination, seed = 1234, ratios = c(0.8)) # 80/20 split
train_h2o <- splits[[1]] # from training data
valid_h2o <- splits[[2]] # from training data
rm(splits)

y <- "Utc_Day_Part" # column name for outcome
x <- setdiff(names(train_h2o), y) # column names for predictors

hyper_params <- list(
  activation = c("Rectifier", "Tanh", "Maxout", "RectifierWithDropout",
                 "TanhWithDropout", "MaxoutWithDropout"),
  hidden = list(c(50,50), c(32,32,32), c(16,16,16), c(25,25,25,25)),
  input_dropout_ratio = c(0, 0.05),
  l1 = seq(from=0, to=1e-4, by=1e-6),
  l2 = seq(from=0, to=1e-4, by=1e-6)
)

search_criteria = list(
  strategy = "RandomDiscrete",
  seed=1234567,
  stopping_metric = "auto", # logloss for classification, deviance for regression
  stopping_rounds=5, # stop when the last 5 models
  stopping_tolerance=0.01, # improve less than 1%
  max_runtime_secs = 600, # stop when the search took more than 360 seconds
  max_models = 100 # stop when the search tried over 100 models
)

grid <- h2o.grid(
  algorithm = "deeplearning",
  grid_id = "dl_grid_random",
  x = x,
  y = y,
  training_frame = train_h2o,
  validation_frame = valid_h2o,
  epochs = 150,
  score_validation_samples = 10000,
  score_duty_cycle = 0.025,
  max_w2 = 10, #can help improve stability for Rectifier
  hyper_params = hyper_params,
  search_criteria = search_criteria
)
rm(hyper_params)
rm(search_criteria)

summary(grid)
best_dl <- h2o.getModel(grid@model_ids[[1]])
plot(best_dl)
summary(best_dl)

# Get the grid results, sorted by validation AUC
grid_sort_auc <- h2o.getGrid(grid_id = "dl_grid_random",
                             sort_by = "accuracy",
                             decreasing = TRUE)

#grid_id <- grid@grid_id
#grid1_model_count <- length(grid@model_ids)
#h2o.saveGrid(grid_id = grid_id, grid_directory = getwd())


#To Train at the specific model and save the H2O File
m1 <- h2o.deeplearning(
  model_id = "model-dl.h2o",
  x = x,
  y = y,
  training_frame = train_h2o,
  validation_frame = valid_h2o,
  overwrite_with_best_model = F,
  hidden = c(128,128,128),
  epochs = 30
)

h2o.saveModel(object = best_dl, # the model you want to save
              path = getwd(), # the folder to save
              force = TRUE,
              filename = "model-dl.h2o") # whether to overwrite an existing file
model_filepath = str_c(getwd(), "/dl_grid_random.h2o") #dl_model_first is model_id
model_dl_grid <- h2o.loadGrid(model_filepath) # load a model from file

h2o.shutdown(prompt=FALSE)
