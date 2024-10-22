#Data Wrangling

install.packages(c("skimr","recipes"))
library(skimr)
library(recipes)
library(stringr)
library(tidyverse)

application_train <- read_csv("data/loan/application_train.csv",na = c("",NA,"-1"))
application_test <- read_csv("data/loan/application_test.csv",na = c("",NA,"-1"))
View(application_train)
View(application_test)

# Training data: Separate into x and y tibbles
x_train_tbl <- application_train |> select(-TARGET)
y_train_tbl <- application_train |> select(TARGET)
# Testing data
x_test_tbl <- application_test
# Remove the original data to save memory
rm(application_train)
rm(application_test)

x_train_tbl_skim = partition(skim(x_train_tbl))
names(x_train_tbl_skim)

x_train_tbl_skim$character
x_train_tbl_skim$numeric |> tibble()

string_2_factor_names <- x_train_tbl_skim$character$skim_variable
string_2_factor_names

rec_obj <- recipe(~ ., data = x_train_tbl) |>
  step_string2factor(all_of(string_2_factor_names)) |>
  step_impute_median(all_numeric()) |>
  # missing values in numeric columns
  step_impute_mode(all_nominal()) |>
  # missing values in factor columns
  prep()
rec_obj

x_train_processed_tbl <- bake(rec_obj, x_train_tbl)
x_test_processed_tbl <- bake(rec_obj, x_test_tbl)

x_train_tbl |> #original data before baking
  select(1:30)|>
  glimpse()
x_train_processed_tbl |> #processed data after baking
  select(1:30)|>
  glimpse()

rec_obj_for_y <- recipe(~., data = y_train_tbl) |>
  step_num2factor("TARGET", levels = c("0","1"), transform = function(x) x+1)|>
  prep(stringsAsFactors = FALSE)
y_train_processed_tbl <- bake(rec_obj_for_y, y_train_tbl)

write_rds(x_train_processed_tbl,"x_train_processed_tbl.rds")
write_rds(x_test_processed_tbl,"x_test_processed_tbl.rds")
write_rds(y_train_processed_tbl,"y_train_processed_tbl.rds")

#H2O

install.packages("h2o")
library(h2o)

h2o.init(nthreads = -1) #-1 to use all cores

# push data into h2o;NOTE:THIS MAY TAKE A WHILE!
data_h2o <- as.h2o(
  bind_cols(y_train_processed_tbl, x_train_processed_tbl),
  destination_frame="train.hex" #destination_frame is optional
  )
new_data_h2o <- as.h2o(
  x_test_processed_tbl,
  destination_frame="test.hex" #destination_frame is optional
)
# what if you do not assign destination_frame
data_h2o_no_destination <- as.h2o(
  bind_cols(y_train_processed_tbl, x_train_processed_tbl)
)

h2o.ls()
h2o_keys = as.character(h2o.ls()$key)
h2o.rm(h2o_keys[str_detect(h2o_keys,"^data")])
h2o.ls()

# Partition the data into training, validation and test sets
splits <- h2o.splitFrame(data = data_h2o, seed =1234,
                         ratios = c(0.7,0.15)) # 70/15/15 split
train_h2o <- splits[[1]] # from training data
valid_h2o <- splits[[2]] # from training data
test_h2o <- splits[[3]] # from training data

#Deep Learning

y <- "TARGET" #column name for outcome
x <- setdiff(names(train_h2o),y) #column names for predictors

m1 <- h2o.deeplearning(
  model_id = "dl_model_first",
  x=x,
  y=y,
  training_frame = train_h2o,
  validation_frame = valid_h2o, #validation dataset: used for scoring and 
                                #early stopping
  #activation="Rectifier", ## default
  #hidden=c(200,200), ## default: 2 hidden layers, 200 neurons each
  epochs = 1
)

summary(m1)

h2o.saveModel(object = m1, # the model you want to save
              path = getwd(), # the folder to save
              force =TRUE) # whether to overwrite an existing file
model_filepath = str_c(getwd(), "/dl_model_first")#dl_model_first is model_id
m1 <- h2o.loadModel(model_filepath)# load a model from file

#Config algorithm/modeling parameters

m2 <- h2o.deeplearning(
  model_id = "dl_model_faster",
  x=x,
  y=y,
  training_frame = train_h2o,
  validation_frame = valid_h2o, 
  hidden=c(32,32,32),                   ## small network, runs faster
  epochs = 1000000,                     ## hopefully converges earlier...
  score_validation_samples = 10000,     ## sample the validation dataset (faster)
  stopping_metric = "misclassification",## could also be "MSE","logloss","r2"
  stopping_rounds = 2,                  ## for 2 consecutive scoring events
  stopping_tolerance = 0.01             ## stop if the improvement is less than 1%
)
summary(m2)

m3 <- h2o.deeplearning(
  model_id="dl_model_tuned",
  x = x,
  y = y,
  training_frame = train_h2o,
  validation_frame = valid_h2o,
  overwrite_with_best_model =F,## Return the final model after 10 epochs,
                              ## even if not the best
  hidden = c(128,128,128), ## more hidden layers -> more complex interactions
  epochs =10,## to keep it short enough
  score_validation_samples =10000,## downsample validation set for faster scoring
  score_duty_cycle =0.025,        ## don't score more than 2.5% of the wall time
  adaptive_rate =F,## manually tuned learning rate
  rate =0.01,
  rate_annealing =2e-6,
  momentum_start =0.2,## manually tuned momentum
  momentum_stable =0.4,
  momentum_ramp =1e7,
  l1 =1e-5,## add some L1/L2 regularization
  l2 =1e-5,
  max_w2 =10 ## helps stability for Rectifier
)
summary(m3)

#Hyper-parameter tuning w/ grid search

hyper_params <- list(
  hidden = list(c(32,32,32), c(64,64)),
  input_dropout_ratio = c(0,0.05),
  rate = c(0.01,0.02),
  rate_annealing = c(1e-8,1e-7,1e-6)
)

grid <- h2o.grid(
  algorithm="deeplearning",
  grid_id="dl_grid",
  x = x,
  y = y,
  training_frame = train_h2o,
  validation_frame = valid_h2o,
  epochs =10,
  stopping_metric ="misclassification",
  stopping_tolerance =1e-2,## stop when misclassification does not
                           ## improve by >=1% for 2 scoring events
  stopping_rounds =2,
  score_validation_samples =10000,## downsample validation set for faster scoring
  score_duty_cycle =0.025,## don't score more than 2.5% of the wall time
  adaptive_rate =F,#manually tuned learning rate
  momentum_start =0.5,#manually tuned momentum
  momentum_stable =0.9,
  momentum_ramp =1e7,
  l1 =1e-5,
  l2 =1e-5,
  activation = c("Rectifier"),
  max_w2 =10,#can help improve stability for Rectifier
  hyper_params = hyper_params
)
grid <- h2o.getGrid("dl_grid", sort_by="logloss", decreasing=FALSE)
dl_grid_summary_table <- grid@summary_table
dl_grid_summary_table

dl_grid_best_model <- h2o.getModel(dl_grid_summary_table$model_ids[1])
summary(dl_grid_best_model)

dl_grid_best_model_params <- dl_grid_best_model@allparameters
dl_grid_best_model_params # too long to show on one slide

#Random Hyper-Parameter Search

hyper_params <- list(
  hidden = list( c(32,32,32), c(64,64)),
  input_dropout_ratio = c(0,0.05),
  rate = c(0.01,0.02),
  rate_annealing = c(1e-8,1e-7,1e-6)
)

hyper_params2 <- list(
  activation = c("Rectifier","Tanh","Maxout","RectifierWithDropout",
                 "TanhWithDropout","MaxoutWithDropout"),
  hidden = list(c(20,20), c(50,50), c(30,30,30), c(25,25,25,25)),
  input_dropout_ratio = c(0,0.05),
  l1 = seq(from=0, to=1e-4, by=1e-6),
  l2 = seq(from=0, to=1e-4, by=1e-6)
)

length(unique(hyper_params2$activation)) *
  length(unique(hyper_params2$hidden)) *
  length(unique(hyper_params2$input_dropout_ratio)) *
  length(unique(hyper_params2$l1)) *
  length(unique(hyper_params2$l2))

# there could be multiple stopping criteria
# so we use a list to put all of them together
search_criteria = list(
  strategy ="RandomDiscrete",
  seed=1234567,
  stopping_metric ="auto",# logloss for classification
                          # deviance for regression
  stopping_rounds=5,# stop when the last 5 models
  stopping_tolerance=0.01,# improve less than 1%
  max_runtime_secs =360,# stop when the search took more than 360 seconds
  max_models =100# stop when the search tried over 100 models
)

grid2 <- h2o.grid(
  algorithm ="deeplearning",
  grid_id ="dl_grid_random",
  x = x,
  y = y,
  training_frame = train_h2o,
  validation_frame = valid_h2o,
  epochs =1,
  stopping_metric ="logloss",
  stopping_tolerance =0.01,#stop when logloss improvement <1%
  stopping_rounds =2,#for 2 scoring events
  score_validation_samples =10000,
  score_duty_cycle =0.025,
  max_w2 =10,#can help improve stability for Rectifier
  hyper_params = hyper_params2,
  search_criteria = search_criteria
)

ordered_grid2 <- h2o.getGrid("dl_grid_random",sort_by="logloss",decreasing=F)
dl_grid_random_summary_table <- ordered_grid2@summary_table
dl_grid_random_summary_table

dl_grid_random_best_model <- h2o.getModel(dl_grid_random_summary_table$model_ids[1])
summary(dl_grid_random_best_model)

dl_grid_random_best_model_params <- dl_grid_random_best_model@allparameters
dl_grid_random_best_model_params

prediction_h2o_dl <- h2o.predict(dl_grid_random_best_model,
                                 newdata = new_data_h2o)
prediction_dl_tbl <- tibble(
  SK_ID_CURR = x_test_processed_tbl$SK_ID_CURR,
  TARGET = as.vector(prediction_h2o_dl$p1)
)

#AutoML

automl_models_h2o <- h2o.automl(
  x = x,
  y = y,
  training_frame = train_h2o,
  validation_frame = valid_h2o,
  leaderboard_frame = test_h2o,
  max_runtime_secs =300 # suppose we only have 5 minutes,
  # which is too short for real-world projects
)

x_train_processed_tbl = x_train_processed_tbl[,1:10]
x_test_processed_tbl = x_test_processed_tbl[,1:10]
# push data into h2o
data_h2o <- as.h2o(
  bind_cols(y_train_processed_tbl, x_train_processed_tbl),
  destination_frame="train.hex" #destination_frame is optional
)
# Partition the data into training, validation and test sets
splits <- h2o.splitFrame(
  data = data_h2o, ratios = c(0.7,0.15), seed =1234)
train_h2o <- splits[[1]]
valid_h2o <- splits[[2]]
test_h2o <- splits[[3]]

y <- "TARGET"
x <- setdiff(names(train_h2o), y)
m4 <- h2o.deeplearning(
  model_id ="dl_model_for_xai",
  x = x,
  y = y,
  training_frame = train_h2o,
  validation_frame = valid_h2o,
  epochs =10,
  variable_importances =T
)

library(DALEXtra)
h2o_exp = explain_h2o(m4, data = x_train_processed_tbl,
  y = y_train_processed_tbl$TARGET ==1,
  label ="H2O", type ="classification"
)
new_application = x_test_processed_tbl[1,1:10]

h2o_exp_bd <- predict_parts(
  explainer = h2o_exp, new_observation = new_application,
  type ="break_down")
plot(h2o_exp_bd) + ggtitle("Break-down plot for the new application")

h2o_exp_shap <- predict_parts(
  explainer = h2o_exp, new_observation = new_application,
  type = "shap", B =25)
h2o_exp_shap

h2o_exp_cp <- predict_profile(
  explainer = h2o_exp, new_observation = new_application)
plot(h2o_exp_cp, variables = c("AMT_CREDIT","CNT_CHILDREN")) +
  ggtitle("Ceteris-paribus profile")

h2o_exp_vip <- model_parts(
  explainer = h2o_exp, B =50, type ="difference")
plot(h2o_exp_vip) +
  ggtitle("Mean variable-importance over 50 permutations")

h2o_exp_pdp <- model_profile(
  explainer = h2o_exp, variables ="AMT_CREDIT")
plot(h2o_exp_pdp, geom="profiles") +
  ggtitle("CP & PD profiles for credit")

h2o.shutdown(prompt = F)
