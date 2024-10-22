install.packages("DALEX")
install.packages("DALEXtra")
install.packages("lime")
install.packages(c("rms","randomForest","e1071"))
library("tidyverse")
library("DALEX")
library("DALEXtra")
library("rms")
library("randomForest")
library("e1071")

glimpse(titanic)
glimpse(apartments)

#Titanic
titanic = titanic |>
  mutate(country = as.character(country)) |>
  replace_na(list(age=30, country ="X", sibsp =0, parch =0)) |>
  mutate(country = factor(country))
titanic$fare[is.na(titanic$fare) & titanic$class =="1st"] =89
titanic$fare[is.na(titanic$fare) & titanic$class =="2nd"] =22
titanic$fare[is.na(titanic$fare) & titanic$class =="3rd"] =13


# the rcs() function allows us to model potentially non-linear effect of age
titanic_lmr <- lrm(survived =="yes"~ gender + rcs(age) + class +
                     sibsp + parch + fare + embarked, titanic)

set.seed(123)
titanic_rf <- randomForest(survived ~ class + gender + age +
                             sibsp + parch + fare + embarked, data = titanic)

titanic_svm <- svm(survived =="yes"~ class + gender + age + sibsp + parch + fare + embarked, data = titanic,
                   type ="C-classification", probability =TRUE)

johnny <- data.frame(class = factor("1st",
  levels = c("1st","2nd","3rd","deck crew","engineering crew",
             "restaurant staff","victualling crew")),
  gender = factor("male", levels = c("female","male")),
  age =8, sibsp =0, parch =0, fare =72,
  embarked = factor("Southampton",
                    levels = c("Belfast","Cherbourg","Queenstown","Southampton")))

(pred_lmr <- predict(titanic_lmr, johnny, type ="fitted"))
(perd_svm <- predict(titanic_svm, johnny, probability = TRUE))
(pred_rf <- predict(titanic_rf, johnny, type ="prob"))

#Apartments

apartments_lm <- lm(m2.price ~ ., data = apartments)
set.seed(123)
apartments_rf <- randomForest(m2.price ~ ., data = apartments)
apartments_svm <- svm(m2.price ~ construction.year + surface + floor +
                        no.rooms + district, data = apartments)

#Model Explainers

titanic_lmr_exp <- explain(
  model = titanic_lmr,
  data = titanic[, -9],
  y = titanic$survived =="yes",
  label="Logistic Regression",
  type="classification")

titanic_rf_exp <- explain(
  model = titanic_rf,
  data = titanic[, -9],
  y = titanic$survived =="yes",
  label ="Random Forest",
  type ="classification")

titanic_svm_exp <- explain(
  model = titanic_svm,
  data = titanic[, -9],
  y = titanic$survived =="yes",
  label ="Support Vector Machine",
  type ="classification")

apartments_svm_exp <- explain(
  model = apartments_svm, data = apartments_test[, -1],
  y = apartments_test$m2.price,
  label ="Support Vector Machine",
  type ="regression")

#BreakDown Plots

titanic_svm_exp_bd <- predict_parts(
  titanic_svm_exp, new_observation = johnny,
  type = "break_down")
plot(titanic_svm_exp_bd)+ggtitle("Break-down plot for Johnny")

new_apt = apartments_test[1, -1]
apartments_svm_exp_bd <- predict_parts(
  apartments_svm_exp, new_observation = new_apt,
  type ="break_down")
plot(apartments_svm_exp_bd) + ggtitle("Break-down plot for a new apartment")

#iBD Plots

titanic_svm_exp_bdi <- predict_parts(
  titanic_svm_exp, new_observation = johnny,
  type = "break_down_interactions")
plot(titanic_svm_exp_bd)+ggtitle("iBreak-down plot for Johnny")

apartments_svm_exp_bdi <- predict_parts(
  apartments_svm_exp, new_observation = new_apt,
  type ="break_down_interactions")
plot(apartments_svm_exp_bdi) + ggtitle("iBreak-down plot for a new apartment")

#SHAP for Average Attributions

titanic_svm_exp_shap <- predict_parts(
  explainer = titanic_svm_exp,
  new_observation = johnny,
  type = "shap", B=25)
titanic_svm_exp_shap
plot(titanic_svm_exp_shap)+ggtitle("SHAP plot for Johnny")

apartments_svm_exp_shap <- predict_parts(
  explainer = apartments_svm_exp, new_observation = new_apt,
  type ="shap", B=25)
plot(apartments_svm_exp_shap) + ggtitle("SHAP plot for a new apartment")

#Lime

model_type.dalex_explainer <- DALEXtra::model_type.dalex_explainer
predict_model.dalex_explainer <- DALEXtra::predict_model.dalex_explainer
# use predict_surrogate(), not predict_parts()
titanic_svm_exp_lime <- predict_surrogate(
  explainer = titanic_svm_exp, new_observation = johnny,
  n_features =3,
  n_permutations =1000,
  type ="lime")
plot(titanic_svm_exp_lime)

apartments_svm_exp_lime <- predict_surrogate(
  explainer = apartments_svm_exp, new_observation = new_apt,
  n_features =3,
  n_permutations =1000, 
  type ="lime")
plot(apartments_svm_exp_lime)

#Ceteris-paribus Profiles

titanic_svm_exp_cp <- predict_profile(
  explainer = titanic_svm_exp, new_observation = johnny)
plot(titanic_svm_exp_cp, variables = c("age","fare")) +
  ggtitle("Ceteris-paribus profile for Titanic")

apartments_svm_exp_cp <- predict_profile(
  explainer = apartments_svm_exp, new_observation = new_apt)
plot(apartments_svm_exp_cp, variables = c("no.rooms","surface")) +
  ggtitle("Ceteris-paribus profile for Apartments")

#Variable-importance Measures

vip_lm <- model_parts(explainer = titanic_lmr_exp, B =20,N =NULL)
vip_rf <- model_parts(explainer = titanic_rf_exp, B =20, N =NULL)
vip_svm <- model_parts(explainer = titanic_svm_exp, B =20, N =NULL)
plot(vip_lm, vip_rf, vip_svm) +ggtitle("Mean variable-importance over 20 permutations")

#Partial-dependence Profiles

titanic_svm_exp_pdp <- model_profile(
  explainer = titanic_svm_exp,
  variables ="age")
plot(titanic_svm_exp_pdp) + ggtitle("PD profile for age")
plot(titanic_svm_exp_pdp, geom="profiles")+ggtitle("CP & PD profiles for age")

apartments_svm_exp_pdp <- model_profile(
  explainer = apartments_svm_exp,
  variables ="surface")
plot(apartments_svm_exp_pdp) + ggtitle("PD profile for surface")
plot(apartments_svm_exp_pdp, geom="profiles")+ggtitle("CP & PD profiles for surface")
