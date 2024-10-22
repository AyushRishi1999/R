library(keras)
library(tidyverse)
library(reticulate)
mnist <- keras::dataset_mnist()
str(mnist)

mnist <- dataset_mnist()
# get the data from the internet; ~200MB
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

length(dim(x_train))
dim(x_train)
typeof(x_train)
digit <- x_train[5,,]
plot(as.raster(digit, max =255))

write_rds(mnist,"mnist.rds")
mnist <- read_rds("mnist.rds")
c(c(x_train, y_train), c(x_test, y_test)) %<-% mnist

my_slice <- x_train[10:99,,]
dim(my_slice)

my_slice <- x_train[10:99,1:28,1:28]
dim(my_slice)

str(x_train)
x_train <- array_reshape(x_train, c(nrow(x_train),784))
x_test <- array_reshape(x_test, c(nrow(x_test),784))
str(x_train)

x_train <- x_train /255
x_test <- x_test /255
str(y_train)

y_train <- to_categorical(y_train,10)
y_test <- to_categorical(y_test,10)
str(y_train)

#Defining the model

model <- keras_model_sequential()
## Define the structure of the neural net
model |>
  # A dense layer is a fully connected layer
  layer_dense(units =256, activation ='relu', input_shape = c(784)) |>
  layer_dropout(rate =0.3) |> # randomly set 40% of weights to 0
  layer_dense(units =128, activation ='relu') |>
  layer_dropout(rate =0.2) |># this helps prevent overfitting
  layer_dense(units =10, activation ='softmax')# probability of each class
summary(model)

model |> compile(
  optimizer ="adam",# see next slide
  loss ="categorical_crossentropy",# since we have 10 categoreis
  metrics = c("accuracy")# for classification
)

history <- model |> fit(
  x_train, y_train,
  batch_size =128,# a set of 128 samples
  epochs =20,# let's go through x_train 20 times
  validation_split =0.2 # use the last 20% of train data for validation
)
plot(history)

#Instantiating a small convnet

model <- keras_model_sequential() |>
  layer_conv_2d(input_shape = c(28,28,1),# input image shape
  filters =32,# output channel size
  kernel_size = c(3,3),
  activation ="relu") |>
  layer_max_pooling_2d(pool_size = c(2,2)) |>
  layer_conv_2d(filters =64, kernel_size = c(3,3), activation ="relu") |>
  layer_max_pooling_2d(pool_size = c(2,2)) |>
  layer_conv_2d(filters =64, kernel_size = c(3,3), activation ="relu")
summary(model)

model <- model |># the output of the last conv2d is 3D (3, 3, 64)
  layer_flatten() |># flatten it into 1D (576)
  layer_dense(units =64, activation ="relu") |>
  layer_dense(units =10, activation ="softmax")
summary(model)

#Training the convnet on MNIST images

mnist <- read_rds("mnist.rds")
# assuming that you have saved the data locally
c(c(x_train, y_train), c(x_test, y_test)) %<-% mnist
x_train <- array_reshape(x_train, c(60000,28,28,1))
x_train <- x_train /255
x_test <- array_reshape(x_test, c(10000,28,28,1))
x_test <- x_test /255
y_train <- to_categorical(y_train)
y_test <- to_categorical(y_test)
model |> compile(
  optimizer ="adam",
  loss ="categorical_crossentropy",
  metrics = c("accuracy")
)
model |> fit(
  x_train, y_train,# For simplicity, we use a small number for
  epochs =5,
  batch_size=64# epochs and batch_size
)
results <- model |> evaluate(x_test, y_test)
results

