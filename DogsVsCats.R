original_dataset_dir <-"data/dogs-vs-cats/train"# we will only use the labelled data
base_dir <-"data/cats_and_dogs_small"# to store a subset of data that we are going to use
dir.create(base_dir)
train_dir <- file.path(base_dir,"train")
dir.create(train_dir)
validation_dir <- file.path(base_dir,"validation")
dir.create(validation_dir)
test_dir <- file.path(base_dir,"test")
dir.create(test_dir)
train_cats_dir <- file.path(train_dir,"cats")
dir.create(train_cats_dir)
train_dogs_dir <- file.path(train_dir,"dogs")
dir.create(train_dogs_dir)
validation_cats_dir <- file.path(validation_dir,"cats")
dir.create(validation_cats_dir)
validation_dogs_dir <- file.path(validation_dir,"dogs")
dir.create(validation_dogs_dir)
test_cats_dir <- file.path(test_dir,"cats")
dir.create(test_cats_dir)
test_dogs_dir <- file.path(test_dir,"dogs")
dir.create(test_dogs_dir)

fnames <- paste0("cat.",1:1000,".jpg")# use invisible to supress output message from file.copy()
invisible(file.copy(file.path(original_dataset_dir, fnames),
                    file.path(train_cats_dir)))
fnames <- paste0("cat.",1001:1500,".jpg")
invisible(file.copy(file.path(original_dataset_dir, fnames),
                    file.path(validation_cats_dir)))
fnames <- paste0("cat.",1501:2000,".jpg")
invisible(file.copy(file.path(original_dataset_dir, fnames),
                    file.path(test_cats_dir)))
fnames <- paste0("dog.",1:1000,".jpg")
invisible(file.copy(file.path(original_dataset_dir, fnames),
                    file.path(train_dogs_dir)))
fnames <- paste0("dog.",1001:1500,".jpg")
invisible(file.copy(file.path(original_dataset_dir, fnames),
                    file.path(validation_dogs_dir)))
fnames <- paste0("dog.",1501:2000,".jpg")
invisible(file.copy(file.path(original_dataset_dir, fnames),
                    file.path(test_dogs_dir)))

cat("total training cat images:", length(list.files(train_cats_dir)),"\n")
cat("total training dog images:", length(list.files(train_dogs_dir)),"\n")
cat("total validation cat images:", length(list.files(validation_cats_dir)),"\n")
cat("total validation dog images:", length(list.files(validation_dogs_dir)),"\n")
cat("total test cat images:", length(list.files(test_cats_dir)),"\n")
cat("total test dog images:", length(list.files(test_dogs_dir)),"\n")

library(keras)
model_v1 <- keras_model_sequential() |>
  layer_conv_2d(filters =32, kernel_size = c(3,3), activation ="relu",input_shape = c(150,150,3)) |>
  layer_max_pooling_2d(pool_size = c(2,2)) |>
  layer_conv_2d(filters =64, kernel_size = c(3,3), activation ="relu") |>
  layer_max_pooling_2d(pool_size = c(2,2)) |>
  layer_conv_2d(filters =128, kernel_size = c(3,3), activation ="relu") |>
  layer_max_pooling_2d(pool_size = c(2,2)) |>
  layer_conv_2d(filters =128, kernel_size = c(3,3), activation ="relu") |>
  layer_max_pooling_2d(pool_size = c(2,2)) |>
  layer_flatten() |>
  layer_dense(units =512, activation ="relu") |>
  layer_dense(units =1, activation ="sigmoid")

model_v1 |> compile(
  optimizer ="adam",
  loss ="binary_crossentropy",
  metrics = c("accuracy")
)
summary(model_v1)

train_datagen <- image_data_generator(rescale =1/255)
validation_datagen <- image_data_generator(rescale =1/255)
train_generator <- flow_images_from_directory(
  train_dir,# Target directory
  train_datagen,# Training data generator
  target_size = c(150,150),# Resizes all images to 150 × 150
  batch_size =20,# 20 samples in one batch
  class_mode ="binary"# Because we use binary_crossentropy loss,
  # we need binary labels.
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(150,150),
  batch_size =20,
  class_mode ="binary"
)

history_v1 <- model_v1 |>
  fit(
    train_generator,
    steps_per_epoch =30,
    epochs =10,
    validation_data =validation_generator,
    validation_steps =10
  )
plot(history_v1)

datagen <- image_data_generator(
  rescale =1/255,
  rotation_range =40,# randomly rotate images up to 40 degrees
  width_shift_range =0.2,# randomly shift 20% pictures horizontally
  height_shift_range =0.2,# randomly shift 20% pictures vertically
  shear_range =0.2,# randomly apply shearing transformations
  zoom_range =0.2,# randomly zooming inside pictures
  horizontal_flip =TRUE,# randomly flipping half the images horizontally
  fill_mode ="nearest"# used for filling in newly created pixels
)

fnames <- list.files(train_cats_dir, full.names =TRUE)
img_path <- fnames[[3]]# Chooses one image to augment
img <- image_load(img_path, target_size = c(150,150))
img_array <- image_to_array(img)# Converts the shape back to (150, 150, 3)
img_array <- array_reshape(img_array, c(1,150,150,3))
augmentation_generator <- flow_images_from_data(
  img_array,
  generator = datagen,
  batch_size =1
)
op <- par(mfrow = c(2,2), pty ="s", mar = c(1,0,1,0))
for(i in 1:4) {
  batch <- generator_next(augmentation_generator)
  plot(as.raster(batch[1,,,]))
}
par(op)

model_v2 <- keras_model_sequential() |>
  layer_conv_2d(filters =32, kernel_size = c(3,3), activation ="relu",input_shape = c(150,150,3)) |>
  layer_max_pooling_2d(pool_size = c(2,2)) |>
  layer_conv_2d(filters =64, kernel_size = c(3,3), activation ="relu") |>
  layer_max_pooling_2d(pool_size = c(2,2)) |>
  layer_conv_2d(filters =128, kernel_size = c(3,3), activation ="relu") |>
  layer_max_pooling_2d(pool_size = c(2,2)) |>
  layer_conv_2d(filters =128, kernel_size = c(3,3), activation ="relu") |>
  layer_max_pooling_2d(pool_size = c(2,2)) |>
  layer_flatten() |>
  layer_dropout(rate =0.5) |> # randomly set 50% of weights to 0
  layer_dense(units =512, activation ="relu") |>
  layer_dense(units =1, activation ="sigmoid")

model_v2 |> compile(
  optimizer ="adam",
  loss ="binary_crossentropy",
  metrics = c("accuracy")
)

test_datagen <- image_data_generator(rescale =1/255)# no data augmentation
train_generator <- flow_images_from_directory(
  train_dir,
  datagen,# Our data augmentation configuration defined earlier
  target_size = c(150,150),
  batch_size =32,
  class_mode ="binary"
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  test_datagen,# Note that the validation data shouldn't be augmented!
  target_size = c(150,150),
  batch_size =32,
  class_mode ="binary"
)

history_v2 <- model_v2 |> fit_generator(
  train_generator,
  steps_per_epoch =30,
  epochs =20,
  validation_data = validation_generator,
  validation_steps =10
)

test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,# Note that the test data shouldn't be augmented!
  target_size = c(150,150),
  batch_size =20,
  class_mode ="binary"
)

model_v2 |> evaluate(test_generator, steps =50)

conv_base <- application_vgg16(
  weights ="imagenet",
  include_top =FALSE,
  input_shape = c(150,150,3)
)
conv_base

datagen <- image_data_generator(rescale =1/255)
batch_size <-20
extract_features <-function(directory, sample_count) {
  features <- array(0, dim = c(sample_count,4,4,512))
  labels <- array(0, dim = c(sample_count))
  generator <- flow_images_from_directory(
    directory = directory, generator = datagen,
    target_size = c(150,150), 
    batch_size = batch_size,
    class_mode ="binary"
  )
  i <-0
  while(TRUE) {
    batch <- generator_next(generator)
    inputs_batch <- batch[[1]]; 
    labels_batch <- batch[[2]]
    features_batch <- conv_base |> predict(inputs_batch)
    index_range <- ((i * batch_size)+1):((i +1) * batch_size)
    features[index_range,,,] <- features_batch
    labels[index_range] <- labels_batch
    i <- i +1 
    if(i * batch_size >= sample_count)
      break
  }
  return (list(features = features,labels = labels))
}

train <- extract_features(train_dir,2000)# will take a while since we are running
validation <- extract_features(validation_dir,1000)# our images through conv_base
test <- extract_features(test_dir,1000)

reshape_features <-function(features) {
  array_reshape(features, dim = c(nrow(features),4*4*512))
}

train$features <- reshape_features(train$features)
validation$features <- reshape_features(validation$features)
test$features <- reshape_features(test$features)

model_v3 <- keras_model_sequential() |>
  layer_dense(units =256, activation ="relu",input_shape =4*4*512) |>
  layer_dropout(rate =0.5) |>
  layer_dense(units =256, activation ="relu",input_shape =4*4*512) |>#your turn
  layer_dropout(rate =0.3) |>#your turn
  layer_dense(units =1, activation ="sigmoid")

model_v3 |> compile(
  optimizer ="adam",
  loss ="binary_crossentropy",
  metrics = c("accuracy")
)

history_v3 <- model_v3 |> fit(
  train$features, train$labels,
  epochs =10,
  batch_size =20,
  validation_data=list(validation$features, validation$labels)
)

plot(history_v3)
model_v3 |> evaluate(test$features, test$labels)
