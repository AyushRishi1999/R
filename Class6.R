library(tidyverse)
library(reticulate)
library(keras)

model <- load_model_hdf5("data/imdb_1d_convnet.h5")

samples <- c("The cat sat on the mat.","The dog ate my homework.")
token_index <- list()
for(sample in samples) {
  for(word in strsplit(sample," ")[[1]]) {
    if(!word % in % names(token_index)) {
      # We don't attribute index 1 to anything. Start from 2.
      token_index[[word]] <- length(token_index) + 2
    }
  }
}
max_length <- 10 # We'll only consider the first 10 words in each sample.
results <- array(0, dim = c(length(samples), max_length, max(as.integer(token_index))))
for(i in 1:length(samples)) {
  sample <- samples[[i]]
  words <- head(strsplit(sample," ")[[1]], n = max_length)
  for(j in 1:length(words)) {
    index <- token_index[[words[[j]]]]
    results[[i, j, index]] <- 1
  }
}

rbind(
  colSums=(results[1,,]),
  colSums=(results[2,,])
)

library(keras)
samples <- c("The cat sat on the mat.","The dog ate my homework.")
# Creates a tokenizer, configured to only take into
# account the 10 most common words
tokenizer <- text_tokenizer(num_words =10) |>
  fit_text_tokenizer(samples)# Builds the word index
word_index <- tokenizer$word_index # Turns strings into lists of integer indices
sequences <- texts_to_sequences(tokenizer, samples)
one_hot_results <- texts_to_matrix(tokenizer, samples, mode = "binary")
one_hot_results

embedding_layer <- layer_embedding(
  input_dim = 1000, # the number of possible tokens
  output_dim =64
)

#IMDB

max_features <- 10000 # Number of words to consider as features
maxlen <- 200 # Cuts off the text after this number of words
imdb <- dataset_imdb(num_words = max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb
# Loads the data
# Turns the lists of integers into a 2D tensor of shape (samples, maxlen)
# Will talk more about pad_sequences later
x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)

model <- keras_model_sequential() |>
  layer_embedding(input_dim = max_features, # num of unique words
                  input_length = maxlen, # num of words in a review
                  output_dim = 8) |>
  layer_flatten() |> # From (samples, maxlen, 8) to (samples, maxlen * 8)
  layer_dense(units = 1, activation = "sigmoid") # Adds the classifier on top
model |> compile(optimizer = "adam", loss = "binary_crossentropy",metrics = c("accuracy"))

summary(model)

history <- model |> fit(
  x_train, y_train,
  epochs =10,
  batch_size =32,
  validation_split =0.2
)

my_word_embedding = get_layer(model, index = 1) |>
  get_weights() %>% .[[1]] # the base R pipe currently does not support .[[1]
str(my_word_embedding)

imdb_word_index = dataset_imdb_word_index()
imdb_word_index

imdb_word_index_sub = imdb_word_index[map_lgl(imdb_word_index, ~.<=max_features)]
imdb_word_index_sub

predictions = model |>
  predict(x_test) %>% # the base R pipe currently does not support `>`
  `>`(0.5) |> k_cast("int32") |>
  as.integer()

df = tibble(true_label = y_test, predicted_label = predictions)
df

imdb_word_index_sub$horror
imdb_word_index_sub$action
imdb_word_index_sub$actor

horror_vec = my_word_embedding[186, ]
action_vec = my_word_embedding[203, ]
actor_vec = my_word_embedding[281, ]

euclidean_dist <- function(x, y) sqrt(sum((x - y)^2))
ed_vec = map_dbl(1:nrow(my_word_embedding), function(i){
  current_vec = my_word_embedding[i, ]
  ed = euclidean_dist(actor_vec, current_vec)
  return(ed)
})
ed_vec

word_vec = names(imdb_word_index_sub)
index_vec = map_int(imdb_word_index_sub, ~.)
df = data.frame(word=word_vec, index=index_vec)
view(df)
df=df|>arrange(index)
df=df|>mutate(ed=ed_vec)

#raw IMDB data

aclImdb_dir <- "data/aclImdb"
train_dir <- file.path(aclImdb_dir,"train")
labels <- c()
texts <- c()
for(label_type in c("neg", "pos")) {
  label <- switch(label_type, neg = 0, pos = 1)
  dir_name <- file.path(train_dir, label_type)
  for(fname in list.files(dir_name, pattern = glob2rx("*.txt"),
   full.names =TRUE)) {
    texts <- c(texts, readChar(fname, file.info(fname)$size))
    labels <- c(labels, label)
  }
}

max_features <-10000# only consider the 10k most common words
maxlen <-200# only consider the first 200 words in each review
training_samples <-200# only use 200 reviews to train our model
validation_samples <-5000
tokenizer <- text_tokenizer(num_words = max_features) |>
  fit_text_tokenizer(texts)
sequences <- texts_to_sequences(tokenizer, texts)
word_index = tokenizer$word_index
cat("Found", length(word_index),"unique tokens.\n")

data <- pad_sequences(sequences, maxlen = maxlen)
labels <- as.array(labels)
cat("Shape of data tensor (Num Docs, Num Words in a Doc):", dim(data),"\n")

cat('Shape of label tensor (Num Docs):', dim(labels),"\n")

word_index
sequences

texts[1]# first review
data[1,1:100]
data[1,101:200]

map_chr(data[1,97:110], ~names(word_index)[.])
map_chr(data[1,190:200], ~names(word_index)[.])

texts[2]# second review
data[2,1:100]
data[2,101:200]
map_chr(data[2,97:110], ~names(word_index)[.])
map_chr(data[2,190:200], ~names(word_index)[.])

# Splits the data into a training set and a validation set,
# but first shuffles the data, because we're starting with
# data in which samples are ordered (all negative first, then all positive)
set.seed(123)
indices <- sample(1:nrow(data))
training_indices <- indices[1:training_samples]
validation_indices <- indices[(training_samples + 1):
                                (training_samples + validation_samples)]
x_train <- data[training_indices,]
y_train <- labels[training_indices]
x_val <- data[validation_indices,]
y_val <- labels[validation_indices]

glove_dir = "data/glove.6B/"
lines <- readLines(file.path(glove_dir, "glove.6B.50d.txt"))
embeddings_index <- new.env(hash = TRUE, parent = emptyenv())
for(i in 1:length(lines)) {
  line <- lines[[i]]
  values <- strsplit(line," ")[[1]]
  word <- values[[1]]
  embeddings_index[[word]] <- as.double(values[-1])
}
cat("Found", length(embeddings_index),"word vectors.\n")

embedding_dim <- 50
embedding_matrix <- array(0, dim = c(max_features, embedding_dim))# 10k x 50
for(word in names(word_index)) {# for every word
  index <- word_index[[word]]# get its index
  if(index < max_features) {
    # only consider the top 10k words
    # get the word's embedding vector from GloVe
    embedding_vector <- embeddings_index[[word]]
    if(!is.null(embedding_vector)) {# if GloVe has the embedding vector
      # index 1 isn't supposed to stand for any word or token
      # --it's a placeholder. So we skip 1 here:
      embedding_matrix[index+1,] <- embedding_vector
    }
  }
}

str(embedding_matrix)
embedding_matrix[1, ]
embedding_matrix[2, ]

#Your Turn
word_index$horror
word_index$action.
word_index$actor
embedding_matrix[185, ]
embedding_matrix[201, ]
actor_matrix = embedding_matrix[280, ]
ed_matrix = map_dbl(1:nrow(embedding_matrix), function(i){
  current_matrix = embedding_matrix[i, ]
  ed = euclidean_dist(actor_matrix, current_matrix)
  return(ed)
})
ed_matrix

word_matrix = names(word_index)
index_matrix = map_int(word_index, ~.)
df_matrix = data.frame(word=word_vec, index=index_vec)
df_matrix=df_matrix|>arrange(index)
df_matrix=df_matrix|>mutate(ed=ed_matrix)
view(df_matrix)

model_with_glove <- keras_model_sequential() |>
  layer_embedding(input_dim = max_features,
                  input_length = maxlen,
                  output_dim = embedding_dim) |>
  layer_flatten() |>
  layer_dense(units =32, activation ="relu") |>
  layer_dense(units =1, activation ="sigmoid")
summary(model_with_glove)

get_layer(model_with_glove, index =1) |> # manually configure the embedding
  set_weights(list(embedding_matrix)) |> # set the weights based on GloVe
  freeze_weights() # do not update the weights in this layer anymore
model_with_glove |> compile(
  optimizer ="adam",
  loss ="binary_crossentropy",
  metrics = c("accuracy")
)
history <- model_with_glove |> fit(
  x_train, y_train,
  epochs =20,
  batch_size =32,
  validation_data = list(x_val, y_val)
)
model_with_glove |> save_model_hdf5("imdb_word_embedding_with_glove.h5")
plot(history)

model_without_glove <- keras_model_sequential() |>
  layer_embedding(input_dim = max_features,
                  input_length = maxlen,
                  output_dim = embedding_dim) |>
  layer_flatten() |>
  layer_dense(units =32, activation ="relu") |>
  layer_dense(units =1, activation ="sigmoid")

model_without_glove |> compile(
  optimizer ="adam",
  loss ="binary_crossentropy",
  metrics = c("accuracy")
)

history2 <- model_without_glove |> fit(
  x_train, y_train,
  epochs =20,
  batch_size =32,
  validation_data = list(x_val, y_val)
)

model_without_glove |> save_model_hdf5("imdb_word_embedding_without_glove.h5")
plot(history2)

test_dir <- file.path(aclImdb_dir,"test")
test_labels <- c()
test_texts <- c()
for(label_type in c("neg","pos")) {
  label <-switch(label_type, neg =0, pos =1)
  dir_name <- file.path(test_dir, label_type)
  for(fname in list.files(dir_name, pattern = glob2rx("*.txt"),
   full.names =TRUE)) {
    test_texts <- c(test_texts, readChar(fname, file.info(fname)$size))
    test_labels <- c(test_labels, label)
  }
}
sequences <- texts_to_sequences(tokenizer, test_texts)
x_test <- pad_sequences(sequences, maxlen = maxlen)
y_test <- as.array(test_labels)

model_with_glove |> evaluate(x_test, y_test)
model_without_glove |> evaluate(x_test, y_test)

#Recurrent Neural Networks

model <- keras_model_sequential() |>
  layer_embedding(input_dim = 10000, output_dim = 32) |>
  layer_simple_rnn(units = 32, , return_sequences = FALSE) # return_sequences = FALSE, by default
summary(model)

model2 <- keras_model_sequential() |>
  layer_embedding(input_dim = 10000, output_dim =32) |>
  layer_simple_rnn(units = 32, return_sequences = TRUE)
summary(model2)

model3 <- keras_model_sequential() |>
  layer_embedding(input_dim = 10000, output_dim = 32) |>
  layer_simple_rnn(units = 32, return_sequences = TRUE) |>
  layer_simple_rnn(units = 32, return_sequences = TRUE) |>
  layer_simple_rnn(units = 32, return_sequences = TRUE) |>
  layer_simple_rnn(units = 32, return_sequences = FALSE)
summary(model3)

max_features <- 10000
maxlen <- 500
imdb <- dataset_imdb(num_words = max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb
cat(length(x_train), "train sequences\n")

cat(length(x_test), "test sequences")

x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)
cat("x_train shape:", dim(x_train), "\n")

cat("x_test shape:", dim(x_test), "\n")

model_dir = "data/deeplearning-pretrained/"
model <- load_model_hdf5(str_c(model_dir,"imdb_simple_rnn.h5"))
history <- readRDS(str_c(model_dir,"imdb_simple_rnn_history.rds"))
''' #Do not run: Will Take time
model <- keras_model_sequential() |>
  layer_embedding(input_dim = max_features, output_dim =32) |>
  layer_simple_rnn(units =32) |>
  layer_dense(units =1, activation ="sigmoid")
model |> compile(
  optimizer ="adam",
  loss ="binary_crossentropy",
  metrics = c("accuracy")
)
history <- model |> fit(
  x_train, y_train,
  epochs =10,
  batch_size =32,
  validation_split =0.2
)
'''
plot(history)
