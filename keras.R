library(keras)



library(data.table)
library(glmnet)
library(gbm)
library(xgboost)
library(tidyverse)
db=rbind(cbind(fread("train.csv"),train=1),cbind(fread("test.csv"),SalePrice=1,train=0))
var_useless=c("Id","train")

# char=>factor
db <- db %>%
  mutate_if(sapply(db, is.character), as.factor)

# NA as a level
db <- db %>% 
  mutate_if(sapply(db, is.factor), addNA)

# scale numeric
scale_vec <- function(x){
  c(scale(x))
}
nums <- names(sapply(db,is.numeric))
db <- db%>%
  mutate_if(setdiff(nums,c("train","SalePrice")),scale_vec)

sum(sapply(db,function(x)sd(x)==1)) #no numerical constant
min(sapply(db,function(x)length(levels(factor(x))))) # no factor constant

#in this dbset, NA actually means "there isn't", not "we don't know"...
NAs <- sapply(db,function(x)sum(is.na(x))/nrow(db))
NAs <- NAs[NAs>0]
sapply(db[,names(NAs)],summary)

#LotFrontage NA is none => 0
db[is.na(db$LotFrontage),]$LotFrontage <- 0
#MasVnrArea NA is none => 0
db[is.na(db$MasVnrArea),]$MasVnrArea <- 0
db[is.na(db$BsmtHalfBath),]
for (nm in setdiff(names(NAs),names(db)[grep(pattern="Yr",x = names(db),ignore.case = T)])){
  print(nm)
  db[is.na(db[[nm]]),nm]<-0
}
library(Hmisc)
db$GarageYrBlt_cut=addNA(cut2(db$GarageYrBlt,g = 10))
db=within(db,rm(GarageYrBlt))

sample_gbm=sample(1:sum(db$train),round(.5*sum(db$train)))



# db$SalePrice=log(db$SalePrice)

db_matrix=model.matrix(~.,data = db)


sample_train=sample(1:sum(db$train),round(.7*sum(db$train)))
labeled_data=db_matrix[db_matrix[,"train"]==1,]
train=labeled_data[sample_train,] #CAREFUL rows with NAs are removed https://stackoverflow.com/questions/6447708/model-matrix-generates-fewer-rows-than-original-data-frame

x_train <- train[,setdiff(colnames(train),"SalePrice")]
y_train <- train[,"SalePrice"]
test <- labeled_data[-sample_train,]
x_test <- test[,setdiff(colnames(test),"SalePrice")]
y_test <- test[,"SalePrice"]

rm(model)
gc()
batch_size <- 32
epochs <- 1000
# without batch norm ~0.15 MSE, with layer_batch_normalization() on 1st layer of 300-80-20-80-100-1. ~0.05773317 MSE ! https://arxiv.org/abs/1502.03167 
# batch norm on 2nd layer ~0.04984472 of 300-80-20-80-100-1. On 3rd layer 0.05692367. 
# don't put norm at each layer ! it won't converge. Don't put the batchNorm at the end 0.1243595, same as no norm.
# layer_alpha_dropout(object, rate, noise_shape = NULL, seed = NULL) self normalizing neural nets https://arxiv.org/abs/1706.02515
model <- keras_model_sequential()
build_model <- function(){
model %>%
  layer_dense(units = 300, input_shape = ncol(x_train)) %>% 
  layer_activation(activation = 'relu') %>% 
  layer_batch_normalization()%>%
  layer_dropout(rate = 0.1) %>% 
    
  layer_dense(units = 80) %>% 
  layer_activation(activation = 'relu') %>% 
  layer_batch_normalization()%>%
  layer_dropout(rate = 0.3) %>% 
    
  layer_dense(units = 20) %>% 
  layer_activation(activation = 'relu') %>%
  # layer_batch_normalization()%>%
  layer_dropout(rate = 0.5) %>% 
    
  layer_dense(units = 80) %>% 
  layer_activation(activation = 'relu') %>% 
  # layer_batch_normalization()%>%
  layer_dropout(rate = 0.5) %>% 
    
  layer_dense(units=100) %>% 
  layer_activation(activation = 'relu') %>% 
  # layer_batch_normalization()%>%
  layer_dropout(rate = 0.2) %>% 
    
  layer_dense(units=1)%>%
  layer_activation(activation = 'linear')

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam',#try also sgd 
  metrics = c('mean_squared_error')
)

history <- model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 2,
  validation_split = 0.1
)

score <- model %>% evaluate(
  x_test, y_test,
  batch_size = batch_size,
  verbose = 1
)

print(score)
}
build_model()




