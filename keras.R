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

  
db <- cbind(db%>%select(-train,-SalePrice)%>%
  mutate_if(is.numeric,scale_vec),
  db%>%select(train,SalePrice))
head(db)

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



db$SalePrice=log(db$SalePrice)

db_matrix=model.matrix(~.,data = db)


sample_train=sample(1:sum(db$train),round(.7*sum(db$train)))
labeled_data=db_matrix[db_matrix[,"train"]==1,]
train=labeled_data[sample_train,] #CAREFUL rows with NAs are removed https://stackoverflow.com/questions/6447708/model-matrix-generates-fewer-rows-than-original-data-frame

x_train <- train[,setdiff(colnames(train),"SalePrice")]
y_train <- train[,"SalePrice"]
test <- labeled_data[-sample_train,]
x_test <- test[,setdiff(colnames(test),"SalePrice")]
y_test <- test[,"SalePrice"]

batch_size <- 32
epochs <- 1000
# without batch norm ~0.15 MSE, with layer_batch_normalization() on 1st layer of 300-80-20-80-100-1. ~0.05773317 MSE ! https://arxiv.org/abs/1502.03167 
# batch norm on 2nd layer ~0.04984472 of 300-80-20-80-100-1. On 3rd layer 0.05692367. 
# don't put multiple batch norm ie at multiple layers ! it won't converge. BN at layer 1 and 2 : 0.3142703 on test set with overfitting 0.08 on validation set
# Don't put the batchNorm at the end 0.1243595, same as no norm.
# RMSE 0.176296329850813 layer_alpha_dropout(object, rate, noise_shape = NULL, seed = NULL) self normalizing neural nets https://arxiv.org/abs/1706.02515
# RMSE 0.158840984011945 layer_gaussian_noise(stddev=1) avant ativation, pas de alpha_dropout mais BN. si on enlève le BN 0.232201888600251. alpha sans BN 0.162787011016188
# RMSE 0.160561183448265 layer_gaussian_noise(stddev=5) avant ativation, pas de alpha_dropout mais BN 
# RMSE 0.181843723778723 layer_gaussian_noise(stddev=.1) avant ativation, pas de alpha_dropout mais BN
# RMSE 0.172755386881182 layer_gaussian_noise(stddev=1) avant activation avec alpha_dropout ET BN ! don't do too much 
# RMSE 0.145480264892886-0.149975798831799 (not very stable, need more iterations) GN + Alpha on layer 1, BN on layer 2 
# RMSE 0.16183850015011 GN + Alpha on layer 1, BN on layer 3
# RMSE 0.15887105310041 GN + Alpha on layer 1, BN on layer 4
# RMSE 0.36342064157412 GN + Alpha on layer 1, BN on layer 5
# RMSE 0.172411217434158 GN + Alpha on layer 1, BN on layer 2, dropout layer 2 from .3 to .1
# RMSE 0.158402344029024 GN + Alpha on layer 1, BN on layer 2, dropout layer 1 from .1 to .3

# Starting with basic NN
# RMSE 2.14580715657458 for 1 layer -> it is like a linear regression, no improvement from epoch 3 to 203
# RMSE 2.2512858435978 for 1 layer with GN and AlphaDropout 0.1 
# RMSE 0.292703298559425 for 2 layers 300 -> 80 -> 1
# RMSE 0.28056443814846 for 3 layers 300 -> 80 -> 20 -> 1




# RMSE 0.198531344319583 GN, Alpha Dropout, no BN
model <- keras_model_sequential()
build_model <- function(){
model %>%
# layer_batch_normalization()%>%
  # layer_gaussian_noise(stddev=1)%>%
  layer_dense(units = 100, input_shape = ncol(x_train)) %>% 
  layer_activation(activation = 'relu') %>% 
    
  # layer_batch_normalization()%>%
  # layer_alpha_dropout(rate = 0.1) %>%
  layer_dense(units = 80) %>%
  layer_activation(activation = 'relu') %>%
    
  # layer_batch_normalization()%>%
  # layer_dropout(rate = 0.3) %>%
  layer_dense(units = 20) %>%
  layer_activation(activation = 'relu') %>%
  # # layer_batch_normalization()%>%
  # layer_dropout(rate = 0.5) %>% 
  #   
  # layer_dense(units = 80) %>% 
  # layer_activation(activation = 'relu') %>% 
  # # layer_batch_normalization()%>%
  # layer_dropout(rate = 0.5) %>% 
  #   
  # layer_dense(units=100) %>% 
  # layer_activation(activation = 'relu') %>% 
  # # layer_batch_normalization()%>%
  # layer_dropout(rate = 0.2) %>% 
    
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
  validation_split = 0.3,callbacks=list(callback_early_stopping(monitor = "val_loss",patience=200),
                                        callback_tensorboard(),
                                        callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5,patience=50,verbose=1))
)

score <- model %>% evaluate(
  x_test, y_test,
  batch_size = batch_size,
  verbose = 1
)

print(paste0("RMSE on test set ",sqrt(score$mean_squared_error)))
}
build_model()
# tensorboard --logdir=~/Documents/HousePrices/logs then http://192.168.99.1:6006

####################################################
######Illustrate Internal Covariate Shift ##########
####################################################
model_1=clone_model(model)
model_2=clone_model(model)
model_3=clone_model(model)
model_4=clone_model(model)
model_5=clone_model(model)

model_1 %>% pop_layer()
model_2 %>% pop_layer()
model_3
model_4
model_5




summary(model)
summary(model_1)


