library(data.table)
library(glmnet)
library(gbm)
library(xgboost)
library(tidyverse)
library(doParallel)
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

##########################
#### ONE HOT ENCODING ####
##########################
# var_factor=names(db)[sapply(db,is.factor)]
# db <- with(db,
#      data.frame(model.matrix(~.-1,db[,var_factor]),
#                 db[,setdiff(names(db),var_factor)]))

# decreases the performance ! takes A LOT MORE time and give significantly less good results.
# depth.depth shrinkage.shrinkage        bag.fraction      n.minobsinnode             n.trees 
# 5.0000000           0.0050000           0.3000000          10.0000000        2000.0000000 
# nb_split.nb_split                RMSE 
# 10.0000000           0.1279916 
# and even with very slow learning
# Using 10000 trees...
# depth.depth shrinkage.shrinkage        bag.fraction      n.minobsinnode             n.trees   nb_split.nb_split                RMSE 
# 8.000000e+00        1.000000e-03        3.000000e-01        1.000000e+01        1.000000e+04        1.000000e+01        1.276744e-01 


#GarageYrBlt don't know when the garage was build ? but is there a garage ?
#sample_train=sample(1:sum(db$train),round(.5*sum(db$train)))
test_db=db%>%filter(train==0)
db_matrix=model.matrix(~.,data = db)

train_db=db%>%filter(train==1)
train_id <- which(colnames(db_matrix)=="train")

test_matrix=db_matrix[db_matrix[,train_id]==0,]
train_db=train_db[,setdiff(names(db),var_useless)]
train_db$logSalePrice=log(train_db$SalePrice)
random=sample(1:nrow(train_db))

iter=c(depth=0,shrinkage=0,bag.fraction=0,n.minobsinnode=0,n.trees=0,nb_split=0,RMSE_glm=0,RMSE_gbm=0)

nb_split=10
# best currently is without one hot encoding
#depth.depth shrinkage.shrinkage        bag.fraction      n.minobsinnode             n.trees 
#5.0000000           0.0050000           0.3000000          10.0000000        2000.0000000 
#nb_split.nb_split                RMSE 
#10.0000000           0.1257294 


# Using 5886 trees...
# depth.depth shrinkage.shrinkage        bag.fraction      n.minobsinnode             n.trees   nb_split.nb_split                RMSE 
# 8.000000e+00        1.000000e-03        3.000000e-01        1.000000e+01        1.000000e+04        1.000000e+01        1.255727e-01 
cl <- makeCluster(spec=6)
registerDoParallel(cl = cl)



grid=expand.grid(depth=c(10),nb_split=c(10),shrinkage=c(.0001),bag.fraction=c(.1,.3,.5),n.minobsinnode=c(40,15,5))
#x=grid[1,]

results <- foreach(i=1:nrow(grid),.combine=cbind,.packages = c("gbm","glmnet","tidyverse","data.table")) %dopar%{
  x=c(grid[i,])
  predictions_table <- data.frame("id"=1:nrow(test_db))
  depth=x[[1]]
  nb_split=x[[2]]
  shrinkage=x[[3]]
  n.trees=10/shrinkage
  bag.fraction=x[[4]]
  n.minobsinnode=x[[5]]

  
  step_size=round(nrow(train_db)/nb_split)
  size=c()
  SS_gbm=c()
  SS_glm=c()
  
  for (i in 1:nb_split){
    if(i==nb_split){
      sample_train=seq(from = (i-1)*step_size+1,to = nrow(train_db),by = 1)
    }
    else sample_train=seq((i-1)*step_size+1,i*step_size)
    # if (n.minobsinnode>=bag.fraction*(step_size)){
    #   print(paste("minobsinnode",n.minobsinnode))
    #   print(paste("bag fraction x sample_size",bag.fraction*(nrow(train_db)-length(sample_train))))
    #   print(paste("bag fraction x sample_size",bag.fraction*(length(sample_train))))
    #   print("gonna crash")
    # }
    ##########################
    ####      TRY GBM     ####
    ##########################
  var_to_keep=setdiff(names(train_db),"SalePrice")
  gbm_1 <- gbm(logSalePrice~.,data=train_db[-sample_train,var_to_keep],n.minobsinnode = n.minobsinnode,distribution="gaussian",
                   interaction.depth=depth,n.trees = n.trees,train.fraction = .7,shrinkage = shrinkage,bag.fraction = bag.fraction,verbose = F)
  var_imp <- summary(gbm_1)
  selected_var <- gsub(pattern = "`",replacement = "",c(as.character(var_imp[var_imp$rel.inf>.1,]$var)))
  ##########################
    ####    TRY GLMNET    ####
    ##########################
  offset=predict(gbm_1,train_db,n.trees = which.min(gbm_1$valid.error))
  var_to_keep_postGBM=colnames(db_matrix)[(unlist(sapply(selected_var,FUN = function(x)grep(pattern = x,x = colnames(db_matrix)))))]
  train_matrix=db_matrix[db_matrix[,train_id]==1,var_to_keep_postGBM][-sample_train,] #CAREFUL rows with NAs are removed https://stackoverflow.com/questions/6447708/model-matrix-generates-fewer-rows-than-original-data-frame
  train_matrix
  dim(train_matrix)
  target=train_db[-sample_train,]$logSalePrice
  length(target)
  glm_1 <- glmnet(x=train_matrix,y = target,family="gaussian",alpha=0.5,standardize=F,nlambda = 100,offset=offset[-sample_train])

  valid_matrix=db_matrix[db_matrix[,train_id]==1,var_to_keep_postGBM][sample_train,]  #CAREFUL rows with NAs are removed https://stackoverflow.com/questions/6447708/model-matrix-generates-fewer-rows-than-original-data-frame
  test_matrix=db_matrix[db_matrix[,train_id]==0,var_to_keep_postGBM]
  ##########################
    ####    TRY XGBOOST   ####
    ##########################
    
      #   param <- list(max_depth = 6, eta = .01,subsample=.4,colsample_bytree=.5,lambda=.1,lambda_bias=.1,alpha=.1)
  #   data_samp <- data.matrix(train_db[-sample_train,setdiff(names(train_db),c("SalePrice","logSalePrice"))])
  #   label_samp <- train_db[-sample_train,"logSalePrice"]
  #   valid_samp <- sample(1:nrow(data_samp),round(.3*nrow(data_samp)))
  #   dtrain <- xgb.DMatrix(data_samp[-valid_samp,], label=label_samp[-valid_samp])
  #   watchlist <- list(eval=xgb.DMatrix(data_samp[valid_samp,],label=label_samp[valid_samp]),train=dtrain)
  # gbm_1 <- xgb.train(data = dtrain,
  #                  verbose = 1,early_stopping_rounds = 50,params = param,nrounds = 10000,watchlist)
  pred_1 <- predict(gbm_1,train_db[sample_train,setdiff(names(train_db),"SalePrice")],n.trees = which.min(gbm_1$valid.error))
  pred_2 <- predict(glm_1,valid_matrix,newoffset = offset[sample_train])
  obs_1 <- train_db[sample_train,"logSalePrice"]
  
  nm_gbm=paste0("pred_gbm_",i)
  gbm_predict_test=predict(gbm_1,test_db,n.trees = which.min(gbm_1$valid.error))
  predictions_table=cbind(predictions_table,x=gbm_predict_test)
  setnames(predictions_table,"x",nm_gbm)
  
  nm_glm=paste0("pred_glm_",i)
  lambda_predictions=data.frame(x=predict(glm_1,newx = valid_matrix,newoffset = offset[sample_train]))
  compute_RMSE <- function(x){
    sqrt(sum((x-obs_1)^2))
  }
  sapply(lambda_predictions,compute_RMSE)
  
  predictions_table=cbind(predictions_table,data.frame(x=predict(glm_1,newx = test_matrix,s = tail(glm_1$lambda,1),newoffset = gbm_predict_test)))
  setnames(predictions_table,"X1",nm_glm)
  
  
  SS_gbm=c(SS_gbm,(pred_1-obs_1)^2)
  SS_glm=c(SS_glm,(pred_2-obs_1)^2)
  
  size=c(size,length(sample_train))
}
RMSE_glm=sqrt(sum(SS_glm)/sum(size))
RMSE_gbm=sqrt(sum(SS_gbm)/sum(size))

iter=c(depth=depth,shrinkage=shrinkage,bag.fraction=bag.fraction,n.minobsinnode=n.minobsinnode,n.trees=n.trees,nb_split=nb_split,RMSE_glm=RMSE_glm,RMSE_gbm=RMSE_gbm)
print(iter)

return(predictions_table)

}

# results <- do.call(cbind,results)
save(list = "results",file = "results.RData")
predictors <- grep(pattern = "pred",x = names(results))
prediction <- rowSums(results[,predictors])/length(predictors)

plot(prediction,ylim=c(11,13))

test_db$predict=exp(prediction)

plot(test_db$predict,ylim=c(10,12))

to_submit=test_db[,c("Id","predict")]
setnames(to_submit,"predict","SalePrice")
write_csv(to_submit,path = "submit_GBM10000GLM_offset.csv")

stopCluster(cl)
