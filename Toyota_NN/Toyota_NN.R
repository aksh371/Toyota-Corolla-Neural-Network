setwd('C:\\Users\\aksha\\Documents\\BUAN 6356')
library(rpart)
library(rpart.plot)
library('caret')
library('dplyr')
library(neuralnet)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#11.3
library('caret')
library('dplyr')
library('neuralnet')
setwd('C:\\Users\\aksha\\Documents\\BUAN 6356')

car<-read.csv('ToyotaCorolla.csv')
head(car)

#Data Preprocessing
#selecting relevant columns
car<-select(car,c('Age_08_04', 'KM','Fuel_Type','HP','Automatic','Doors','Quarterly_Tax','Mfr_Guarantee','Guarantee_Period','Airco','Automatic_airco'
                  ,'CD_Player','Powered_Windows','Sport_Model','Tow_Bar','Price'))

head(car)

#selecting numerical columns for normalization
car_num<-select(car,c('Age_08_04','KM','HP','Quarterly_Tax','Doors','Guarantee_Period','Price'))
car_dummy<-select(car,c('Fuel_Type'))
car_cat<-select(car,c('Automatic','Mfr_Guarantee','Airco','Automatic_airco','CD_Player','Powered_Windows','Sport_Model','Tow_Bar'))

min_max_norm <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}
#normalizing the numerical columns
car_num <- as.data.frame(lapply(car_num, min_max_norm))

head(car_num)

#creating dummy variables for the fuel type column
library('fastDummies')

car_dummy<-dummy_cols(car_dummy)
head(car_dummy)

car_dummy<-select(car_dummy,-'Fuel_Type')
head(car_dummy)

car_new<-data.frame(car_num,car_dummy,car_cat)
head(car_new)


#partitioning data into training and validation data
set.seed(1234)
rows_train_car <- sample(rownames(car_new), dim(car_new)[1]*0.6)
rows_valid_car <- sample(setdiff(rownames(car_new), rows_train_car),
                         dim(car_new)[1]*0.4)
train_df_car<-car_new[rows_train_car,];train_df_car

valid_df_car<-car_new[rows_valid_car,];valid_df_car

head(train_df_car)

head(valid_df_car)

#building nn model

#single hidden layer with 2 nodes
car_nn_single <- neuralnet(Price ~ .,data =train_df_car, hidden = 2, linear.output = T,lifesign = 'full',threshold = 0.5)
plot(car_nn_single)

#predicting data using model on validation data
pred_price_single = compute(car_nn_single,valid_df_car,rep = 1)
pred_price_single$net.result
multiplier <-max(car$Price) - min(car$Price);multiplier
predicted_price_scaled_single <- pred_price_single$net.result * multiplier + min(car$Price);predicted_price_scaled_single
valid_car_price <- as.data.frame((valid_df_car$Price)*multiplier + min(car$Price));valid_car_price
df_val_met_single<-data.frame(predicted_price_scaled_single,valid_car_price)
df_val_met_single


head(df_val_met_single)
err_single<-df_val_met_single$X.valid_df_car.Price....multiplier...min.car.Price.-df_val_met_single$predicted_price_scaled_single
df_val_met_single<-data.frame(err_single,df_val_met_single)
head(df_val_met_single)
squared_error_single<-(df_val_met_single$err_single)^2;squared_error_single

df_val_met_single<-data.frame(squared_error_single,df_val_met_single)
head(df_val_met_single)
mean(df_val_met_single$squared_error_single)
RMSE1_valid_single = sqrt(mean(df_val_met_single$squared_error_single));RMSE1_valid_single;RMSE1_valid_single

#training data RMSE
pred_price_train_single = compute(car_nn_single,train_df_car)
pred_price_train_single$net.result

pred_price_train_scaled_single = pred_price_train_single$net.result*multiplier + min(car$Price)

train_car_price <- as.data.frame((train_df_car$Price)*multiplier + min(car$Price));train_car_price

data.frame(pred_price_train_scaled_single,train_car_price)
df_train_met_single<-data.frame(pred_price_train_scaled_single,train_car_price)
head(df_train_met_single)
err_single_train<-df_train_met_single$X.train_df_car.Price....multiplier...min.car.Price. - df_train_met_single$pred_price_train_scaled_single
df_train_met_single<-data.frame(err_single_train,df_train_met_single)
head(df_train_met_single)
squared_error_train_single<-(df_train_met_single$err_single_train)^2;squared_error_train_single
df_train_met_single<-data.frame(squared_error_train_single,df_train_met_single);df_train_met_single
mean(df_train_met_single$squared_error_train_single)
RMSE1_train_single = sqrt(mean(df_train_met_single$squared_error_train_single));RMSE1_train_single



#single hidden layer with 5 nodes
car_nn <- neuralnet(Price ~ .,data =train_df_car, hidden = 5, linear.output = T,lifesign = 'full',threshold = 0.5)
plot(car_nn)

#predicting data using model on validation data
pred_price = compute(car_nn,valid_df_car,rep = 1)
pred_price$net.result

#scaling back data
multiplier <-max(car$Price) - min(car$Price);multiplier

predicted_price_scaled <- pred_price$net.result * multiplier + min(car$Price)
predicted_price_scaled

valid_car_price <- as.data.frame((valid_df_car$Price)*multiplier + min(car$Price));valid_car_price

head(valid_df_car$Price)

df_val_met1<-data.frame(predicted_price_scaled,valid_car_price)

head(df_val_met1)
err1<-df_val_met1$X.valid_df_car.Price....multiplier...min.car.Price.-df_val_met1$predicted_price_scaled
df_val_met1<-data.frame(err1,df_val_met1)
head(df_val_met1)
squared_error1<-(df_val_met1$err)^2;squared_error1

df_val_met1<-data.frame(squared_error1,df_val_met1)
head(df_val_met1)
mean(df_val_met1$squared_error1)
RMSE1_valid_1 = sqrt(mean(df_val_met1$squared_error1));RMSE1_valid_1 

#predicting data using model on training data
pred_price_train = compute(car_nn,train_df_car)
pred_price_train$net.result

pred_price_train_scaled = pred_price_train$net.result*multiplier + min(car$Price)

train_car_price <- as.data.frame((train_df_car$Price)*multiplier + min(car$Price));valid_car_price

data.frame(pred_price_train_scaled,train_car_price)
df_train_met1<-data.frame(pred_price_train_scaled,train_car_price)
head(df_train_met1)
err2<-df_train_met1$X.train_df_car.Price....multiplier...min.car.Price.-df_train_met1$pred_price_train_scaled
df_train_met1<-data.frame(err2,df_train_met1)
head(df_train_met1)
squared_error2<-(df_train_met1$err2)^2;squared_error2
df_train_met1<-data.frame(squared_error2,df_train_met1);df_train_met1
mean(df_train_met1$squared_error2)
RMSE1_train_1 = sqrt(mean(df_train_met1$squared_error2));RMSE1_train_1 


#double layer 5 nodes
#validation data
car_nn2 <- neuralnet(Price ~ .,data =train_df_car, hidden = c(5,5), linear.output = T,lifesign = 'full',threshold = 0.5)
plot(car_nn2)

#predicting data using model on validation data
pred_price_2 = compute(car_nn2,valid_df_car,rep = 1)
pred_price_2$net.result

predicted_price_scaled_2 <- pred_price_2$net.result * multiplier + min(car$Price)

data.frame(predicted_price_scaled_2,valid_car_price)
df_val_met2<-data.frame(predicted_price_scaled_2,valid_car_price)
head(df_val_met2)
err3<-df_val_met2$X.valid_df_car.Price....multiplier...min.car.Price.-df_val_met2$predicted_price_scaled_2
df_val_met2<-data.frame(err3,df_val_met2)
head(df_val_met2)
squared_error3<-(df_val_met2$err3)^2;squared_error3
df_val_met2<-data.frame(squared_error3,df_val_met2);df_val_met2
mean(df_val_met2$squared_error3)
RMSE1_val_2 = sqrt(mean(df_val_met2$squared_error3));RMSE1_val_2 



#predicting data using model on training data
pred_price_2_train = compute(car_nn2,train_df_car,rep = 1)
pred_price_2_train$net.result

predicted_price_2_train_scaled <- pred_price_2_train$net.result * multiplier + min(car$Price);predicted_price_2_train_scaled
data.frame(predicted_price_2_train_scaled,train_car_price)

data.frame(predicted_price_2_train_scaled,train_car_price)
df_train_met2<-data.frame(predicted_price_2_train_scaled,train_car_price)
head(df_train_met2)
err4<-df_train_met2$X.train_df_car.Price....multiplier...min.car.Price.- df_train_met2$predicted_price_2_train_scaled
df_train_met2<-data.frame(err4,df_train_met2)
head(df_train_met2)
squared_error4<-(df_train_met2$err4)^2;squared_error4
df_train_met2<-data.frame(squared_error4,df_train_met2);df_train_met2
mean(df_train_met2$squared_error4)
RMSE1_train_2 = sqrt(mean(df_train_met2$squared_error4));RMSE1_train_2 

#RMSE values
#single layer with 2 nodes
RMSE1_valid_single
RMSE1_train_single


#single layer 5 nodes
RMSE1_valid_1 
RMSE1_train_1

#double layer 5 nodes
RMSE1_val_2 
RMSE1_train_2 
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#c.comparing the RMSE values
#Regression Trees
rmse_train
rmse_valid


#Neural networks
#single layer with 2 nodes
RMSE1_valid_single
RMSE1_train_single


#single layer 5 nodes
RMSE1_valid_1 
RMSE1_train_1

#double layer 5 nodes
RMSE1_val_2 
RMSE1_train_2 


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
