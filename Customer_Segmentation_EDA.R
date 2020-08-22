rm(list=ls(all=T))

setwd("C:/Users/RISHI MUKUNTHAN/Desktop/Data Science/Projects/Edwisor Customer Segmentation/Final/R")

getwd()

#Load Libraries with help of lapply function
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')

lapply(x, require, character.only = TRUE)
rm(x)

#Load data into data object(data frame)
data = read.csv("credit-card-data.csv", header = T, na.strings = c(" ", "", "NA"))

#Structure of data
str(data)
#Column Names
colnames(data)
#view head
head(data,5)
#View stats of each column
summary(data)

##################################Missing Values Analysis###############################################
#Don't impute if missing percent more than 30
#Create data frame with missing value count in each column
missing_val = data.frame(apply(data,2,function(x){sum(is.na(x))}))
#New column with column name so that we can drop index
missing_val$Columns = row.names(missing_val)
#Renaming count column
names(missing_val)[1] =  "Missing_percentage"
#Convert to percent
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(data)) * 100
#Descending order
missing_val = missing_val[order(-missing_val$Missing_percentage),]
#row name index change
row.names(missing_val) = NULL
#interchange column order
missing_val = missing_val[,c(2,1)]
#Save to csv
write.csv(missing_val, "Missing_perc_R.csv", row.names = F)

#install.packages("tidyverse")
#install.packages("funModeling")
#install.packages("Hmisc")

library(funModeling) 
library(tidyverse) 
library(Hmisc)

pdf("EDA_R.pdf")  

plot_num(data)
describe(data)

############################################Outlier Analysis#############################################
# ## BoxPlots

boxplotshow = function(data){
  for(i in 2:ncol(data)){
    boxplot(data[i],main = colnames(data[i]))
  }
}

boxplotshow(data)
dev.off()