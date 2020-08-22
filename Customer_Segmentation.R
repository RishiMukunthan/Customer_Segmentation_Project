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
#write.csv(missing_val, "Missing_perc_R.csv", row.names = F)

data_output = data

data = subset(data, select = -c(CUST_ID))
# kNN Imputation
library(DMwR)
data_imputed = knnImputation(data, k = 3)

sum(is.na(data))
sum(is.na(data_imputed))

data_imputed$MA_PURCHASE = data_imputed$PURCHASES / data_imputed$TENURE
data_imputed$MA_CASH_ADVANCE = data_imputed$CASH_ADVANCE / data_imputed$TENURE
data_imputed$LIMIT_USAGE = data_imputed$BALANCE / data_imputed$CREDIT_LIMIT
data_imputed$PAY_MINPAY_RATIO = data_imputed$PAYMENTS / data_imputed$MINIMUM_PAYMENTS

x = data_imputed$ONEOFF_PURCHASES
y = data_imputed$INSTALLMENTS_PURCHASES
w = ifelse((x == 0) & (y == 0), "NONE", ifelse((x > 0) & (y > 0), "BOTH",
    ifelse((x > 0) & (y == 0), "ONE_OFF", ifelse((x == 0) & (y > 0),"INSTALLMENT",""))))

data_imputed$PURCHASE_TYPE = w
table(data_imputed$PURCHASE_TYPE)


# dummify the data
library(caret)
dmy <- dummyVars(" ~ .", data = data_imputed)
data_encoded <- data.frame(predict(dmy, newdata = data_imputed))

# data with KPIs
data_kpi = subset(data_encoded, select = -c(PURCHASES,CASH_ADVANCE,
                                            TENURE,BALANCE,CREDIT_LIMIT,PURCHASE_TYPEBOTH,
                                            PAYMENTS,MINIMUM_PAYMENTS))

# data with KPIs
data_kpi_original = subset(data_encoded, select = -c(PURCHASES,CASH_ADVANCE,
                                            TENURE,BALANCE,CREDIT_LIMIT,PURCHASE_TYPEBOTH,
                                            PAYMENTS,MINIMUM_PAYMENTS))

rm(dmy,missing_val,w,x,y)

#Log transformation
data_kpi$BALANCE_FREQUENCY=log(data_kpi$BALANCE_FREQUENCY+1)
data_kpi$ONEOFF_PURCHASES=log(data_kpi$ONEOFF_PURCHASES+1)
data_kpi$INSTALLMENTS_PURCHASES=log(data_kpi$INSTALLMENTS_PURCHASES+1)
data_kpi$PURCHASES_FREQUENCY=log(data_kpi$PURCHASES_FREQUENCY+1)
data_kpi$ONEOFF_PURCHASES_FREQUENCY=log(data_kpi$ONEOFF_PURCHASES_FREQUENCY+1)
data_kpi$PURCHASES_INSTALLMENTS_FREQUENCY=log(data_kpi$PURCHASES_INSTALLMENTS_FREQUENCY+1)
data_kpi$CASH_ADVANCE_FREQUENCY=log(data_kpi$CASH_ADVANCE_FREQUENCY+1)
data_kpi$CASH_ADVANCE_TRX=log(data_kpi$CASH_ADVANCE_TRX+1)
data_kpi$PURCHASES_TRX=log(data_kpi$PURCHASES_TRX+1)
data_kpi$PRC_FULL_PAYMENT=log(data_kpi$PRC_FULL_PAYMENT+1)
data_kpi$MA_PURCHASE=log(data_kpi$MA_PURCHASE+1)
data_kpi$MA_CASH_ADVANCE=log(data_kpi$MA_CASH_ADVANCE+1)
data_kpi$LIMIT_USAGE=log(data_kpi$LIMIT_USAGE+1)
data_kpi$PAY_MINPAY_RATIO=log(data_kpi$PAY_MINPAY_RATIO+1)

data_scaled = data_kpi
#Normalization
for(i in colnames(data_kpi)){
  print(i)
  data_scaled[,i] = (data_kpi[,i] - min(data_kpi[,i]))/
    (max(data_kpi[,i] - min(data_kpi[,i])))
}
#PCA
# Applying PCA
# install.packages('caret')

# install.packages('e1071')
library(caret)
library(e1071)
pca = preProcess(x = data_scaled, method = 'pca', pcaComp = 4)

data_pca = predict(pca, data_scaled)

# Fitting K-Means to the dataset
set.seed(0)
kmeans = kmeans(x = data_pca, centers = 4)
res_kmeans = kmeans$cluster

pdf("Customer Segmentation R Plots.pdf")
# Visualising the clusters

library(factoextra)
fviz_cluster(kmeans, data_pca[, 1:2], ellipse.type = "norm")
fviz_cluster(kmeans, data_scaled[, 2:3], ellipse.type = "norm")

data_output$Group = res_kmeans
data_imputed$Group = res_kmeans
data_kpi_original$Group = res_kmeans

library(dplyr)
data_grouped <- group_by(data_kpi_original, Group)
data_summary = summarise(data_grouped, MA_CASH_ADVANCE = mean(MA_CASH_ADVANCE),
          MA_PURCHASE = mean(MA_PURCHASE), PURCHASES_TRX=mean(PURCHASES_TRX),
          LIMIT_USAGE = mean(LIMIT_USAGE), PAY_MINPAY_RATIO = mean(PAY_MINPAY_RATIO),
          PRC_FULL_PAYMENT = mean(PRC_FULL_PAYMENT), ONEOFF_PURCHASES = mean(ONEOFF_PURCHASES),
          INSTALLMENTS_PURCHASES = mean(INSTALLMENTS_PURCHASES))
#BarPlot
Cluster <- data_summary$Group
Cash_Advance <- data_summary$MA_CASH_ADVANCE
Monthly_Purchase <- data_summary$MA_PURCHASE
Limit_Used <- data_summary$LIMIT_USAGE
PurchasePerTransaction <- data_summary$PURCHASES_TRX
Pay_MinPay_R <- data_summary$PAY_MINPAY_RATIO
Full_Due_Paid <- data_summary$PRC_FULL_PAYMENT
Oneoff <- data_summary$ONEOFF_PURCHASES
Installments <- data_summary$INSTALLMENTS_PURCHASES

# Plot the bar chart 
barplot(Cash_Advance,names.arg=Cluster,xlab="Cluster",ylab="Cash_Advance",col="blue",
        main="Cash_Advance",border="red")

barplot(Limit_Used,names.arg=Cluster,xlab="Cluster",ylab="Limit_Used",col="blue",
        main="Limit_Used",border="red")

barplot(Monthly_Purchase,names.arg=Cluster,xlab="Cluster",ylab="Monthly_Purchase",col="blue",
        main="Monthly_Purchase",border="red")

barplot(PurchasePerTransaction,names.arg=Cluster,xlab="Cluster",ylab="PurchasePerTransaction",col="blue",
        main="PurchasePerTransaction",border="red")

barplot(Pay_MinPay_R,names.arg=Cluster,xlab="Cluster",ylab="Pay_MinPay_R",col="blue",
        main="Pay_MinPay_R",border="red")

barplot(Full_Due_Paid,names.arg=Cluster,xlab="Cluster",ylab="Full_Due_Paid",col="blue",
        main="Full_Due_Paid",border="red")

#x11()
Purchase_Types <- rbind(Oneoff,Installments)

barplot(Purchase_Types,beside=T,names = c("C1", "C2","C3","C4"),
        xlab="Cluster",ylab="Purchases",col = c("red", "green"),
        main="Purchase Type",border="red",
        legend = c("Oneoff", "Installments"))

data_output$Group[data_output$Group %in% "1"] = "Withdrawers"
data_output$Group[data_output$Group %in% "2"] = "Installment_Purchasers"
data_output$Group[data_output$Group %in% "3"] = "One_Off_Purchasers"
data_output$Group[data_output$Group %in% "4"] = "Big_Spenders"

# Writing a csv (output)
write.csv(data_output, "Clustering_Output_R.csv", row.names = F)
dev.off()