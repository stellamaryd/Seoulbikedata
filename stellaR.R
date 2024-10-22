#set working directory
setwd(dirname(file.choose()))
getwd()

SeoulBikeDataset.dis <- read.csv("SeoulBikeDataset.csv", stringsAsFactors = FALSE, fileEncoding = "UTF-8")
str(SeoulBikeDataset.dis)

# Read the dataset
SeoulBikeDataset.dis <- read.csv("SeoulBikeDataset.csv", stringsAsFactors = FALSE, fileEncoding = "UTF-8")

# Check the structure of the dataset
str(SeoulBikeDataset.dis)

# Assuming you want to create a boxplot for the 'Rented.Bike.Count' variable
# Replace 'Rented.Bike.Count' with the actual variable name if different

# Create the boxplot
boxplot(SeoulBikeData.dis$Rented.Bike.Count, 
        main = "Boxplot of Rented Bike Count",
        ylab = "Rented Bike Count",
        col = "skyblue",
        border = "black")



# Read the dataset
SeoulBikeDataset.dis <- read.csv("SeoulBikeDataset.csv", stringsAsFactors = FALSE, fileEncoding = "UTF-8")

# Check the structure of the dataset
str(SeoulBikeDataset.dis)

# Assuming you want to create a boxplot for the 'Rented.Bike.Count' variable
# Replace 'Rented.Bike.Count' with the actual variable name if different

# Create the boxplot
boxplot(SeoulBikeDataset.dis$Rented.Bike.Count, 
        main = "Boxplot of Rented Bike Count",
        ylab = "Rented Bike Count",
        col = "skyblue",
        border = "black")
# Create the boxplot
boxplot(SeoulBikeDataset.dis$Rented.Bike.Count, 
        main = "Boxplot of Rented Bike Count",
        ylab = "Rented Bike Count",
        col = "skyblue",
        border = "black")



# Create the boxplot
boxplot(SeoulBikeDataset.dis$Rented.Bike.Count, 
        main = "Boxplot of Rented Bike Count",
        ylab = "Rented Bike Count",
        col = "skyblue",
        border = "black")



# Create the scatter plot
plot(SeoulBikeDataset.dis$Temperature.C., 
     SeoulBikeDataset.dis$Rented.Bike.Count,
     main = "Scatter Plot of Temperature vs. Rented Bike Count",
     xlab = "Temperature (C)",
     ylab = "Rented Bike Count",
     col = "blue")


plot(SeoulBikeDataset.dis$Humidity, 
     SeoulBikeDataset.dis$Rented.Bike.Count,
     main = "Scatter Plot of Humidity vs. Rented Bike Count",
     xlab = "Humidity",
     ylab = "Rented Bike Count",
     col = "blue")

# Selecting a subset of variables for pairwise scatter plots
subset_data <- SeoulBikeDataset.dis[, c("Rented.Bike.Count", "Temperature.C.", "Humidity", "Wind.speed..m.s.", "Visibility..10m.", "Dew.point.temperature.C.")]

# Creating pairwise scatter plots
pairs(subset_data, 
      main = "Pairwise Scatter Plots")


plot(SeoulBikeDataset.dis$Wind.speed, 
     SeoulBikeDataset.dis$Rented.Bike.Count,
     main = "Scatter Plot of Wind.speed vs. Rented Bike Count",
     xlab = "Wind.speed",
     ylab = "Rented Bike Count",
     col = "blue")


plot(SeoulBikeDataset.dis$Visibility..10m., 
     SeoulBikeDataset.dis$Rented.Bike.Count,
     main = "Scatter Plot of Visibility..10m. vs. Rented Bike Count",
     xlab = "Visibility..10m.",
     ylab = "Rented Bike Count",
     col = "blue")

plot(SeoulBikeDataset.dis$Dew.point.temperature.C.", 
     SeoulBikeDataset.dis$Rented.Bike.Count,
     main = "Scatter Plot of Dew.point.temperature.C." vs. Rented Bike Count",
     xlab = "Dew.point.temperature.C."",
     ylab = "Rented Bike Count",
     col = "blue")
     
     
     
     # Create the scatter plot
plot(SeoulBikeDataset.dis$Dew.point.temperature.C., 
     SeoulBikeDataset.dis$Rented.Bike.Count,
     main = "Scatter Plot of Dew point temperature vs. Rented Bike Count",
     xlab = "Dew point temperature (C)",
     ylab = "Rented Bike Count",
     col = "blue")
     
     
     # Define the number of columns for the grid layout
num_cols <- 3

# Calculate the number of rows needed based on the number of variables
num_vars <- ncol(SeoulBikeDataset.dis)
num_rows <- ceiling(num_vars / num_cols)

# Create histograms for all variables in the dataset
par(mfrow = c(num_rows, num_cols))

# Loop through each variable and create a histogram
for (i in 1:num_vars) {
  hist(SeoulBikeDataset.dis[, i], 
       main = paste("Histogram of", names(SeoulBikeDataset.dis)[i]), 
       xlab = names(SeoulBikeDataset.dis)[i],
       col = "skyblue")
}

# Reset the plotting layout to default
par(mfrow = c(1, 1))

# Create histograms for all variables in the dataset
par(mfrow = c(3, 5)) # Setting up multiple plots in a grid layout

# Loop through each variable and create a histogram
for (i in 1:ncol(SeoulBikeDataset.dis)) {
  hist(SeoulBikeDataset.dis[, i], 
       main = paste("Histogram of", names(SeoulBikeDataset.dis)[i])
       xlab = names(SeoulBikeDataset.dis)[i],
       col = "skyblue")
}

# Reset the plotting layout to default
par(mfrow = c(1, 1))





#Define the number of columns for the grid layout
    num_cols <- 3

# Calculate the number of rows needed based on the number of variables
num_vars <- ncol(SeoulBikeDataset.dis)
num_rows <- ceiling(num_vars / num_cols)

# Create histograms for all variables in the dataset
par(mfrow = c(num_rows, num_cols))

# Loop through each variable and create a histogram
for (i in 1:num_vars) {
  hist(SeoulBikeDataset.dis[, i], 
       main = paste("Histogram of", names(SeoulBikeDataset.dis)[i]), 
       xlab = names(SeoulBikeDataset.dis)[i],
       col = "skyblue")
}

# Reset the plotting layout to default
par(mfrow = c(1, 1))


# Filter numeric variables
numeric_vars <- SeoulBikeDataset.dis[sapply(SeoulBikeDataset.dis, is.numeric)]

# Define the number of columns for the grid layout
num_cols <- 3

# Calculate the number of rows needed based on the number of variables
num_vars <- ncol(numeric_vars)
num_rows <- ceiling(num_vars / num_cols)

# Create histograms for all numeric variables in the dataset
par(mfrow = c(num_rows, num_cols))

# Loop through each numeric variable and create a histogram
for (i in 1:num_vars) {
  hist(numeric_vars[, i], 
       main = paste("Histogram of", names(numeric_vars)[i]), 
       xlab = names(numeric_vars)[i],
       col = "skyblue")
}

# Reset the plotting layout to default
par(mfrow = c(1, 1))




# Reduce the size of individual plots
par(mfrow = c(num_rows, num_cols), mar = c(4, 4, 2, 1))

# Loop through each numeric variable and create a histogram
for (i in 1:num_vars) {
  hist(numeric_vars[, i], 
       main = paste("Histogram of", names(numeric_vars)[i]), 
       xlab = names(numeric_vars)[i],
       col = "skyblue")
}

# Reset the plotting layout to default
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1)















# Define the number of columns for the grid layout
num_cols <- 3

# Calculate the number of rows needed based on the number of variables
num_vars <- ncol(numeric_vars)
num_rows <- ceiling(num_vars / num_cols)

# Create boxplots for all numeric variables in the dataset
par(mfrow = c(num_rows, num_cols))

# Loop through each numeric variable and create a boxplot
for (i in 1:num_vars) {
  boxplot(numeric_vars[, i], 
          main = paste("Boxplot of", names(numeric_vars)[i]), 
          ylab = names(numeric_vars)[i],
          col = "skyblue")
}



# Reduce the size of individual plots
par(mfrow = c(num_rows, num_cols), mar = c(4, 4, 2, 1))

# Loop through each numeric variable and create a boxplot
for (i in 1:num_vars) {
  boxplot(numeric_vars[, i], 
          main = paste("Boxplot of", names(numeric_vars)[i]), 
          ylab = names(numeric_vars)[i],
          col = "skyblue")
}

# Reset the plotting layout to default
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1)


# Reset the plotting layout to default
par(mfrow = c(1, 1))





# Reduce the size of individual plots
par(mfrow = c(num_rows, num_cols), mar = c(4, 4, 2, 1))

# Loop through each numeric variable and create a boxplot
for (i in 1:num_vars) {
  boxplot(numeric_vars[, i], 
          main = paste("Boxplot of", names(numeric_vars)[i]), 
          ylab = names(numeric_vars)[i],
          col = "skyblue")
}

# Reset the plotting layout to default
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1)


# Perform correlation analysis
correlation_matrix <- cor(SeoulBikeDataset.dis)

# Print correlation matrix
print(correlation_matrix)

# Alternatively, you can visualize the correlation matrix using a heatmap
library(ggplot2)
library(reshape2)

# Convert the correlation matrix to a long format
correlation_data <- melt(correlation_matrix)

# Create a heatmap of the correlation matrix
ggplot(data = correlation_data, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "blue", high = "red") +
  theme_minimal() +
  labs(title = "Correlation Heatmap")
  
  
  # Filter numeric variables
numeric_vars <- SeoulBikeDataset.dis[sapply(SeoulBikeDataset.dis, is.numeric)]

# Perform correlation analysis
correlation_matrix <- cor(numeric_vars)

# Print correlation matrix
print(correlation_matrix)


# Install and load the psych package if not already installed
if (!requireNamespace("psych", quietly = TRUE)) {
  install.packages("psych")
}
library(psych)

# Perform factor analysis
factor_analysis <- fa(SeoulBikeDataset.dis)

# Print factor analysis results
print(factor_analysis)


# Filter numeric variables
numeric_vars <- SeoulBikeDataset.dis[sapply(SeoulBikeDataset.dis, is.numeric)]

# Perform factor analysis
factor_analysis <- fa(numeric_vars)

# Print factor analysis results
print(factor_analysis)




# Perform a two-sample t-test
holiday_group <- SeoulBikeDataset.dis$Rented.Bike.Count[SeoulBikeDataset.dis$Holiday == "Holiday"]
non_holiday_group <- SeoulBikeDataset.dis$Rented.Bike.Count[SeoulBikeDataset.dis$Holiday == "No Holiday"]

# Check if the variances are equal
var.equal <- var.test(holiday_group, non_holiday_group)$p.value > 0.05

# Perform the t-test
t_test_result <- t.test(holiday_group, non_holiday_group, var.equal = var.equal)

# Print the result
print(t_test_result)


# Load required libraries
library(ggplot2)

# Selecting a subset of variables for the scatterplot matrix
subset_data <- SeoulBikeDataset.dis[, c("Rented.Bike.Count", "Temperature.C.", "Humidity", "Wind.speed..m.s.", "Visibility..10m.", "Dew.point.temperature.C.")]

# Create scatterplot matrix
pairs(subset_data)



# Load the required library
library(rpart)

# Fit the decision tree regression model
tree_model <- rpart(Rented.Bike.Count ~ ., data = SeoulBikeDataset.dis)

# Print the summary of the model
summary(tree_model)

# Plot the decision tree
plot(tree_model)
text(tree_model, use.n = TRUE)


# Load the required library
library(rpart.plot)

# Plot the decision tree interactively
rpart.plot(tree_model)


# Install the rpart.plot package if not already installed
if (!requireNamespace("rpart.plot", quietly = TRUE)) {
  install.packages("rpart.plot")
}

# Load the required library
library(rpart.plot)



# Load the required library
library(rpart.plot)

# Plot the decision tree interactively
rpart.plot(tree_model)





# Fit Lasso regression model
lasso_model <- glmnet(X_train, y_train, alpha = 1)  # Lasso regression with alpha = 1
lasso_predictions <- predict(lasso_model, newx = X_test)

# Calculate Mean Absolute Error (MAE)
lasso_mae <- mean(abs(lasso_predictions - y_test))

# Print MAE
print(lasso_mae)















# Load the required library
library(glmnet)

# Prepare data
X <- SeoulBikeDataset.dis[, -c("Rented.Bike.Count")]  # Predictor variables
y <- SeoulBikeDataset.dis$Rented.Bike.Count  # Response variable

# Split data into train and test sets
set.seed(123)  # Set seed for reproducibility
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Fit Lasso regression model
lasso_model <- glmnet(X_train, y_train, alpha = 1)  # Lasso regression with alpha = 1

# Predict on test data
lasso_predictions <- predict(lasso_model, newx = X_test)

# Calculate Mean Absolute Error (MAE)
lasso_mae <- mean(abs(lasso_predictions - y_test))

# Print MAE
print(lasso_mae)












# Load the required library
library(rpart)

# Fit the decision tree regression model
tree_model <- rpart(Rented.Bike.Count ~ ., data = SeoulBikeDataset.dis)

# Print the summary of the model
summary(tree_model)

# Plot the decision tree
plot(tree_model)
text(tree_model, use.n = TRUE)




# Load the required library
library(glmnet)

# Prepare the data
X <- as.matrix(SeoulBikeDataset.dis[, -c("Rented.Bike.Count")])  # Predictor variables
y <- SeoulBikeDataset.dis$Rented.Bike.Count  # Response variable

# Fit Lasso regression model
lasso_model <- glmnet(X, y, alpha = 1)  # Set alpha = 1 for Lasso regression

# Print the summary of the model
print(lasso_model)

# Plot the coefficient trajectory
plot(lasso_model)




# Load the required library
library(rpart)

# Fit the decision tree regression model
tree_model <- rpart(Rented.Bike.Count ~ ., data = SeoulBikeDataset.dis)

# Print the summary of the model
summary(tree_model)

# Plot the decision tree
plot(tree_model)
text(tree_model, use.n = TRUE)


# Plot the scatter plot
plot(SeoulBikeDataset.dis$Dew.point.temperature.C., 
     SeoulBikeDataset.dis$Rented.Bike.Count,
     main = "Scatter Plot of Dew.point.temperature.C. vs. Rented Bike Count")
     
     
     
     # Load the required library
library(glmnet)

# Prepare data
X <- SeoulBikeDataset.dis[, -c("Rented.Bike.Count")]  # Predictor variables
y <- SeoulBikeDataset.dis$Rented.Bike.Count  # Response variable

# Split data into train and test sets
set.seed(123)  # Set seed for reproducibility
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Fit Lasso regression model
lasso_model <- glmnet(X_train, y_train, alpha = 1)  # Lasso regression with alpha = 1

# Predict on test data
lasso_predictions <- predict(lasso_model, newx = X_test)

# Calculate Mean Absolute Error (MAE)
lasso_mae <- mean(abs(lasso_predictions - y_test))

# Calculate Mean Squared Error (MSE)
lasso_mse <- mean((lasso_predictions - y_test)^2)

# Calculate Root Mean Squared Error (RMSE)
lasso_rmse <- sqrt(mean((lasso_predictions - y_test)^2))

# Print metrics
print(paste("Mean Absolute Error (MAE):", lasso_mae))
print(paste("Mean Squared Error (MSE):", lasso_mse))
print(paste("Root Mean Squared Error (RMSE):", lasso_rmse))

     
     
     
     
     
     
     
     
     
     
     
     
     # Load the required library
library(glmnet)

# Prepare the data
X <- as.matrix(SeoulBikeDataset.dis[, -c("Rented.Bike.Count")])  # Predictor variables
y <- SeoulBikeDataset.dis$Rented.Bike.Count  # Response variable

# Fit Lasso regression model
lasso_model <- glmnet(X, y, alpha = 1)  # Set alpha = 1 for Lasso regression

# Print the summary of the model
print(lasso_model)

# Plot the coefficient trajectory
plot(lasso_model)


