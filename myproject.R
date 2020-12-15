# https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
# https://www.kaggle.com/sakshigoyal7/credit-card-customers/tasks

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(devtools)) install.packages("devtools", repos = "http://cran.us.r-project.org")
if(!require(regtools)) install_github("matloff/regtools")
if(!require(dummies)) install.packages("dummies", repos = "http://cran.us.r-project.org")
if(!require(partykit)) install.packages("partykit", repos = "http://cran.us.r-project.org")

############# regtools package - kNN() function ####

######################### read in the data ###############################################

#https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/

abalone <- read.csv("abalone.data", header = FALSE, sep = ",")
colnames(abalone) <- c("sex", "length", "diameter", "height", "whole_weight",
                       "shucked_weight", "viscera_weight", "shell_weight",
                       "total_rings")

######################## look at the data ###############################################
# look at the data type of the variables
glimpse(abalone)

# now let's look at the first 10 items of the dataset
abalone %>% as_tibble()

# then look at the distribution of the classes in the outcome vector
table(abalone$total_rings)

# and again in a histogram
hist(abalone$total_rings, xlab = "total rings")

# lastly, (certainly not least), check for NAs
sum(is.na(abalone))

# Look to see how correlated the dimesnions are.
cor(abalone[2:9])

#################### a simple model ###################################################

# Now assume we foraged an abalone along the False Bay coast in South Africa with
# shell weight 497 grams (0.497kg), and we want to predict its age. How would we proceed?
# We'd most likely look at what the total rings are for a few abalone, in our dataset,
# with shell weight closest to 497 grams and calculate the average number of rings
# of the selected items.

#Let's look at the 5 'nearest' shell_weights and calculate the average of the selected items.

options(scipen=999)
shell <- abalone$shell_weight
dists <- abs(shell - 0.497) # distances of shell_weight closest to 0.497
close5 <- order(dists)[1:5]

# The 5 closest distances to shell_weight of 0.497 are:
dists[close5]

# and the 5 corresponding closest shell_weights are:
abalone$shell_weight[close5]

# the total_rings for each of these 5 closest items are:
abalone$total_rings[close5]

# and the mean total_rings for the 5 closest shell weights is:
mean(abalone$total_rings[close5])

############################# regtools kNN() ############################################

###################### change character to factor ######################################

# Now let's predict total_rings with the regtools package function kNN() again
# assuming we foraged an abalone in the shallows and recorded all the features' measurements.
abalone <- abalone %>% mutate_if(is.character, as.factor)

############################# dummify the set ##########################################

aBalone <- factorsToDummies(abalone, omitLast = TRUE) # factorsToDummies()
# coerces the data.frame to a matrix array

############################ look at the dummified dataset ##############################

# Look at the column names after converting the 'sex' feature to dummy.
colnames(aBalone)

# Observe that the number of variables has increased from 9 to 10.
dim(aBalone)

####################### creat training set #############################################
# Create the X matrix to be the training set.
abalone.x <- aBalone[, 1:9]

# And the Y vector for the training set.
age <- aBalone[, 10]

################################ kNN() #################################################

# Now let's begin using kNN() by predicting the rings for one abalone (some random abalone).
knnout <- kNN(abalone.x, age, c(0, 0, 0.35, 0.39, 0.09, 0.46, 0.2, 0.11, 0.12), 5)

# Here we see the 5 positions (row numbers) of the predictions.
knnout$whichClosest

# The output below shows us the average of the 5 predictions. It is the regression estimate.
knnout$regests

# Below we access the total_rings that corresponds with the indexes shown above.
aBalone[c(2205, 3154, 1, 12, 3837), 10]

# And confirm the average.
mean(aBalone[c(2205, 3154, 1, 12, 3837), 10])

################### predict using original dataset #####################################

# Let's continue to demonstrate usage of the kNN() function by predicting the
# rings for the original dataset. To begin, let's set **k** = 5.
knnout <- kNNallK(abalone.x, age, abalone.x, 5, allK = TRUE)

# Here we see the structure of the new object which is a list.
str(knnout)

# The matrix below shows that we generate 60 sets of 4177 predictions.
# Each row has 60 nearest neighbour predictions. (Indicates which row we will find the
# closest estimate). Below we show the nearest neigbour predictions for only the first
# 3 rows of the prediction dataset.

knnout$whichClosest[1:3, 1:5]

# See the predicted values (Y) using the original data. These are the regression estimates.
# Row 1 has all the predicted values for k=1, row 2 shows the predicted values for k = 2, etc.
knnout$regests[1:3, 1:5]

# The first 5 real Y values are shown below
age[1:5] # the predicted values for k=1 is the same as this

# Exclude k = 1
# Here we run the function but exclude k = 1
knnout <- kNNallK(abalone.x, age, abalone.x, 60, allK = TRUE, leave1out = TRUE)

# Now see the predicted row positions for first 3 items.
knnout$whichClosest[1:3, 1:60]

# An see the regression estimates for the first 3 items
knnout$regests[1:3, 1:60]

# Below see the loss error for each k value. The Mean Absolute Prediction Error (MAPE)
findOverallLoss(knnout$regests, age)

# The optimal k is:
which.min(findOverallLoss(knnout$regests, age))

# And the minimum loss error is:
min(findOverallLoss(knnout$regests, age))

############### partykit package ctree() function #####################################
library(partykit)

#################### First dummify the data ###########################################
dmy <- dummyVars("~ .", data = abalone)

############################### fit the model #########################################

# do the fit with hyperparameter maxdepth set to 3
abalone <- data.frame(predict(dmy, newdata = abalone))
abba <- ctree(total_rings ~.,
              data = abalone,
              control = ctree_control(maxdepth = 3))

############ take a look at the flow chart ##########################################
plot(abba)

# Nodes 4, 5, 7, 8, 11, 12, 14, 15 are terminal nodes.
# They do not split. We can access these programmatically as seen below.
nodeids(abba, terminal = TRUE)

########################## Printed version of the output #############################
abba

# First see the node terminals.
nodeIDs <- nodeids(abba, terminal = TRUE)

# Then the median Y for each node.
mdn <- function(yvals, weights) median(yvals)
predict_party(abba, id=nodeIDs, FUN = mdn)

# And finally the mean Y for each node
predict_party(abba, id=nodeIDs, FUN = function(yvals, weights) mean(yvals))

# The Y values in a given node. Select the node number and insert in the id argument.
f1 <- function(yvals, weights) c(yvals)
predict_party(abba, id=4, FUN = f1)

# The median of the Y values of a specific node.
mdn <- function(yvals, weights) median(yvals)
predict_party(abba, id=15, FUN = mdn)

########### Here's how to predict the total_rings for a recently retrieved abalone #########

newx <- data.frame(sex.I=0,sex.M=1,length=0.495,diameter=0.503,height=0.137,
                   whole_weight=0.5347, shucked_weight=0.372, viscera_weight=0.301,
                   shell_weight=0.267)
predict(abba,newx,FUN=mdn)

############################################################################################
