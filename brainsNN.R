# Load required libraries
library(png)
library(caret)
library(magick)
library(plot.matrix)

# Define hyperparameters and constants
VARIANCE <- 0.5  # Variance for weight initialization
IMAGE_HEIGHT <- 128  # Height of input images
IMAGE_WIDTH <- 128  # Width of input images
INPUTS <- IMAGE_HEIGHT * IMAGE_WIDTH  # Total number of input neurons
HIDDEN <- 12  # Number of hidden neurons
OUTPUTS <- 1  # Number of output neurons
LEARNINGRATE <- 0.2  # Learning rate

# Set random seed for reproducibility
set.seed(16)

# Initialize weights for hidden layer
hiddenWeights <- vector("list", HIDDEN)
for(i in seq_along(hiddenWeights)) {
    hiddenWeights[[i]] <- runif(INPUTS, -VARIANCE, VARIANCE)
}

# Initialize biases for hidden layer
hiddenBias <- numeric(HIDDEN)

# Initialize weights for output layer
outputWeights <- vector("list", HIDDEN)
for(i in seq_along(outputWeights)) {
    outputWeights[[i]] <- runif(HIDDEN, -VARIANCE, VARIANCE)
}

# Initialize biases for output layer
outputBias <- numeric(OUTPUTS)

# Define Sigmoid activation function
Sigmoid <- function(x) {
    1 / (1 + exp(-x))
}

# Define derivative of Sigmoid function
SigmoidPrime <- function(x) {
    x * (1 - x)
}

# Function to predict the output for given inputs
Predict <- function(inputs) {
    # Compute activations for the hidden layer
    hiddens <- numeric(HIDDEN)
    for(i in 1:HIDDEN) {
        hidden <- 0
        for(j in 1:INPUTS) {
            hidden <- hidden + (hiddenWeights[[i]][j] * inputs[j])
        }
        hiddens[i] <- Sigmoid(hidden + hiddenBias[i])
    }
    
    # Compute activations for the output layer
    outputs <- numeric(OUTPUTS)
    for(i in 1:OUTPUTS) {
        output <- 0
        for(j in 1:HIDDEN) {
            output <- output + (outputWeights[[i]][j] * hiddens[j])
        }
        outputs[i] <- Sigmoid(output + outputBias[i])
    }
    
    return(outputs)
}

# Function to update weights and biases through backpropagation
Learn <- function(inputs, targets) {
    # Compute activations for the hidden layer
    hiddens <- numeric(HIDDEN)
    for(i in 1:HIDDEN) {
        hidden <- 0
        for(j in 1:INPUTS) {
            hidden <- hidden + (hiddenWeights[[i]][j] * inputs[j])
        }
        hiddens[i] <- Sigmoid(hidden + hiddenBias[i])
    }
    
    # Compute activations for the output layer
    outputs <- numeric(OUTPUTS)
    for(i in 1:OUTPUTS) {
        output <- 0
        for(j in 1:HIDDEN) {
            output <- output + (outputWeights[[i]][j] * hiddens[j])
        }
        outputs[i] <- Sigmoid(output + outputBias[i])
    }
    
    # Calculate errors
    errors <- numeric(OUTPUTS)
    for(i in 1:OUTPUTS) {
        errors[i] <- targets - outputs[i]
    }
    
    # Compute derivatives of errors
    derrors <- numeric(OUTPUTS)
    for(i in 1:OUTPUTS) {
        derrors[i] <- errors[i] * SigmoidPrime(outputs[i])
    }
    
    # Backpropagate errors to hidden layer
    ds <- numeric(HIDDEN)
    for(i in 1:OUTPUTS) {
        for(j in 1:HIDDEN) {
            ds[j] <- ds[j] + (derrors[i] * outputWeights[[i]][j] * 
                                  SigmoidPrime(hiddens[j]))
        }
    }
    
    # Update output layer weights and biases
    for(i in 1:OUTPUTS) {
        for(j in 1:HIDDEN) {
            outputWeights[[i]][j] <<- outputWeights[[i]][j] + 
                (LEARNINGRATE * hiddens[j] * derrors[i])
        }
        outputBias[i] <<- outputBias[i] + (LEARNINGRATE * derrors[i])
    }
    
    # Update hidden layer weights and biases
    for(i in 1:HIDDEN) {
        for(j in 1:INPUTS) {
            hiddenWeights[[i]][j] <<- hiddenWeights[[i]][j] +
                (LEARNINGRATE * inputs[j] * ds[i])
        }
        hiddenBias[i] <<- hiddenBias[i] + (LEARNINGRATE * ds[i])
    }
}

# Folder containing Brain MRI images
trainfolderpath <- "./braintumor/Training/"
testfolderpath <- "./braintumor/Testing/"
brains <- c("notumor", "glioma", "meningioma", "pituitary")

# Function to load and preprocess images
LoadImages <- function(folderpath) {
    imageData <- list()
    for(brain in brains) {
        brainFiles <- list.files(paste0(folderpath, brain))
        for(i in seq_along(brainFiles)) {
            print(i)
            image <- image_read(paste0(folderpath, brain, "/", brainFiles[i]))
            image <- image_convert(image, type = "Grayscale")
            image <- image_scale(image, "128x128!")
            image <- as.integer(image[[1]])
            image <- image / 255
            imageData[[brainFiles[i]]] <- image
        }
    }
    return(imageData)
}

# Load and preprocess images
#trainImageData <- LoadImages(trainfolderpath)

# Load and preprocess images
#testImageData <- LoadImages(testfolderpath)

# Create labels (1 for circles, 0 for non-circles)
testlabels <- ifelse(startsWith(names(testImageData), "Te-no"), 1, 0)

# Create labels (1 for circles, 0 for non-circles)
trainlabels <- ifelse(startsWith(names(trainImageData), "Tr-no"), 1, 0)

costList <- c()

# Training loop
for(epoch in 1:15) {
    # Randomly shuffle the dataset
    indexes <- sample(1:length(trainImageData))
    print(epoch)
    for(i in indexes) {
        input <- trainImageData[[i]]
        output <- trainlabels[i]
        Learn(input, output)
    }
    if (epoch %% 1 == 0) {
        cost <- 0
        # Calculate mean squared error
        for(i in 1:length(trainImageData)) {
            input <- trainImageData[[i]]
            target <- trainlabels[i]
            o <- Predict(input)
            cost <- cost + (target - o) ** 2
        }
        cost <- cost / INPUTS
        print(paste("epoch", epoch, "mean squared error:", cost))
        costList <- append(costList, cost)
    }
}

# Predict labels for images and evaluate
predictedLabelList <- c()
probabilityList <- c()

for(i in 1:length(testImageData)) {
    input <- testImageData[[i]]
    result <- Predict(input)
    probabilityList <- append(probabilityList, result)
    predictedLabel <- round(result)
    predictedLabelList <- append(predictedLabelList, predictedLabel)
    trueLabel <- testlabels[i]
    compare <- ifelse(predictedLabel == trueLabel, "correct", "incorrect")
    print(paste("Image", i, " predicted:", predictedLabel,
                "true:", trueLabel, "which is" , compare))
}

# Generate a confusion matrix for evaluation
confusionMatrix(data = as.factor(predictedLabelList), 
               reference = as.factor(testlabels), 
               mode = "everything", positive = "1")

# Plot an ROC curve
library(pROC)
par(pty = "s")
roc(testlabels, probabilityList, plot = TRUE, legacy.axes = TRUE,
    col = "#00008B", lwd = 4, percent = TRUE, print.auc = TRUE, print.auc.x = 45)

# Plot the MSE over 15 epochs

epochList <- 1:15
costDf <- data.frame(epochList, costList)

ggplot(costDf, aes(x = epochList, y = costList)) +
    geom_line(color = "#69b3a2", linewidth = 1, alpha = 0.9, linetype = 1) +
    scale_x_continuous(breaks = seq(1, 15)) +
    scale_y_continuous(breaks = seq(0, 0.07, by = 0.01)) +
    theme_gray(base_size = 16) +
    xlab("Epoch") + 
    ylab("Mean Squared Error") 
