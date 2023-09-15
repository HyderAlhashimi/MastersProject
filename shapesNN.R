# Load necessary libraries
library(png)                  
library(caret)                
library(magick)               
library(image.ContourDetector) 
library(plot.matrix)          

# Hyper parameters and Constants
VARIANCE <- 0.5               # Variance for weight initialization
IMAGE_HEIGHT <- 28            # Height of input images
IMAGE_WIDTH <- 28             # Width of input images
INPUTS <- IMAGE_HEIGHT * IMAGE_WIDTH  # Number of input neurons
HIDDEN <- 16                  # Number of hidden neurons
OUTPUTS <- 1                  # Number of output neurons
LEARNINGRATE <- 0.01          # Learning rate for weight updates

# Initialize weights for the hidden layer
hiddenWeights <- vector("list", HIDDEN)
for(i in seq_along(hiddenWeights)) {
    hiddenWeights[[i]] <- runif(INPUTS, -VARIANCE, VARIANCE)
}

# Initialize biases for the hidden layer
hiddenBias <- numeric(HIDDEN)

# Initialize weights for the output layer
outputWeights <- vector("list", HIDDEN)
for(i in seq_along(outputWeights)) {
    outputWeights[[i]] <- runif(HIDDEN, -VARIANCE, VARIANCE)
}

# Initialize biases for the output layer
outputBias <- numeric(OUTPUTS)

# Sigmoid activation function
Sigmoid <- function(x) {
    return(1 / (1 + exp(-x)))
}

# Derivative of the Sigmoid activation function
SigmoidPrime <- function(x) {
    return(x * (1 - x))
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

# Folder containing shape images
folderpath <- "./shapes/"
shapes <- c("circles", "squares", "triangles")

# Function to load and preprocess images
LoadImages <- function(folderpath) {
    imageData <- list()
    for(shape in shapes) {
        for(i in 1:length(list.files(paste0(folderpath, shape)))) {
            filepath <- paste0(folderpath, shape)
            image <- image_read(paste0(filepath, "/drawing(", i, ").png"))
            image <- image_convert(image, type = "Grayscale")
            image <- as.integer(image[[1]])
            image <- image / 255
            imageData[[paste0(shape, "_", i)]] <- image
        }
    }
    return(imageData)
}

# Load and preprocess images
imageData <- LoadImages(folderpath)

# Create labels (1 for circles, 0 for non-circles)
labels <- ifelse(startsWith(names(imageData), "circles"), 1, 0)

# Training loop
for(epoch in 1:1000) {
    # Randomly shuffle the dataset
    indexes <- sample(1:length(imageData))
    for(i in indexes) {
        input <- imageData[[i]]
        output <- labels[i]
        Learn(input, output)
    }
    if (epoch %% 100 == 0) {
        cost <- 0
        # Calculate mean squared error
        for(i in 1:length(imageData)) {
            input <- imageData[[i]]
            target <- labels[i]
            o <- Predict(input)
            cost <- cost + (target - o) ** 2
        }
        cost <- cost / INPUTS
        print(paste("epoch", epoch, "mean squared error:", cost))
    }
}

# Predict labels for images and evaluate
predictedLabelList <- c()

for(i in 1:length(imageData)) {
    input <- imageData[[i]]
    result <- Predict(input)
    predictedLabel <- round(result)
    predictedLabelList <- append(predictedLabelList, predictedLabel)
    trueLabel <- labels[i]
    compare <- ifelse(predictedLabel == trueLabel, "correct", "incorrect")
    print(paste("Image", i, " predicted:", predictedLabel,
                "true:", trueLabel, "which is" , compare))
}

# Generate a confusion matrix for evaluation
confusionMatrix(data = as.factor(predictedLabelList), 
                reference = as.factor(labels))