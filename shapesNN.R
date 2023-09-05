library(matlab)
library(png)

VARIANCE <- 0.5

INPUTS <- 2352
HIDDEN <- 10
OUTPUTS <- 3
LEARNINGRATE <- 0.5

hiddenWeights <- vector("list", HIDDEN)

for(i in seq_along(hiddenWeights)) {
    hiddenWeights[[i]] <- runif(INPUTS, -VARIANCE, VARIANCE)
}

hiddenBias <- numeric(HIDDEN)

outputWeights <- vector("list", HIDDEN)

for(i in seq_along(outputWeights)) {
    outputWeights[[i]] <- runif(HIDDEN, -VARIANCE, VARIANCE)
}

outputBias <- numeric(OUTPUTS)



Sigmoid <- function(x) {
    return(1 / (1 + exp(-x)))
}



SigmoidPrime <- function(x) {
    return(x * (1 - x))
}



Predict <- function(inputs) {
    
    hiddens <- numeric(HIDDEN)
    for(i in 1:HIDDEN) {
        hidden <- 0
        for(j in 1:INPUTS) {
            hidden <- hidden + (hiddenWeights[[i]][j] * inputs[j])
        }
        hiddens[i] <- Sigmoid(hidden + hiddenBias[i])
    }
    
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

Learn <- function(inputs, targets) {
    
    hiddens <- numeric(HIDDEN)
    for(i in 1:HIDDEN) {
        hidden <- 0
        for(j in 1:INPUTS) {
            hidden <- hidden + (hiddenWeights[[i]][j] * inputs[j])
        }
        hiddens[i] <- Sigmoid(hidden + hiddenBias[i])
    }
    
    outputs <- numeric(OUTPUTS)
    for(i in 1:OUTPUTS) {
        output <- 0
        for(j in 1:HIDDEN) {
            output <- output + (outputWeights[[i]][j] * hiddens[j])
        }
        outputs[i] <- Sigmoid(output + outputBias[i])
    }
    
    errors <- numeric(OUTPUTS)
    for(i in 1:OUTPUTS) {
        errors[i] <- targets - outputs[i]
    }
    
    derrors <- numeric(OUTPUTS)
    for(i in 1:OUTPUTS) {
        derrors[i] <- errors[i] * SigmoidPrime(outputs[i])
    }
    
    ds <- numeric(HIDDEN)
    for(i in 1:OUTPUTS) {
        for(j in 1:HIDDEN) {
            ds[j] <- ds[j] + (derrors[i] * outputWeights[[i]][j] * 
                                  SigmoidPrime(hiddens[j]))
        }
    }
    
    for(i in 1:OUTPUTS) {
        for(j in 1:HIDDEN) {
            outputWeights[[i]][j] <<- outputWeights[[i]][j] + 
                (LEARNINGRATE * hiddens[j] * derrors[i])
        }
        outputBias[i] <<- outputBias[i] + (LEARNINGRATE * derrors[i])
    }
    
    for(i in 1:HIDDEN) {
        for(j in 1:INPUTS) {
            hiddenWeights[[i]][j] <<- hiddenWeights[[i]][j] +
                (LEARNINGRATE * inputs[j] * ds[i])
        }
        hiddenBias[i] <<- hiddenBias[i] + (LEARNINGRATE * ds[i])
    }
    
}


folderpath <- "./shapes/"

LoadImages <- function(folderpath) {
    shapes <- c("circles", "squares", "triangles")
    imageData <- list()
    for(shape in shapes) {
        for(i in 1:length(list.files(paste0(folderpath, shape)))) {
            filepath <- paste0(folderpath, shape)
            image <- readPNG(paste0(filepath, "/drawing(", i,").png"))
            imageData[[paste0(shape, "_", i)]] <- image
        }
    }
    return(imageData)
}

#imageData <- LoadImages(folderpath)


labels <- rep(1:3, each = 100)

for (epoch in 1:10000) {
    #print(epoch)
    indexes <- sample(1:length(imageData))
    for (i in indexes) {
        input <- imageData[[i]]
        output <- labels[i]
        Learn(input, output)
    }
    if (epoch %% 1000 == 0) {
        cost <- 0
        for (i in 1:length(imageData)) {
            input <- imageData[[i]]
            target <- labels[i]
            o <- Predict(input)
            cost <- cost + (target - o) ** 2
        }
        cost <- cost / INPUTS
        print(paste("epoch", epoch, "mean squared error:", cost))
    }
}

for (i in 1:length(imageData)) {
    input <- imageData[[i]]
    result <- Predict(input)
    predictedLabel <- which.max(result)
    trueLabel <- labels[i]
    compare <- ifelse(predictedLabel == trueLabel, "correct", "incorrect")
    print(paste("Image", i, " predicted:", predictedLabel,
                "true:", trueLabel, "which is" , compare))
}



