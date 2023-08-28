learning_rate <- 0.2

#Generate inptor of 2 randomly generated weights for first layer
weights1 <- runif(2)
bias1 <- 0

weights2 <- runif(2)
bias2 <- 0

weights3 <- runif(2) 
bias3 <- 0

outputWeights <- runif(3) 
outputBias <- 0

Sigmoid <- function(output) {
    return(1 / (1 + exp(-output)))
}

SigmoidPrime <- function(output) {
    return(output * (1 - output))
}

Predict <- function(input1, input2) {
    
    input1 <- as.numeric(input1)
    input2 <- as.numeric(input2)
    
    # Compute activations for the first layer
    sig1 <- (input1 * weights1[1]) + 
        (input2 * weights1[2]) + (bias1)
    sig1 <- Sigmoid(sig1)
    
    sig2 <- (input1 * weights2[1]) + 
        (input2 * weights2[2]) + (bias2)
    sig2 <- Sigmoid(sig2)
    
    sig3 <- (input1 * weights3[1]) + 
        (input2 * weights3[2]) + (bias3)
    sig3 <- Sigmoid(sig3)
    
    output <- (sig1 * outputWeights[1]) + (sig2 * outputWeights[2]) +
        (sig3 * outputWeights[3]) + outputBias
    output <- Sigmoid(output)
    
    return(output)
}

Learn <- function(input1, input2, target) {
    
    input1 <- as.numeric(input1)
    input2 <- as.numeric(input2)
    
    # Compute activations for the first layer
    sig1 <- (input1 * weights1[1]) + 
        (input2 * weights1[2]) + (bias1)
    sig1 <- Sigmoid(sig1)
    
    sig2 <- (input1 * weights2[1]) + 
        (input2 * weights2[2]) + (bias2)
    sig2 <- Sigmoid(sig2)
    
    sig3 <- (input1 * weights3[1]) + 
        (input2 * weights3[2]) + (bias3)
    sig3 <- Sigmoid(sig3)
    
    
    # Compute final output
    output <- (sig1 * outputWeights[1]) + (sig2 * outputWeights[2]) +
        (sig3 * outputWeights[3]) + outputBias
    output <- Sigmoid(output)
    
    
    # Compute errors and deltas for back propagation
    error <<- target - output
    deltError <- error * SigmoidPrime(output)
    deltSig1 <- deltError * outputWeights[1] * SigmoidPrime(sig1)
    deltSig2 <- deltError * outputWeights[2] * SigmoidPrime(sig2)
    deltSig3 <- deltError * outputWeights[3] * SigmoidPrime(sig3)
    
    
    outputWeights[1] <- outputWeights[1] + (deltError * sig1 * learning_rate)
    outputWeights[2] <- outputWeights[2] + (deltError * sig2 * learning_rate)
    outputWeights[3] <- outputWeights[3] + (deltError * sig3 * learning_rate)
    outputBias <- outputBias * deltError
    
    weights1[1] <- weights1[1] + (input1 * deltSig1 * learning_rate)
    weights1[2] <- weights1[2] + (input2 * deltSig1 * learning_rate)
    bias1 <- bias1 + (deltSig1 * learning_rate)
    weights2[1] <- weights2[1] + (input1 * deltSig2 * learning_rate)
    weights2[2] <- weights2[2] + (input2 * deltSig2 * learning_rate)
    bias2 <- bias2 + (deltSig2 * learning_rate)
    weights3[1] <- weights3[1] + (input1 * deltSig3 * learning_rate)
    weights3[2] <- weights3[2] + (input2 * deltSig3 * learning_rate)
    bias3 <- bias3 + (deltSig3 * learning_rate)

}

inp1 <- c(1, 1)
inp2 <- c(1, 0)
inp3 <- c(0, 1)
inp4 <- c(0, 0)
INPUTS <- list(inp1, inp2, inp3, inp4)

out1 <- c(0)
out2 <- c(1)
out3 <- c(1)
out4 <- c(0)
OUTPUTS <- list(out1, out2, out3, out4)

#Training the network

for (epoch in 1:10000) {
    indexes <- c(1, 2, 3, 4)
    indexes <- sample(indexes)
    for (i in indexes) {
        Learn(INPUTS[[i]][1], INPUTS[[i]][2], OUTPUTS[[i]][1])
    }
    if (epoch %% 1000 == 0) {
        cost <- 0
        for (i in 1:4) {
            o <- Predict(INPUTS[[i]][1], INPUTS[[i]][2])
            cost <- cost + (OUTPUTS[[i]][1] - o) ** 2
        }
        cost <- cost / 4
        print(paste("epoch ", epoch, " mean squared error: ", error))
    }
}

for (i in 1:4) {
    result <- Predict(INPUTS[[i]][1], INPUTS[[i]][2])
    compare <- ifelse(round(result) == OUTPUTS[[i]][1], "correct", "incorrect")
    print(paste("for input ", INPUTS[i], " expected: ", OUTPUTS[[i]][1],
                " predicted: ", format(result, nsmall = 4), "which is "
                , compare))
}




