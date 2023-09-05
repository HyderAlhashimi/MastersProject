library(matlab)

VARIANCE <- 0.5

INPUTS <- 2
HIDDEN <- 3
OUTPUTS <- 1
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
    
    inputs <- sapply(inputs, as.numeric)
    
    hiddens <- c()
    for(i in range(HIDDEN)) {
        hidden <- 0
        for(j in range(INPUTS)) {
            hidden <- hidden + (hiddenWeights[[i]][j] * inputs[j])
        }
        hidden <- Sigmoid(hidden + hiddenBias[i])
        hiddens <- append(hiddens, hidden)
    }
    
    outputs <- c()
    for(i in range(OUTPUTS)) {
        output <- 0
        for(j in range(HIDDEN)) {
            output <- output + (outputWeights[[i]][j] * outputs[j])
        }
        output <- Sigmoid(output + outputBias[i])
        outputs <- append(outputs, output)
    }
    
    return(output) ##maybe 'outputs' instead
}


Learn <- function(inputs, targets) {
    
    inputs <- sapply(inputs, as.numeric)
    targets <- sapply(targets, as.numeric)
    
    hiddens <- c()
    for(i in range(HIDDEN)) {
        hidden <- 0
        for(j in range(INPUTS)) {
            hidden <- hidden + (hiddenWeights[[i]][j] * inputs[j])
        }
        hidden <- Sigmoid(hidden + hiddenBias[i])
        hiddens <- append(hiddens, hidden)
    }
    
    outputs <- c()
    for(i in range(OUTPUTS)) {
        output <- 0
        for(j in range(HIDDEN)) {
            output <- output + (outputWeights[[i]][j] * outputs[j])
        }
        output <- Sigmoid(output + outputBias[i])
        outputs <- append(outputs, output)
    }
    
    errors <- c()
    for(i in range(OUTPUTS)) {
        error <- targets[i] - outputs[i]
        append(errors, error)
    }
    
    derrors <- c()
    for(i in range(OUTPUTS)) {
        derror <- errors[i] * SigmoidPrime(outputs[i])
        append(derrors, derror)
    }
    
    ds <- vector("list", HIDDEN)
    ds <- sapply(ds, as.numeric)
    for(i in range(OUTPUTS)) {
        for(j in range(HIDDEN)) {
            ds[[j]] <- ds[[j]] + (derrors[i] * outputWeights[[i]][j] * 
                                  SigmoidPrime(hiddens[j]))
        }
    }
    
    for(i in range(OUTPUTS)) {
        print(i)
        print(j)
        for(j in range(HIDDEN)) {
            print(i)
            print(j)
            print(outputWeights[[1]][3])
            outputWeights[[i]][j] <- outputWeights[[i]][j] + 
                (LEARNINGRATE * hiddens[i] * derrors[i])
        }
    }
    
    for(i in range(HIDDEN)) {
        for(j in range(INPUTS)) {
            hiddenWeights[[i]][j] <- hiddenWeights[[i]][j] +
                (LEARNINGRATE * inputs[j] * ds[i])
        }
        hiddenBias[i] <- hiddenBias[i] + (LEARNINGRATE * ds[i])
    }
}

computeCost <- function(){
    
}






inp1 <- c(0, 0)
inp2 <- c(0, 1)
inp3 <- c(1, 0)
inp4 <- c(1, 1)
INPUT <- list(inp1, inp2, inp3, inp4)

out1 <- c(0)
out2 <- c(1)
out3 <- c(1)
out4 <- c(0)
OUTPUT <- list(out1, out2, out3, out4)

#Training the network

for (epoch in 1:10000) {
    indexes <- c(1, 2, 3, 4)
    indexes <- sample(indexes)
    for (i in indexes) {
        Learn(INPUT[i], OUTPUT[i])
    }
    if (epoch %% 1000 == 0) {
        cost <- 0
        for (i in 1:4) {
            o <- Predict(INPUT[i])
            cost <- cost + (OUTPUT[[i]][1] - o) ** 2
        }
        cost <- cost / 4
        print(paste("epoch ", epoch, " mean squared error: ", cost))
    }
}

for (i in 1:4) {
    result <- Predict(INPUT[i])
    compare <- ifelse(round(result) == OUTPUT[i], "correct", "incorrect")
    print(paste("for input ", INPUT[i], " expected: ", OUTPUT[[i]][1],
                " predicted: ", format(result, nsmall = 4), "which is "
                , compare))
}


