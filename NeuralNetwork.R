learning_rate <- 0.2
bias <- 0.0
weights <- runif(3)
success_rate <- 0  # Initialize success rate counter

sigmoid <- function(output) {
    output <- 1 / (1 + exp(-output))
    return(output)
}

Perceptron <- function(input1, input2, output) {
    
    perOutput <- (input1 * weights[1]) + (input2 * 
                                              weights[2]) + (bias * weights[3])
    
    if (perOutput > 0) {
        perOutput <- 1
    } else {
        perOutput <- 0
    }
    
    error <- output - perOutput
    
    weights[1] <- weights[1] + (error * input1 * learning_rate)
    weights[2] <- weights[2] + (error * input2 * learning_rate)
    weights[3] <- weights[3] + (error * bias * learning_rate)
    
    if (error == 0) {
        success_rate <<- success_rate + 1
    }
}

for (i in 1:100) {
    Perceptron(1, 1, 1)
    Perceptron(1, 0, 1)
    Perceptron(0, 1, 1)
    Perceptron(0, 0, 0)
    
    # Calculate and print success rate after each set of four instructions
    success_rate_percentage <- (success_rate / (i * 4)) * 100
    cat("Success Rate after set", i, ": ", success_rate_percentage, "%\n")
}

x <- as.numeric(readline(prompt = "Enter x: "))
y <- as.numeric(readline(prompt = "Enter y: "))

perOutput <- (x * weights[1]) + (y * weights[2]) +
    (bias * weights[3])

if (perOutput > 0) {
    perOutput <- 1
} else {
    perOutput <- 0
}

print(paste(x, " OR ", y, "is: ", perOutput))