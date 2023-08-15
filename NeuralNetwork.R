learning_rate <- 1
bias <- 1
#Generate vector of 3 randomly generated weights
weights <- runif(3)

Perceptron <- function(input1, input2, output) {
    
    perOutput <- input1 * weights[1] * input2 * 
        weights[2] + bias * weights[3]
    
    if (perOutput > 0) {
        perOutput <- 1
    } else {
        perOutput <- 0
    }
    
    error <- output - perOutput
    
    weights[1] <- weights[1] + error * input1 * learning_rate
    weights[2] <- weights[2] + error * input2 * learning_rate
    weights[3] <- weights[3] + error * bias * learning_rate
    
}

for (i in 1:500) {
    Perceptron(1, 1, 1)
    Perceptron(1, 0, 1)
    Perceptron(0, 1, 1)
    Perceptron(0, 0, 0)
}

x <- 0
y <- 0

x <- readline(prompt = "Enter x: ")
y <- readline(prompt = "Enter y: ")

x <- as.numeric(x)
y <- as.numeric(y)

perOutput <- x * weights[1] * y * weights[2] +
    bias * weights[3]

if (perOutput > 0) {
    perOutput <- 1
} else {
    perOutput <- 0
}

print(paste(x, " OR ", y, "is: ", perOutput))
