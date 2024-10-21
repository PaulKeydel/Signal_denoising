library(ggplot2)

generateNoise <- function(N, alpha = 0.0) {
    stopifnot(alpha <= 0.0)
    x <- rnorm(N)
    if (alpha == 0) {
        return(x)
    } else {
        numUniquePts <- ceiling((N + 1) / 2)
        x_ <- fft(x)[1:numUniquePts]
        s_ <- c(1:numUniquePts) ** (alpha / 2)
        x_ <- x_ * s_
        if (N %% 2 == 0) {
            x_ <- c(x_, rev(Conj(x_))[2:(numUniquePts - 1)])
        } else {
            x_ <- c(x_, rev(Conj(x_))[1:(numUniquePts - 1)])
        }
        x_ <- Re(fft(x_, inverse = TRUE))
        v_ <- sqrt(var(x_))
        x_ <- (x_ - mean(x_)) / v_
        return(x_)
    }
}

t <- rep(4, 1000)
n <- length(t)
df <- data.frame(id = c(1:n), orig = t, noise <- t + generateNoise(n, -2))
ggplot(df, aes(x = id)) +
    geom_point(aes(y = orig)) +
    geom_point(aes(y = noise))