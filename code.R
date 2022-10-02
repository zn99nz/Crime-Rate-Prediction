rm(ls())

library(dplyr)
library(tidyr)
library(dr)
library(gam)
library(glmnet)
library(splines)


setwd('C:\\Users\\uccoo\\Desktop\\학교\\대학원1\\고급머신러닝\\final project')

crime <- read.csv('crimedata.csv') 

#####################
####  1. 전처리  ####
#####################
# 해당 데이터는 결측치가 '?'로 표시되어있음
NA.ind <- rep(0, ncol(crime))
for(i in 1:ncol(crime)){
  NA.ind[i] <- is.character(crime[,i])
}
NA.ind <- which(NA.ind == TRUE)
NA.rate <- apply(crime[,NA.ind], 2, function(x) mean(x == '?'))
round(NA.rate[NA.rate <= 0.5], 3)

# 결측치 비율이 50%가 넘는 변수는 삭제
round(NA.rate[NA.rate > 0.5], 3)
remove.var <- names(NA.rate)[NA.rate > 0.5]
crime.1 <- crime %>% dplyr::select(-c(remove.var))
str(crime.1)
# 24개 변수 제거(id 변수 2개, 설명변수 22개) -> 123개의 변수가 남음

# 결측치가 존재하는 observation 삭제
NA.obs <- which(apply(crime.1, 1, function(x) any(x == '?')) == TRUE)
length(NA.obs)
crime.2 <- crime.1[-NA.obs,]
str(crime.2) # 314개의 관측치 제거 -> 1901개의 관측치가 남음
summary(crime.2)

# 의미없는 id variable 제거 
# + 수치형 변수인데 문자형 변수로 존재하는 변수의 형변환
crime.3 <- as.data.frame(apply(crime.2[,-c(1:3)], 2, as.numeric))
str(crime.3)
summary(crime.3)
crime <- crime.3

# train-test split
500/nrow(crime) # test set size = 500
set.seed(2021021302)
train.ind <- sample(nrow(crime), nrow(crime) - 500)
crime.train <- crime[train.ind,]; raw.train <- crime.train
crime.test <- crime[-train.ind,]; raw.test <- crime.test

# parameter tuning 및 validation을 위한 train/validation set split
set.seed(2021021352)
index <- sample(nrow(crime.train), nrow(crime.train))
cv.index <- list()
for(i in 1:10){
  cv.index[[i]] <- index[(140*(i-1)+1):(140*i)]
}
cv.index[[10]] <- c(cv.index[[10]], index[1401])
names(cv.index) <- paste0('fold', 1:10)

# standardize
crime.train.std <- scale(crime.train[,1:102], scale = T, center = T)
crime.train <- as.data.frame(cbind(crime.train.std, round(crime.train$ViolentCrimesPerPop)))
colnames(crime.train)[103] <- 'ViolentCrimesPerPop'
raw.train.std <- crime.train


# test set standardize (train set의 mean과 sd 사용)
# crime.test <- raw.test
mean <- matrix(apply(raw.train[,1:102], 2, mean), nrow(crime.test), 102, byrow = T)
sd <- matrix(apply(raw.train[,1:102], 2, sd), nrow(crime.test), 102, byrow = T)
crime.test.std <- (crime.test[,1:102]-mean)/sd
crime.test <- as.data.frame(cbind(crime.test.std, round(crime.test$ViolentCrimesPerPop)))
colnames(crime.test)[103] <- 'ViolentCrimesPerPop'
raw.test.std <- crime.test

########################
#### 2. DR by LASSO ####
########################
# 해당 데이터는 총 102개의 설명변수가 존재하므로 차원축소가 필요함
crime.train <- raw.train.std

#### lasso glm ####
y <- crime.train$ViolentCrimesPerPop

grid <- 10^seq(1, -4, length = 100)
train.ls.error <- val.ls.error <- rep(0, 100)
ftrain.ls.error <- fval.ls.error <- rep(0, 10)
for(i in 1:100){
  for(j in 1:10){
    # train-valid set split
    val.ind <- cv.index[[j]]
    crime.val <- as.matrix(crime.train[val.ind,-103]); y.val <- y[val.ind]
    crime.cv.train <- as.matrix(crime.train[-val.ind,-103]); y.train <- y[-val.ind]
    
    # fitting lm
    crime.lasso <- glmnet(crime.cv.train, y.train, alpha = 1, lambda = grid[i],
                          family = poisson('sqrt'))
    ftrain.ls.error[j] <- mean((y.train - predict(crime.lasso, crime.cv.train, type = 'response'))^2)
    fval.ls.error[j] <- mean((y.val - predict(crime.lasso, crime.val, type = 'response'))^2)
  }
  train.ls.error[i] <- mean(ftrain.ls.error)
  val.ls.error[i] <- mean(fval.ls.error)
}

train.ls.error
val.ls.error 

windows(10, 10)
plot(1:100, sqrt(val.ls.error), type = 'l', lwd = 2, col = 2, ylab = 'RMSE', xlab = 'lambda',
     ylim = c(330, 420))
lines(1:100, sqrt(train.ls.error), type = 'l', lwd = 2, col = 3)
legend('topright', c('10-fold training RMSE', '10-fold CV RMSE'), lwd = c(2, 2), col = 3:2)


best.lambda <- grid[which.min(val.ls.error)] # 54번째가가 best 135326.8
best.crime.lasso <- glmnet(as.matrix(crime.train[, -103]), y, alpha = 1, lambda = best.lambda,
                           family = poisson('sqrt'))
sum(best.crime.lasso$beta != 0) # 86개의 변수. 16개의 변수는 0으로 shrink되었다.

sqrt(min(val.ls.error))

grid.crime.lasso <- glmnet(as.matrix(crime.train[,-103]), y, alpha = 1, lambda = grid,
                           family = poisson('sqrt'))
not.zero <- apply(grid.crime.lasso$beta, 2, function(x) sum(x != 0))
# 사용된 변수 개수와 validation RMSE
rbind(not.zero, sqrt(val.ls.error))
# 3~9개의 변수가 사용된 모델에서 GAM을 적용해보자!

round(grid[1:9], 4)

#### lasso gam ####
imp.var.coef <- list()
for(i in 1:9){
  imp.var.coef[[i]] <- grid.crime.lasso$beta[which(grid.crime.lasso$beta[,i] != 0), i]
}
imp.var.coef

imp.var.list <- list()
for(i in 1:9){
  j <- length(imp.var.list)
  imp.var <- names(grid.crime.lasso$beta[which(grid.crime.lasso$beta[,i] != 0), i])
  if(i != 1){
    if(length(imp.var.list[[j]]) != length(imp.var)) {imp.var.list[[j+1]] <- imp.var}
  }
  if(i == 1){imp.var.list[[1]] <- imp.var}
}
imp.var.list

#######################
## model selection 1 ##
#######################
## s.s
formula.list <- list()
for(i in 1:5){
  var.list <- imp.var.list[[i]]; n.var <- length(var.list)
  p <- ''
  for(j in 1:(n.var-1)){
    p <- paste(p, 's(', var.list[j], ')',' + ', sep = '')
  }
  p <- paste(p, 's(', var.list[n.var], ')',sep = '')
  formula.list[[i]] <- as.formula(paste('ViolentCrimesPerPop ~', p))
}

train.gamls.error <- val.gamls.error <- rep(0, 5)
ftrain.gamls.error <- fval.gamls.error <- rep(0, 10)
for(i in 1:5){
  f <- formula.list[[i]]
  for(j in 1:10){
    # train-valid set split
    val.ind <- cv.index[[j]]
    crime.val <- crime.train[val.ind,]
    crime.cv.train <- crime.train[-val.ind,]
    
    # fitting lm
    crime.gam.lasso <- gam(f, data = crime.cv.train, family = poisson('sqrt'))
    ftrain.gamls.error[j] <- mean((crime.cv.train$ViolentCrimesPerPop - fitted(crime.gam.lasso))^2)
    fval.gamls.error[j] <- mean((crime.val$ViolentCrimesPerPop -
                                   predict(crime.gam.lasso, crime.val, type = 'response'))^2)
  }
  train.gamls.error[i] <- mean(ftrain.gamls.error)
  val.gamls.error[i] <- mean(fval.gamls.error)
}
train.gamls.error
sqrt(val.gamls.error) 

f <- formula.list[[3]]
crime.gam.lasso <- gam(f, data = crime.train, family = poisson('sqrt'))

windows(10, 10)
par(mfrow = c(2, 3))
plot(crime.gam.lasso, se = T, col = 3)

### n.s
formula.list <- list()
for(i in 1:5){
  var.list <- imp.var.list[[i]]; n.var <- length(var.list)
  p <- ''
  for(j in 1:(n.var-1)){
    p <- paste(p, 'ns(', var.list[j], ',4)',' + ', sep = '')
  }
  p <- paste(p, 'ns(', var.list[n.var], ',4)',sep = '')
  formula.list[[i]] <- as.formula(paste('ViolentCrimesPerPop ~', p))
}

train.gamls.error <- val.gamls.error <- rep(0, 5)
ftrain.gamls.error <- fval.gamls.error <- rep(0, 10)
for(i in 1:5){
  f <- formula.list[[i]]
  for(j in 1:10){
    # train-valid set split
    val.ind <- cv.index[[j]]
    crime.val <- crime.train[val.ind,]
    crime.cv.train <- crime.train[-val.ind,]
    
    # fitting lm
    crime.gam.lasso <- gam(f, data = crime.cv.train, family = poisson('sqrt'))
    ftrain.gamls.error[j] <- mean((crime.cv.train$ViolentCrimesPerPop - fitted(crime.gam.lasso))^2)
    fval.gamls.error[j] <- mean((crime.val$ViolentCrimesPerPop -
                                   predict(crime.gam.lasso, crime.val, type = 'response'))^2)
  }
  train.gamls.error[i] <- mean(ftrain.gamls.error)
  val.gamls.error[i] <- mean(fval.gamls.error)
}
train.gamls.error
sqrt(val.gamls.error)

f <- formula.list[[4]]
crime.gam.lasso <- gam(f, data = crime.train, family = poisson('sqrt'))

windows(10, 10)
par(mfrow = c(2, 4))
plot(crime.gam.lasso, se = T, col = 3)

#######################
## model selection 2 ##
#######################
## s.s df cv
s.df <- 2:10
train.sls.error <- val.sls.error <- rep(0, 9)
ftrain.sls.error <- fval.sls.error <- rep(0, 10)
for(i in 1:9){
  d <- s.df[i]
  for(j in 1:10){
    val.ind <- cv.index[[j]]
    crime.val <- crime.train[val.ind,]
    crime.cv.train <- crime.train[-val.ind,]
    
    # fitting lm
    
    crime.gam.lasso <- gam(ViolentCrimesPerPop ~ s(racePctWhite, d) + s(FemalePctDiv, d) + s(TotalPctDiv, d) + 
                             s(PctKids2Par, d) + s(PctKidsBornNeverMar, d) +
                             s(PctPersDenseHous, d), data = crime.cv.train, family = poisson('sqrt'))
    ftrain.sls.error[j] <- mean((crime.cv.train$ViolentCrimesPerPop - fitted(crime.gam.lasso))^2)
    fval.sls.error[j] <- mean((crime.val$ViolentCrimesPerPop -
                                 predict(crime.gam.lasso, crime.val, type = 'response'))^2)
    predict(crime.gam.lasso, crime.val)
  }
  train.sls.error[i] <- mean(ftrain.sls.error)
  val.sls.error[i] <- mean(fval.sls.error)
}
train.sls.error
val.sls.error # df = 9일때 가장 좋음 132143.2
sqrt(min(val.sls.error))


## n.s eff.df cv
n.df <- 2:10
train.nls.error <- val.nls.error <- rep(0, 9)
ftrain.nls.error <- fval.nls.error <- rep(0, 10)
for(i in 1:9){
  d <- n.df[i]
  for(j in 1:10){
    val.ind <- cv.index[[j]]
    crime.val <- crime.train[val.ind,]
    crime.cv.train <- crime.train[-val.ind,]
    
    # fitting lm
    crime.gam.lasso <- gam(ViolentCrimesPerPop ~ ns(racePctWhite, d) + ns(MalePctDivorce, d) + 
                             ns(FemalePctDiv, d) + ns(TotalPctDiv, d) + ns(PctKids2Par, d) + 
                             ns(PctKidsBornNeverMar, d) + ns(PctPersDenseHous, d) + 
                             ns(HousVacant, d), data = crime.cv.train, family = poisson('sqrt'))
    ftrain.nls.error[j] <- mean((crime.cv.train$ViolentCrimesPerPop - fitted(crime.gam.lasso))^2)
    fval.nls.error[j] <- mean((crime.val$ViolentCrimesPerPop -
                                 predict(crime.gam.lasso, crime.val, type = 'response'))^2)
  }
  train.nls.error[i] <- mean(ftrain.nls.error)
  val.nls.error[i] <- mean(fval.nls.error)
}
train.nls.error
val.nls.error # df = 6일때 가장 좋음 128429.7
sqrt(min(val.nls.error))

windows(10, 10)
plot(2:10, sqrt(val.sls.error), type = 'l', lwd = 2, col = 3, 
     ylab = '10-fold CV RMSE', xlab = 'df', ylim = c(356, 372))
lines(2:10, sqrt(val.nls.error), type = 'l', lwd = 2, col = 2)
legend('topright', c('smoothing splines', 'natural cubic splines'), lwd = c(2, 2), col = 3:2)


crime.gams.lasso <- gam(ViolentCrimesPerPop ~ s(racePctWhite, 9) + s(FemalePctDiv, 9) + s(TotalPctDiv, 9) + 
                          s(PctKids2Par, 9) + s(PctKidsBornNeverMar, 9) +
                          s(PctPersDenseHous, 9), data = crime.train, family = poisson('sqrt'))

crime.gamn.lasso <- gam(ViolentCrimesPerPop ~ ns(racePctWhite, 6) + ns(MalePctDivorce, 6) + 
                          ns(FemalePctDiv, 6) + ns(TotalPctDiv, 6) + ns(PctKids2Par, 6) + 
                          ns(PctKidsBornNeverMar, 6) + ns(PctPersDenseHous, 6) + 
                          ns(HousVacant, 6), data = crime.train, family = poisson('sqrt'))

windows(10, 10)
par(mfrow = c(2, 3))
plot(crime.gams.lasso, se = T, col = 3)

windows(10, 10)
par(mfrow = c(2, 4))
plot(crime.gamn.lasso, se = T, col = 4)

#######################
## model selection 3 ##
#######################
key.var <- imp.var.list[[1]]
div.var <- c("FemalePctDiv", "MalePctDivorce", "TotalPctDiv")
hs.var <- c("HousVacant", "PctPersDenseHous")
var.list <- list()

for(i in 1:3){
  div.comb <- combn(3, i); div.n <- ncol(div.comb)
  for(j in 1:div.n){
    div <- div.var[div.comb[,j]]
    for(k in 1:2){
      hs.comb <- combn(2, k); hs.n <- ncol(hs.comb)
      for(l in 1:hs.n){
        hs <- hs.var[hs.comb[,l]]
        var <- c(key.var, div, hs)
        if(length(var.list) != 0){n.list <- length(var.list); var.list[[n.list+1]] <- var}
        if(length(var.list) == 0){var.list <- list(var)}
      }
    }
  }
}

formula.list <- list()
for(i in 1:21){
  var.list.1 <- var.list[[i]]; n.var <- length(var.list.1)
  p <- ''
  for(j in 1:(n.var-1)){
    p <- paste(p, 'ns(', var.list.1[j], ',6)',' + ', sep = '')
  }
  p <- paste(p, 'ns(', var.list.1[n.var], ',6)',sep = '')
  formula.list[[i]] <- as.formula(paste('ViolentCrimesPerPop ~', p))
}


train.cv.error <- val.cv.error <- rep(0, 21)
for(i in 1:21){
  f <- formula.list[[i]]
  ftrain.cv.error <- fval.cv.error <- rep(0, 10)
  for(j in 1:10){
    val.ind <- cv.index[[j]]
    crime.val <- crime.train[val.ind,]
    crime.cv.train <- crime.train[-val.ind,]
    
    # fitting gam
    crime.gam.lasso <- gam(f, data = crime.cv.train, family = poisson('sqrt'))
    ftrain.cv.error[j] <- mean((crime.cv.train$ViolentCrimesPerPop - fitted(crime.gam.lasso))^2)
    fval.cv.error[j] <- mean((crime.val$ViolentCrimesPerPop -
                                predict(crime.gam.lasso, crime.val, type = 'response'))^2)
  }
  train.cv.error[i] <- mean(ftrain.cv.error)
  val.cv.error[i] <- mean(fval.cv.error)
}
train.cv.error
val.cv.error
# full natural spline model보다 더 좋은 모델! 변수 개수도 8개 -> 6개로 줄었음
formula.list[[which.min(val.cv.error)]]
# ViolentCrimesPerPop ~ ns(racePctWhite, 6) + ns(PctKids2Par, 6) + ns(PctKidsBornNeverMar, 6) 
# + ns(MalePctDivorce, 6) + ns(HousVacant, 6) + ns(PctPersDenseHous, 6)

f <- formula.list[[which.min(val.cv.error)]]
crime.gam.lasso <- gam(f, data = crime.train, family = poisson('sqrt'))
windows(10, 10)
par(mfrow = c(2, 3))
plot(crime.gam.lasso, se = T, col = 3)

#######################
## model selection 4 ##
#######################
df.mat <- matrix(0, 3^6, 6)
i <- 1
for(a in 2:4){for(b in 4:6){for(c in 2:4){for(d in 4:6){for(e in 4:6){for(f in 2:4){
  df <- c(a, b, c, d, e, f); df.mat[i,] <- df; i <- i+1
}}}}}}

train.cv.error <- val.cv.error <- rep(0, 3^6)
for(i in 1:(3^6)){
  df <- df.mat[i,]
  ftrain.cv.error <- fval.cv.error <- rep(0, 10)
  for(j in 1:10){
    val.ind <- cv.index[[j]]
    crime.val <- crime.train[val.ind,]
    crime.cv.train <- crime.train[-val.ind,]
    
    # fitting gam
    crime.gam.lasso <- gam(ViolentCrimesPerPop ~ ns(racePctWhite, df[1]) + ns(PctKids2Par, df[2]) + 
                             ns(PctKidsBornNeverMar, df[3]) + ns(MalePctDivorce, df[4]) + 
                             ns(HousVacant, df[5]) + ns(PctPersDenseHous, df[6]), 
                           data = crime.cv.train, family = poisson('sqrt'))
    ftrain.cv.error[j] <- mean((crime.cv.train$ViolentCrimesPerPop - fitted(crime.gam.lasso))^2)
    fval.cv.error[j] <- mean((crime.val$ViolentCrimesPerPop -
                                predict(crime.gam.lasso, crime.val, type = 'response'))^2)
  }
  train.cv.error[i] <- mean(ftrain.cv.error)
  val.cv.error[i] <- mean(fval.cv.error)
}
train.cv.error
val.cv.error
min(val.cv.error) # 127299.8
df.mat[which.min(val.cv.error),] # 3 5 2 6 6 2


crime.gam.lasso <- gam(ViolentCrimesPerPop ~ ns(racePctWhite, 3) + ns(PctKids2Par, 5) + 
                         ns(PctKidsBornNeverMar, 2) + ns(MalePctDivorce, 6) + 
                         ns(HousVacant, 6) + ns(PctPersDenseHous, 2), 
                       data = crime.train, family = poisson('sqrt'))
windows(10, 10)
par(mfrow = c(2, 3))
plot(crime.gam.lasso, se = T, col = 2)



for(j in 1:10){
  val.ind <- cv.index[[j]]
  crime.val <- crime.train[val.ind,]
  crime.cv.train <- crime.train[-val.ind,]
  
  # fitting gam
  crime.gam.lasso <- gam(ViolentCrimesPerPop ~ ns(racePctWhite, 3) + ns(PctKids2Par, 5) + 
                           PctKidsBornNeverMar + ns(MalePctDivorce, 6) + 
                           ns(HousVacant, 6) + ns(PctPersDenseHous, 2), 
                         data = crime.cv.train, family = poisson('sqrt'))
  ftrain.cv.error[j] <- mean((crime.cv.train$ViolentCrimesPerPop - fitted(crime.gam.lasso))^2)
  fval.cv.error[j] <- mean((crime.val$ViolentCrimesPerPop -
                              predict(crime.gam.lasso, crime.val, type = 'response'))^2)
}
mean(ftrain.cv.error)
mean(fval.cv.error) # 126928.3
sqrt(126928.3)

# 최종 모델
crime.gam.lasso <- gam(ViolentCrimesPerPop ~ ns(racePctWhite, 3) + ns(PctKids2Par, 5) + 
                         PctKidsBornNeverMar + ns(MalePctDivorce, 6) + 
                         ns(HousVacant, 6) + ns(PctPersDenseHous, 2), 
                       data = crime.train, family = poisson('sqrt'))
windows(10, 10)
par(mfrow = c(2, 3))
plot(crime.gam.lasso, se = T, col = 2)


######################
#### 3. DR by SIR ####
######################
# 해당 데이터는 총 102개의 설명변수가 존재하므로 차원축소가 필요함
crime.train <- raw.train.std
set.seed(123); add <- sample(1:102, 2)
x.mat.std <- crime.train[,1:102]
sir.vc.std <- dr(crime.train$ViolentCrimesPerPop ~ as.matrix(x.mat.std), method = 'sir')

sir.vc.1 <- sir.vc.std$evectors[,1:10]
x.mat.sir <- as.matrix(x.mat.std[,1:100]) %*% as.matrix(sir.vc.1)
dim(x.mat.sir) # 10개의 변수로 축약
crime.sir.train <- as.data.frame(cbind(x.mat.sir, crime.train[,103]))
crime.train <- crime.sir.train
colnames(crime.train)[11] <- 'ViolentCrimesPerPop'
str(crime.train)


##### GLM 변수 개수별로 비교해보려고 함 ####
formula.list <- list()
for(i in 1:10){
  p <- ''
  for(j in 1:(i-1)){
    if(i == 1) break
    p <- paste(p, 'Dir', j, ' + ', sep = '')
  }
  p <- paste(p, 'Dir', i, sep = '')
  formula.list[[i]] <- as.formula(paste('ViolentCrimesPerPop ~', p))
}


train.MSE.lm <- val.MSE.lm <- rep(0, 10)
ftrain.MSE.lm <- fval.MSE.lm <- rep(0, 5)
for(i in 1:10){
  for(j in 1:10){
    # train-valid set split
    val.ind <- cv.index[[j]]
    crime.val <- crime.train[val.ind,]
    crime.cv.train <- crime.train[-val.ind,]
    
    # fitting gam
    crime.lm.sir <- glm(formula.list[[i]], data = crime.cv.train, family = poisson('sqrt'))
    ftrain.MSE.lm[j] <- mean((crime.cv.train$ViolentCrimesPerPop - fitted(crime.lm.sir))^2)
    fval.MSE.lm[j] <- mean((crime.val$ViolentCrimesPerPop -
                              predict(crime.lm.sir, crime.val, type = 'response'))^2)
  }
  train.MSE.lm[i] <- mean(ftrain.MSE.lm)
  val.MSE.lm[i] <- mean(fval.MSE.lm)
}

train.MSE.lm
# 145000.1 144729.5 141337.5 141363.6 141291.8 141054.8 126470.0 126204.3 123209.8 122817.9
val.MSE.lm
# 145952.5 145894.1 143467.9 156873.6 157832.5 158465.4 129042.9 129067.5 126058.3 126173.7


##### GAM 변수 개수별로 비교해보려고 함 ####
formula.list <- list()
for(i in 1:10){
  p <- ''
  for(j in 1:(i-1)){
    if(i == 1) break
    p <- paste(p, 's(Dir', j, ') + ', sep = '')
  }
  p <- paste(p, 's(Dir', i, ')', sep = '')
  formula.list[[i]] <- as.formula(paste('ViolentCrimesPerPop ~', p))
}


train.MSE.gam <- val.MSE.gam <- rep(0, 10)
ftrain.MSE.gam <- fval.MSE.gam <- rep(0, 10)
for(i in 1:10){
  for(j in 1:10){
    # train-valid set split
    val.ind <- cv.index[[j]]
    crime.val <- crime.train[val.ind,]
    crime.cv.train <- crime.train[-val.ind,]
    
    # fitting gam
    crime.gam.sir <- gam(formula.list[[i]], data = crime.cv.train, 
                         family = poisson('sqrt'))
    ftrain.MSE.gam[j] <- mean((crime.cv.train$ViolentCrimesPerPop - fitted(crime.gam.sir))^2)
    fval.MSE.gam[j] <- mean((crime.val$ViolentCrimesPerPop -
                               predict(crime.gam.sir, crime.val, type = 'response'))^2)
  }
  train.MSE.gam[i] <- mean(ftrain.MSE.gam)
  val.MSE.gam[i] <- mean(fval.MSE.gam)
}

train.MSE.gam
# 139002.2 135591.9 128936.9 122294.5 122330.2 121898.5 116866.1 116004.3 113691.1 113247.7
val.MSE.gam
# 141185.8 143554.3 170772.9 232539.9 232970.6 194181.0 125266.4 128962.2 131734.3 123771.1


windows(10, 10)
plot(1:10, sqrt(val.MSE.lm), col = 3, type = 'l', lwd = 2, 
     ylab = '10-fold CV RMSE', xlab = 'number of component', ylim = c(350, 490))
lines(1:10, sqrt(val.MSE.gam), col = 2, lwd = 2)
legend('topright', c('GLM', 'GAM'), col = 3:2, lwd = c(2, 2))

sqrt(min(val.MSE.lm))
sqrt(min(val.MSE.gam))

# GAM 모델 10 선택

#### smoothing spline ####
ef.df <- 2:8

# 모델 10
train.MSE.s <- val.MSE.s <- rep(0, length(ef.df))
for(i in 1:length(ef.df)){
  df.cv <- ef.df[i]
  ftrain.MSE.s <- fval.MSE.s <- rep(0, 10)
  for(j in 1:10){
    # train-valid set split
    val.ind <- cv.index[[j]]
    crime.val <- crime.train[val.ind,]
    crime.cv.train <- crime.train[-val.ind,]
    
    # fitting GAM
    crime.gam.s <- gam(ViolentCrimesPerPop ~ s(Dir1, df = df.cv) + s(Dir2, df = df.cv) + 
                         s(Dir3, df = df.cv) + s(Dir4, df = df.cv) + s(Dir5, df = df.cv) +
                         s(Dir6, df = df.cv) + s(Dir7, df = df.cv) + s(Dir8, df = df.cv) +
                         s(Dir9, df = df.cv) + s(Dir10, df = df.cv),
                       data = crime.cv.train, family = poisson('sqrt'))
    ftrain.MSE.s[j] <- mean((crime.cv.train$ViolentCrimesPerPop - fitted(crime.gam.s))^2)
    fval.MSE.s[j] <- mean((crime.val$ViolentCrimesPerPop -
                             predict(crime.gam.s, crime.val, type = 'response'))^2)
  }
  train.MSE.s[i] <- mean(ftrain.MSE.s)
  val.MSE.s[i] <- mean(fval.MSE.s)
}
train.MSE.s
val.MSE.s; min(val.MSE.s) # 123771.1, df = 4


#### natural spline ####
n.df <- 2:8

# 모델 10
train.MSE.n <- val.MSE.n <- rep(0, length(n.df))
for(i in 1:length(n.df)){
  df.cv <- n.df[i]
  ftrain.MSE.n <- fval.MSE.n <- rep(0, 10)
  for(j in 1:10){
    # train-valid set split
    val.ind <- cv.index[[j]]
    crime.val <- crime.train[val.ind,]
    crime.cv.train <- crime.train[-val.ind,]
    
    # fitting GAM
    crime.gam.n <- gam(ViolentCrimesPerPop ~ ns(Dir1, df = df.cv) + ns(Dir2, df = df.cv) + 
                         ns(Dir3, df = df.cv) + ns(Dir4, df = df.cv) + ns(Dir5, df = df.cv) +
                         ns(Dir6, df = df.cv) + ns(Dir7, df = df.cv) + ns(Dir8, df = df.cv) +
                         ns(Dir9, df = df.cv) + ns(Dir10, df = df.cv),
                       data = crime.cv.train, family = poisson('sqrt'))
    ftrain.MSE.n[j] <- mean((crime.cv.train$ViolentCrimesPerPop - fitted(crime.gam.n))^2)
    fval.MSE.n[j] <- mean((crime.val$ViolentCrimesPerPop -
                             predict(crime.gam.n, crime.val, type = 'response'))^2)
  }
  train.MSE.n[i] <- mean(ftrain.MSE.n)
  val.MSE.n[i] <- mean(fval.MSE.n)
}
train.MSE.n
val.MSE.n; min(val.MSE.n) # 127170.5, df = 4


windows(10, 10)
plot(2:8, sqrt(val.MSE.s), type = 'l', col = 3, lwd = 2,
     ylim = c(349, 450), ylab = '10-fold CV RMSE', xlab = 'df')
lines(2:8, sqrt(val.MSE.n), type = 'l', col = 2, lwd = 2)
legend('topleft', c('smoothing splines', 'natural cubic splines'), lwd = c(2, 2), col = 3:2)

crime.gam.s <- gam(ViolentCrimesPerPop ~ s(Dir1, 4) + s(Dir2, 4) + 
                     s(Dir3, 4) + s(Dir4, 4) + s(Dir5, 4) + s(Dir6, 4) + s(Dir7, 4) + s(Dir8, 4) +
                     s(Dir9, 4) + s(Dir10, 4), data = crime.train, family = poisson('sqrt'))
windows(10, 10)
par(mfrow = c(2, 5))
plot(crime.gam.s, se = T, col = 3)


# 자유도 조절
df.mat <- matrix(0, 8, 10)
i <- 1
for(a in 2:3){for(b in 3:4){for(c in 4:5){
  df.mat[i, c(1, 3, 4, 9)] <- rep(a, 4)
  df.mat[i, 6:7] <- rep(b, 2)
  df.mat[i, c(2, 5, 8, 10)] <- rep(c, 4)
  i <- i+1
}}}

train.MSE.s <- val.MSE.s <- rep(0, nrow(df.mat))
for(i in 1:nrow(df.mat)){
  d <- df.mat[i,]
  ftrain.MSE.s <- fval.MSE.s <- rep(0, 10)
  for(j in 1:10){
    # train-valid set split
    val.ind <- cv.index[[j]]
    crime.val <- crime.train[val.ind,]
    crime.cv.train <- crime.train[-val.ind,]
    
    # fitting GAM
    crime.gam.s <- gam(ViolentCrimesPerPop ~ s(Dir1, df = d[1]) + s(Dir2, df = d[2]) + 
                         s(Dir3, df = d[3]) + s(Dir4, df = d[4]) + s(Dir5, df = d[5]) +
                         s(Dir6, df = d[6]) + s(Dir7, df = d[7]) + s(Dir8, df = d[8]) +
                         s(Dir9, df = d[9]) + s(Dir10, df = d[10]),
                       data = crime.cv.train, family = poisson('sqrt'))
    ftrain.MSE.s[j] <- mean((crime.cv.train$ViolentCrimesPerPop - fitted(crime.gam.s))^2)
    fval.MSE.s[j] <- mean((crime.val$ViolentCrimesPerPop -
                             predict(crime.gam.s, crime.val, type = 'response'))^2)
  }
  train.MSE.s[i] <- mean(ftrain.MSE.s)
  val.MSE.s[i] <- mean(fval.MSE.s)
}
train.MSE.s
val.MSE.s; min(val.MSE.s) # 124996.9
df.mat[which.min(val.MSE.s),] # 3 4 3 3 4 4 4 4 3 4
# 성능 향상이 전혀 되지 않음...
# 자유도를 임의로 조정해보자

for(j in 1:10){
  # train-valid set split
  val.ind <- cv.index[[j]]
  crime.val <- crime.train[val.ind,]
  crime.cv.train <- crime.train[-val.ind,]
  
  # fitting GAM
  crime.gam.s <- gam(ViolentCrimesPerPop ~ s(Dir1, 4) + s(Dir2, 2) + 
                       s(Dir3, 4) + s(Dir4, 3) + s(Dir5, 2) + s(Dir6, 4) + s(Dir7, 4) + 
                       s(Dir8, 2) + s(Dir9, 2) + s(Dir10, 4),
                     data = crime.cv.train, family = poisson('sqrt'))
  ftrain.MSE.s[j] <- mean((crime.cv.train$ViolentCrimesPerPop - fitted(crime.gam.s))^2)
  fval.MSE.s[j] <- mean((crime.val$ViolentCrimesPerPop -
                           predict(crime.gam.s, crime.val, type = 'response'))^2)
}
mean(fval.MSE.s)
sqrt(121799.5)
# 121799.5
# 가장 좋은 성능! 최종 모델로 선택


# 최종 모델
crime.gam.s <- gam(ViolentCrimesPerPop ~ s(Dir1, 4) + s(Dir2, 2) + 
                     s(Dir3, 4) + s(Dir4, 3) + s(Dir5, 2) + s(Dir6, 4) + s(Dir7, 4) + 
                     s(Dir8, 2) + s(Dir9, 2) + s(Dir10, 4), 
                   data = crime.train, family = poisson('sqrt'))
windows(10, 10)
par(mfrow = c(2, 5))
plot(crime.gam.s, se = T, col = 3)


##### 2차 SIR #####
crime.train <- raw.train.std
x.mat.std <- cbind(crime.train[,1:102], crime.train[,add])
sir.vc.std <- dr(crime.train$ViolentCrimesPerPop ~ as.matrix(x.mat.std), method = 'sir')
round(cumsum(sir.vc.std$evalues/sum(sir.vc.std$evalues)), 3)
# 5개의 변수를 선택하면 전체 분산의 약 23%만 설명 가능함
# 30개의 변수를 사용해야 전체 분산의 약 70%를 설명 가능
# -> 차원을 좀 더 줄이면서 전체 분산을 더 많이 설명할 수 없을까?

# 코드 설명 : 첫 30개의 eigenvectors(eigenvale가 큰 순서대로)에서 loading 절댓값 상위 30개를 뽑았다.
# 즉, 각 PC에 기여도가 높은 상위 30개 변수를 뽑은 것
# -> 30개의 vectors에서 뽑은 30개의 loading 상위 변수 중 unique한 변수만 찾는다
# -> 각 변수가 30개의 eigenvector 중 몇개의 벡터에서 뽑혔는지 비율을 확인
# 즉, 전체 분산의 70%를 설명하는 30개의 PC에서 기여도가 높은 변수들이 무엇인지 확인하는 작업!
impV <- apply(sir.vc.std$evectors[,1:30], 2, 
              function(x) colnames(x.mat.std[head(order(abs(x), decreasing = T), 30)]))

impV.uni <- unique(as.vector(impV)); length(impV.uni) # unique한 변수 : 75개
imp.p <- rep(0, length(impV.uni)); names(imp.p) <- impV.uni
for(i in 1:length(imp.p)){
  var <- impV.uni[i]
  where.v <- which(as.vector(impV) == var)
  imp.p[i] <- length(where.v)/30
}
sort(round(imp.p, 4), decreasing = T)

impV.1 <- names(imp.p)[which(imp.p >= 0.5)]; length(impV.1)
# 각 PC의 기여도 상위 20개 변수 그룹 중 절반 이상의 그룹에 속한 변수 : 27개
# 해당 27개 변수만을 사용해 SIR 방법을 다시 사용해본다.

x.mat.1.std <- crime.train[,impV.1]
sir.vc.1.std <- dr(crime.train$ViolentCrimesPerPop ~ as.matrix(x.mat.1.std), method = 'sir')
round(cumsum(sir.vc.1.std$evalues/sum(sir.vc.1.std$evalues)), 4)
# SIR 결과, 첫 10개의 PC를 사용하면 전체 분산의 약 90%가 설명된다.
# 첫 5개 PC -> 77%

sir.vc.2 <- sir.vc.1.std$evectors[,1:10]
x.mat.sir.1 <- as.matrix(x.mat.1.std) %*% as.matrix(sir.vc.2)
dim(x.mat.sir.1) # 10개의 변수로 축약
crime.sir.train <- as.data.frame(cbind(x.mat.sir.1, crime.train[,103]))
crime.train <- crime.sir.train
colnames(crime.train)[11] <- 'ViolentCrimesPerPop'
str(crime.train)

ev <- round(apply(sir.vc.2, 2, function(x) ifelse(abs(x) < 0.2, 0, x)), 2)
PC.list <- list()
for(i in 1:10){
  pc <- ev[which(ev[,i] != 0),i]; names(pc) <- impV.1[which(ev[,i] != 0)]
  PC.list[[i]] <- pc
}
PC.list

##### GLM 변수 개수별로 비교해보려고 함 ####
formula.list <- list()
for(i in 1:10){
  p <- ''
  for(j in 1:(i-1)){
    if(i == 1) break
    p <- paste(p, 'Dir', j, ' + ', sep = '')
  }
  p <- paste(p, 'Dir', i, sep = '')
  formula.list[[i]] <- as.formula(paste('ViolentCrimesPerPop ~', p))
}

train.MSE.lm.1 <- val.MSE.lm.1 <- rep(0, 10)
ftrain.MSE.lm.1 <- fval.MSE.lm.1 <- rep(0, 10)
for(i in 1:10){
  for(j in 1:10){
    # train-valid set split
    val.ind <- cv.index[[j]]
    crime.val <- crime.train[val.ind,]
    crime.cv.train <- crime.train[-val.ind,]
    
    # fitting lm
    crime.lm.sir.1 <- glm(formula.list[[i]], data = crime.cv.train,
                          family = poisson('sqrt'))
    ftrain.MSE.lm.1[j] <- mean((crime.cv.train$ViolentCrimesPerPop - fitted(crime.lm.sir.1))^2)
    fval.MSE.lm.1[j] <- mean((crime.val$ViolentCrimesPerPop - 
                                predict(crime.lm.sir.1, crime.val, type = 'response'))^2)
  }
  train.MSE.lm.1[i] <- mean(ftrain.MSE.lm.1)
  val.MSE.lm.1[i] <- mean(fval.MSE.lm.1)
}
train.MSE.lm.1
# 133670.4 133702.6 133564.5 133638.1 133582.3 133589.7 133468.1 133376.6 133334.1 133301.8
val.MSE.lm.1
# 134533.2 135461.4 136309.4 136484.7 136553.4 136718.3 137399.3 137505.8 138062.6 138110.6
which.min(val.MSE.lm.1)
sqrt(134533.2)

##### GAM 변수 개수별로 비교해보려고 함 ####
formula.list <- list()
for(i in 1:10){
  p <- ''
  for(j in 1:(i-1)){
    if(i == 1) break
    p <- paste(p, 's(Dir', j, ') + ', sep = '')
  }
  p <- paste(p, 's(Dir', i, ')', sep = '')
  formula.list[[i]] <- as.formula(paste('ViolentCrimesPerPop ~', p))
}


train.MSE.gam.1 <- val.MSE.gam.1 <- rep(0, 10)
ftrain.MSE.gam.1 <- fval.MSE.gam.1 <- rep(0, 10)
for(i in 1:10){
  for(j in 1:10){
    # train-valid set split
    val.ind <- cv.index[[j]]
    crime.val <- crime.train[val.ind,]
    crime.cv.train <- crime.train[-val.ind,]
    
    # fitting gam
    crime.gam.sir.1 <- gam(formula.list[[i]], data = crime.cv.train, 
                           family = poisson('sqrt'))
    ftrain.MSE.gam.1[j] <- mean((crime.cv.train$ViolentCrimesPerPop - fitted(crime.gam.sir.1))^2)
    fval.MSE.gam.1[j] <- mean((crime.val$ViolentCrimesPerPop -
                                 predict(crime.gam.sir.1, crime.val, type = 'response'))^2)
  }
  train.MSE.gam.1[i] <- mean(ftrain.MSE.gam.1)
  val.MSE.gam.1[i] <- mean(fval.MSE.gam.1)
}

train.MSE.gam.1
# 132431.7 129909.8 128375.6 127502.4 127189.6 126664.1 125672.6 124778.7 124133.2 123716.4
val.MSE.gam.1
# 134660.6 134650.9 134782.1 134738.9 134953.6 135543.9 134547.7 134312.0 134943.5 134665.3
which.min(val.MSE.gam.1)
sqrt(134312.0)

# GLM vs GAM 비교
windows(10, 10)
plot(1:10, sqrt(val.MSE.lm.1), type = 'l', lwd = 2, col = 3,
     ylab = '10-fold CV RMSE', xlab = 'number of component', ylim = c(365, 372))
lines(1:10, sqrt(val.MSE.gam.1), lwd = 2, col = 2)
legend('topleft', c('GLM', 'GAM'), lwd = c(2, 2), col = 3:2)

# GLM 1차 SIR vs 2차 SIR 비교
windows(20, 10)
par(mfrow = c(1, 2))
plot(1:10, sqrt(val.MSE.lm), type = 'l', lwd = 3, col = 3, ylim = c(350, 490),
     ylab = '10-fold CV RMSE', xlab = 'number of component', main = 'GLM', xaxt="n",yaxt="n")
axis(side=1,at=1:10)
axis(side=2,at=1:10)
lines(1:10, sqrt(val.MSE.lm.1), lwd = 3, col = 2)
abline(v = 7, lwd = 2, col = 'gray', lty = 2)
legend('topright', c('1차 SIR', '2차 SIR'), lwd = c(3, 3), col = 3:2)
# GAM 1차 SIR vs 2차 SIR 비교
plot(1:10, sqrt(val.MSE.gam), type = 'l', lwd = 3, col = 3, ylim = c(350, 490),
     ylab = '10-fold CV RMSE', xlab = 'number of component', main = 'GAM', xaxt="n",yaxt="n")
axis(side=1,at=1:10)
axis(side=2,at=1:10)
lines(1:10, sqrt(val.MSE.gam.1), lwd = 3, col = 2)
abline(v = 7, lwd = 2, col = 'gray', lty = 2)
legend('topright', c('1차 SIR', '2차 SIR'), lwd = c(3, 3), col = 3:2)



# GAM모델 8 선택

#### smoothing spline ####
ef.df <- 2:10

# 모델 8
train.MSE.s <- val.MSE.s <- rep(0, length(ef.df))
for(i in 1:length(ef.df)){
  df.cv <- ef.df[i]
  ftrain.MSE.s <- fval.MSE.s <- rep(0, 10)
  for(j in 1:10){
    # train-valid set split
    val.ind <- cv.index[[j]]
    crime.val <- crime.train[val.ind,]
    crime.cv.train <- crime.train[-val.ind,]
    
    # fitting GAM
    crime.gam.s <- gam(ViolentCrimesPerPop ~ s(Dir1, df = df.cv) + s(Dir2, df = df.cv) + 
                         s(Dir3, df = df.cv) + s(Dir4, df = df.cv) + s(Dir5, df = df.cv) +
                         s(Dir6, df = df.cv) + s(Dir7, df = df.cv) + s(Dir8, df = df.cv),
                       data = crime.cv.train, family = poisson('sqrt'))
    ftrain.MSE.s[j] <- mean((crime.cv.train$ViolentCrimesPerPop - fitted(crime.gam.s))^2)
    fval.MSE.s[j] <- mean((crime.val$ViolentCrimesPerPop -
                             predict(crime.gam.s, crime.val, type = 'response'))^2)
  }
  train.MSE.s[i] <- mean(ftrain.MSE.s)
  val.MSE.s[i] <- mean(fval.MSE.s)
}
train.MSE.s
val.MSE.s; min(val.MSE.s) # 133884, df = 6
sqrt(133884)



windows(10, 10)
plot(ef.df, val.MSE.s, type = 'l', lwd = 2, col = 'blue', ylim = c(124000, 160000),
     xlab = 'effective df', ylab = '', main = '10-fold CV MSE of smoothing spline')
lines(ef.df, train.MSE.s, lwd = 2, col = 'red')
legend('topleft', c('training MSE','validation MSE'), lwd = c(2, 2), col = c('red', 'blue'))
which.min(val.MSE.s)+1 # effective df = 8일때가 가장 좋음
min(val.MSE.s)

#### natural spline ####
n.df <- 2:10

# 모델 8
train.MSE.n <- val.MSE.n <- rep(0, length(n.df))
for(i in 1:length(n.df)){
  df.cv <- n.df[i]
  ftrain.MSE.n <- fval.MSE.n <- rep(0, 10)
  for(j in 1:10){
    # train-valid set split
    val.ind <- cv.index[[j]]
    crime.val <- crime.train[val.ind,]
    crime.cv.train <- crime.train[-val.ind,]
    
    # fitting GAM
    crime.gam.n <- gam(ViolentCrimesPerPop ~ ns(Dir1, df = df.cv) + ns(Dir2, df = df.cv) + 
                         ns(Dir3, df = df.cv) + ns(Dir4, df = df.cv) + ns(Dir5, df = df.cv) +
                         ns(Dir6, df = df.cv) + ns(Dir7, df = df.cv) + ns(Dir8, df = df.cv),
                       data = crime.cv.train, family = poisson('sqrt'))
    ftrain.MSE.n[j] <- mean((crime.cv.train$ViolentCrimesPerPop - fitted(crime.gam.n))^2)
    fval.MSE.n[j] <- mean((crime.val$ViolentCrimesPerPop -
                             predict(crime.gam.n, crime.val, type = 'response'))^2)
  }
  train.MSE.n[i] <- mean(ftrain.MSE.n)
  val.MSE.n[i] <- mean(fval.MSE.n)
}
train.MSE.n
val.MSE.n; min(val.MSE.n) # 136237.8, df = 3


windows(10, 10)
plot(2:10, sqrt(val.MSE.s), type = 'l', col = 3, lwd = 2,
     ylim = c(363, 377), ylab = '10-fold CV RMSE', xlab = 'df')
lines(2:10, sqrt(val.MSE.n), type = 'l', col = 2, lwd = 2)
legend('topleft', c('smoothing splines', 'natural cubic splines'), lwd = c(2, 2), col = 3:2)


#######################
## model selection 2 ##
#######################
crime.gam.s <- gam(ViolentCrimesPerPop ~ s(Dir1, df = 6) + s(Dir2, df = 6) + 
                     s(Dir3, df = 6) + s(Dir4, df = 6) + s(Dir5, df = 6) +
                     s(Dir6, df = 6) + s(Dir7, df = 6) + s(Dir8, df = 6),
                   data = crime.train, family = poisson('sqrt'))

windows(10, 10)
par(mfrow = c(2, 4))
plot(crime.gam.s, se = T, col = 3)
# 해당 모델의 성능이 가장 좋게 나왔다. 여기서 더 발전시켜본다.
# 각 변수마다 비선형 함수 및 자유도를 달리 적용해본다.

df.mat <- matrix(0, ((2^5)*(3^3)), 8)
i <- 1
for(a in 2:4){for(b in 4:6){for(c in 4:6){for(d in 2:4){
  for(e in 4:6){for(f in 4:6){for(g in 4:6){for(h in 4:6){
    df <- c(a, b, c, d, e, f, g, h); df.mat[i,] <- df; i <- i+1
  }}}}}}}}

formula.list <- list()
for(i in 1:nrow(df.mat)){
  p <- ''
  for(j in 1:7){
    if(j %in% 5:6){p <- paste(p, 'ns(Dir', j, ', ', df.mat[i, j], ')',' + ', sep = '')}
    if(!(j %in% 5:6)){p <- paste(p, 's(Dir', j, ', ', df.mat[i, j], ')',' + ', sep = '')}
  }
  p <- paste(p, 'ns(Dir', j, ', ', df.mat[i, 8], ')',sep = '')
  formula.list[[i]] <- as.formula(paste('ViolentCrimesPerPop ~', p))
}

train.MSE.cv <- val.MSE.cv <- rep(0, nrow(df.mat))
for(i in 1:nrow(df.mat)){
  f <- formula.list[[i]]
  ftrain.MSE.cv <- fval.MSE.cv <- rep(0, 10)
  for(j in 1:10){
    # train-valid set split
    val.ind <- cv.index[[j]]
    crime.val <- crime.train[val.ind,]
    crime.cv.train <- crime.train[-val.ind,]
    
    # fitting GAM
    crime.gam.cv <- gam(f, data = crime.cv.train, family = poisson('sqrt'))
    ftrain.MSE.cv[j] <- mean((crime.cv.train$ViolentCrimesPerPop - fitted(crime.gam.cv))^2)
    fval.MSE.cv[j] <- mean((crime.val$ViolentCrimesPerPop -
                              predict(crime.gam.cv, crime.val, type = 'response'))^2)
  }
  train.MSE.cv[i] <- mean(ftrain.MSE.cv)
  val.MSE.cv[i] <- mean(fval.MSE.cv)
}
train.MSE.cv
val.MSE.cv; min(val.MSE.cv) # 성능 그다지 향상되지 않음. 임의로 자유도 조정해보기

ftrain.MSE.n <- fval.MSE.n <- rep(0, 10)
for(j in 1:10){
  # train-valid set split
  val.ind <- cv.index[[j]]
  crime.val <- crime.train[val.ind,]
  crime.cv.train <- crime.train[-val.ind,]
  
  # fitting GAM
  crime.gam.n <- gam(ViolentCrimesPerPop ~ s(Dir1, 4) + s(Dir2, 6) + 
                       s(Dir3, 6) + s(Dir4, 3) + ns(Dir5, 4) +
                       ns(Dir6, 4) + s(Dir7, 8) + ns(Dir8, 5),
                     data = crime.cv.train, family = poisson('sqrt'))
  ftrain.MSE.n[j] <- mean((crime.cv.train$ViolentCrimesPerPop - fitted(crime.gam.n))^2)
  fval.MSE.n[j] <- mean((crime.val$ViolentCrimesPerPop -
                           predict(crime.gam.n, crime.val, type = 'response'))^2)
}
mean(ftrain.MSE.n)
mean(fval.MSE.n)
sqrt(131568.5)
# 131568.5 성능 가장 좋음
# ViolentCrimesPerPop ~ s(Dir1, 4) + s(Dir2, 6) + s(Dir3, 6) + s(Dir4, 3) + ns(Dir5, 4) +
#  ns(Dir6, 4) + s(Dir7, 8) + ns(Dir8, 5)


# 최종 모델
crime.gam.s <- gam(ViolentCrimesPerPop ~ s(Dir1, 4) + s(Dir2, 6) + s(Dir3, 6) + s(Dir4, 3) + 
                     ns(Dir5, 4) + ns(Dir6, 4) + s(Dir7, 8) + ns(Dir8, 5),
                   data = crime.train, family = poisson('sqrt'))

windows(10, 10)
par(mfrow = c(2, 4))
plot(crime.gam.s, se = T, col = 3)

#### test!!! ####
#### 1차 SIR ####
crime.test <- raw.test.std
crime.train <- raw.train.std

x.mat.sir <- as.matrix(crime.train[,1:100]) %*% as.matrix(sir.vc.1)
dim(x.mat.sir) # 10개의 변수로 축약
crime.sir.train <- as.data.frame(cbind(x.mat.sir, crime.train[,103]))
crime.train <- crime.sir.train
colnames(crime.train)[11] <- 'ViolentCrimesPerPop'
str(crime.train)
crime.train.SIR.1 <- crime.train


x.mat.sir <- as.matrix(crime.test[,1:100]) %*% as.matrix(sir.vc.1)
dim(x.mat.sir) # 10개의 변수로 축약
crime.sir.test <- as.data.frame(cbind(x.mat.sir, crime.test[,103]))
crime.test <- crime.sir.test
colnames(crime.test)[11] <- 'ViolentCrimesPerPop'
str(crime.test)
crime.test.SIR.1 <- crime.test

crime.gam.SIR.1 <- gam(ViolentCrimesPerPop ~ s(Dir1, 4) + s(Dir2, 2) + 
                      s(Dir3, 4) + s(Dir4, 3) + s(Dir5, 2) + s(Dir6, 4) + s(Dir7, 4) + 
                      s(Dir8, 2) + s(Dir9, 2) + s(Dir10, 4), 
                      data = crime.test.SIR.1, family = poisson('sqrt'))

mean((crime.test.SIR.1$ViolentCrimesPerPop -
        predict(crime.gam.SIR.1, crime.test.SIR.1, type = 'response'))^2) # 106412
sqrt(106412) # 326.2085

windows(10, 10)
par(mfrow = c(2, 5))
plot(crime.gam.SIR.1, se = T, col = 3)

#### 2차 SIR ####
crime.test <- raw.test.std
crime.train <- raw.train.std


x.mat.sir <- as.matrix(crime.train[,impV.1]) %*% as.matrix(sir.vc.2)
dim(x.mat.sir) # 10개의 변수로 축약
crime.sir.train <- as.data.frame(cbind(x.mat.sir, crime.train[,103]))
crime.train <- crime.sir.train
colnames(crime.train)[11] <- 'ViolentCrimesPerPop'
str(crime.train)
crime.train.SIR.2 <- crime.train


x.mat.sir <- as.matrix(crime.test[,impV.1]) %*% as.matrix(sir.vc.2)
dim(x.mat.sir) # 10개의 변수로 축약
crime.sir.test <- as.data.frame(cbind(x.mat.sir, crime.test[,103]))
crime.test <- crime.sir.test
colnames(crime.test)[11] <- 'ViolentCrimesPerPop'
str(crime.test)
crime.test.SIR.2 <- crime.test


crime.gam.SIR.2 <- gam(ViolentCrimesPerPop ~ s(Dir1, 4) + s(Dir2, 6) + s(Dir3, 6) + s(Dir4, 3) + 
                     ns(Dir5, 4) + ns(Dir6, 4) + s(Dir7, 8) + ns(Dir8, 5),
                     data = crime.train.SIR.2, family = poisson('sqrt'))

mean((crime.test.SIR.2$ViolentCrimesPerPop -
        predict(crime.gam.SIR.2, crime.test.SIR.2, type = 'response'))^2) # 138597.3
sqrt(138597.3) # 372.2866


#### LASSO ####
crime.test.ls <- crime.test <- raw.test.std
crime.train.ls <- crime.train <- raw.train.std

crime.gam.ls <- gam(ViolentCrimesPerPop ~ ns(racePctWhite, 3) + ns(PctKids2Par, 5) + 
                    PctKidsBornNeverMar + ns(MalePctDivorce, 6) + 
                    ns(HousVacant, 6) + ns(PctPersDenseHous, 2), 
                    data = crime.train.ls, family = poisson('sqrt'))

mean((crime.test.ls$ViolentCrimesPerPop -
        predict(crime.gam.ls, crime.test.ls, type = 'response'))^2) # 131527.5
sqrt(131527.5) # 362.6672

windows(10, 10)
par(mfrow = c(2, 3))
plot(crime.gam.ls, se = T, col = 3, lwd = 2)


#### 범죄단계 예측 성능 ####
crime.q <- quantile(crime.train$ViolentCrimesPerPop, c(0.25, 0.5, 0.75))

test.set <- crime.test$ViolentCrimesPerPop
pred.SIR.1 <- predict(crime.gam.SIR.1, crime.test.SIR.1, type = 'response')
pred.SIR.2 <- predict(crime.gam.SIR.2, crime.test.SIR.2, type = 'response')
pred.ls <- predict(crime.gam.ls, crime.test.ls, type = 'response')

test.grade <- ifelse(test.set < crime.q[1], 1, 
                     ifelse(crime.q[1] <= test.set & test.set < crime.q[2], 2,
                            ifelse(crime.q[2] <= test.set & test.set < crime.q[3], 3, 4)))

pred.grade.SIR.1 <- ifelse(pred.SIR.1 < crime.q[1], 1, 
                     ifelse(crime.q[1] <= pred.SIR.1 & pred.SIR.1 < crime.q[2], 2,
                            ifelse(crime.q[2] <= pred.SIR.1 & pred.SIR.1 < crime.q[3], 3, 4)))

pred.grade.SIR.2 <- ifelse(pred.SIR.2 < crime.q[1], 1, 
                     ifelse(crime.q[1] <= pred.SIR.2 & pred.SIR.2 < crime.q[2], 2,
                            ifelse(crime.q[2] <= pred.SIR.2 & pred.SIR.2 < crime.q[3], 3, 4)))

pred.grade.ls <- ifelse(pred.ls < crime.q[1], 1, 
                     ifelse(crime.q[1] <= pred.ls & pred.ls < crime.q[2], 2,
                            ifelse(crime.q[2] <= pred.ls & pred.ls < crime.q[3], 3, 4)))


addmargins(table(test.grade, pred.grade.SIR.1))
addmargins(table(test.grade, pred.grade.SIR.2))
addmargins(table(test.grade, pred.grade.ls))
