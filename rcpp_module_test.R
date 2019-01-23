### rcpp_module_test.R

library(Rcpp)
#library(inline)
sourceCpp("rcpp_module_tst.cpp")

#create a new object using the C++ class Uniform
u = new(Uniform, 0, 10)

#call method draw
u$draw(10)

#call method range
u$range()

u$min
u$max

#update the max (a public element)
u$max = 1
u$max

#call range and draw methods again
u$range()
u$draw(10)

save.image('uniform.RData')

rm(list=ls())
load('uniform.RData')

u = new(Uniform, 0, 10)
u$draw(10)
u$min


sourcecpp(....)
rf = new(OnlineRF, MatrixOfParameters)
rf2 = rf$update(newData)
rf2$eval(moreData)

