setwd('C:/Users/k2111/program/research/data/csv/attention')

files <- list.files()
for (path in files) {
  dat <- read.csv(path, header=TRUE)
  result <- wilcox.test(dat$pos, dat$nega, paired=F, alternative='g', exact=T)
  print(path)
  print(result)
}