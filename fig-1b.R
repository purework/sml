library(M3C)
library(ggplot2)
#library(data.table)

load('mtab-tcga.rda')
load('TCGA-K562.rda.gz')

mtab.tcga$pat = substr(mtab.tcga$barcode, 1, 12)
colnames(mat)[-c(1:6)] = toupper(colnames(mat)[-c(1:6)])
mat = mat[, -c(1:6)]
cn = mtab.tcga[match(colnames(mat), bw)]$barcode  
mat = mat[, !duplicated(cn)]
colnames(mat) = cn[!duplicated(cn)]
mat = log2(1 + mat)

lab = mtab.tcga[match(colnames(mat), barcode), c('proj', 'stype')]
lab$proj = gsub('TCGA-', '', lab$proj)
lt = 'Cancer'
pdf('fig-1b.pdf', width = 12, height = 8)
tsne(mat, labels = lab$proj, legendtextsize= 12, axistextsize = 12, dotsize = 1, legendtitle= lt, seed= 12345) + labs(title= 'Clustering of cancer types by STAs')
dev.off()
