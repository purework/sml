library(data.table)
library(readxl)
library(ggpubr)
library(stringr)

perf.test = read_excel('slc-ml-table.xlsx', sheet = 1)
colnames(perf.test)[1] = 'Cancer'
perf.test$Cancer = gsub('TCGA-', '', perf.test$Cancer)
perf.test = melt(perf.test)
perf.test = cbind(perf.test[, c(1, 3)], str_split(perf.test$variable, ' ', simplify = T)[, -1]) %>% data.table()
colnames(perf.test) = c('Cancer', 'Perf', 'Top', 'Type')
perf.test$Top = paste0('Top', perf.test$Top)
perf.test$Top = gsub('Top2661', 'All', perf.test$Top)
perf.test = perf.test[!(Cancer %in% c('Overall')) & !(Type == 'Specificity' & Cancer %in% strsplit('ACC BLCA CESC CHOL DLBC GBM LAML LGG MESO OV PAAD PCPG READ SARC SKCM TGCT THYM UCS UVM', ' ')[[1]])]

pdf('fig-3a.pdf', width= 20, height= 6)
ggbarplot(perf.test, x= 'Top', y= 'Perf', fill = 'Cancer', color = 'Cancer', 
          xlab= 'STA features', ylab= 'Performance', title= 'Top 100 STAs accurately distinguish tumors from non-tumors across cancer types',
          sort.by.groups = T, facet.by = 'Type', position = position_dodge2()) + theme(legend.position= 'right')
dev.off()
