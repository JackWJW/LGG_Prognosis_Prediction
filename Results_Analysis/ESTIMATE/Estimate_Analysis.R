library(tidyestimate)
library(tidyverse)

df <- read.csv("C:/Python_Projects/LGG_Prognosis_Prediction/CGGA_Data/CGGA.mRNAseq_693.RSEM-genes.20200506.txt", sep="\t")
rownames(df) <- df$Gene_Name
df <- select(df, -Gene_Name)
head(df[1:5])

filtered <- filter_common_genes(df, 
                                id = "hgnc_symbol", 
                                tidy = FALSE, 
                                tell_missing = TRUE, 
                                find_alias = TRUE)

scored <- estimate_score(filtered,
                         is_affymetrix = TRUE)

plot_purity(scored, is_affymetrix = TRUE)

head(scored)
write.csv(scored,"C:/Python_Projects/LGG_Prognosis_Prediction/Results_Analysis/ESTIMATE_Results.csv",row.names=FALSE)
