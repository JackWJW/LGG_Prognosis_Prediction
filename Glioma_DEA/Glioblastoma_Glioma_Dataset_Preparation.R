#Importing Packages
library(dplyr)
library(readr)
library(tibble)

#Downloading Counts Data
counts_download <- read.csv("C:/Users/jw2278/OneDrive - University of Cambridge/PhD Year 3/GSL_Metabolism_Project/Datasets/Raw_Data/TCGA_TARGET_GTEX_gene_expected_count.csv",header=TRUE, stringsAsFactors = FALSE)

#Downloading metadata
metadata <- read.delim("C:/Users/jw2278/OneDrive - University of Cambridge/PhD Year 3/GSL_Metabolism_Project/Datasets/Raw_Data/TcgaTargetGTEX_phenotype.txt", header = TRUE, sep = "\t", stringsAsFactors = FALSE)

#Fixing the Counts Rownames
rownames(counts_download) <- counts_download$sample
counts_download <- counts_download[,-1]
counts_pre <- counts_download[,-1]

#Backtransforming the count data to be whole number raw counts
counts_unlogged <- round(2^(as.matrix(counts_pre))-1)
counts <- counts_unlogged

#Adjusting sample names so it will match metadata
colnames(counts) <- gsub("\\.", "-", colnames(counts))

#Filter metatada to TCGA, Primary Tumor and Solid Tissue Normal
meta_tcga <- metadata %>%
  filter(X_study == "TCGA",
         `X_sample_type`== "Primary Tumor",
         `primary.disease.or.tissue` %in% c("Glioblastoma Multiforme","Brain Lower Grade Glioma")) %>%
  # ensure sample IDs match your counts colnames
  mutate(sample = gsub("\\.", "-", sample))

#Subset the counts matrix to only include the desired samples
samples.keep <- meta_tcga$sample
counts_tcga <- counts[, intersect(colnames(counts), samples.keep)]

# turn rownames into a real column called "gene_id"
df_counts <- as.data.frame(counts_tcga) %>%
  rownames_to_column(var = "gene_id")

write_csv(df_counts, file = "C:/Users/jw2278/OneDrive - University of Cambridge/PhD Year 3/GSL_Metabolism_Project/Datasets/Raw_Data/Glioblastoma_Glioma_Data/Glioblastoma_Glioma_counts.csv")

# 5) write metadata as before
write_csv(meta_tcga, file = "C:/Users/jw2278/OneDrive - University of Cambridge/PhD Year 3/GSL_Metabolism_Project/Datasets/Raw_Data/Glioblastoma_Glioma_Data/Glioblastoma_Glioma_metadata.csv")


