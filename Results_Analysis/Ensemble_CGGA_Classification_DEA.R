library(DESeq2)
library(dplyr)
library(ggplot2)
library(plotly)
library(readr)
library(ggrepel)

count_path = "C:/Python_Projects/LGG_Prognosis_Prediction/CGGA_Data/CGGA.mRNAseq_693.Read_Counts-genes.20220620.txt"
counts <- read.csv(count_path, header = TRUE, row.names=1,stringsAsFactors = FALSE,sep="\t")

metadata_path <- "C:/Python_Projects/LGG_Prognosis_Prediction/CGGA_Data/CGGA_Results.csv"
metadata <- read.csv(metadata_path, header = TRUE, stringsAsFactors = TRUE)
metadata <- metadata %>% select(CGGA_ID,pred_Ensemble)
metadata$pred_Ensemble <- factor(metadata$pred_Ensemble)
rownames(metadata) <- metadata$CGGA_ID

#Ensuring all count sample columns are in metadata
common_samples <- intersect(colnames(counts), rownames(metadata))

#Subsetting both datasets to common samples
counts_filtered <- counts[, common_samples]
metadata_filtered <- metadata[common_samples, ]

#Checking order matches
all(colnames(counts_filtered) == rownames(metadata_filtered))  # should return TRUE

# Create a DESeqDataSet and Run Differential Expression Analysis
dds <- DESeqDataSetFromMatrix(countData = counts_filtered,
                              colData = metadata_filtered,
                              design = ~ pred_Ensemble)

#Filter out genes with low counts.
n_samps <- ncol(dds)
min_samps <- ceiling(0.9 * n_samps)

keep <- (rowSums(counts(dds) > 0) >= min_samps) & rowMeans(counts(dds)) >= 10
dds <- dds[keep, ]

# Run the DESeq2 pipeline.
dds <- DESeq(dds)

#Preparing the comparisons we want to analyse
res_cancer <- results(dds, contrast = c("pred_Ensemble",1,0))

#Converting the comparisons to dataframes
res_cancer_df <- as.data.frame(res_cancer)

res_cancer_df <- res_cancer_df %>% mutate(Change = ifelse(padj < 0.01 & log2FoldChange > 1, "Up", ifelse(padj < 0.01 & log2FoldChange < -1, "Down", "NS")))

#Saving Results Dataframes
save_name <- "C:/Python_Projects/LGG_Prognosis_Prediction/CGGA_Data/CCGA_Ensemble_DEG.csv"
write.csv(as.data.frame(res_cancer_df), save_name, quote = FALSE, row.names = TRUE)


plot_title <- "Test"
p1 <- ggplot(res_cancer_df, aes(x = log2FoldChange, y = -log10(padj))) +
  geom_point(aes(color = Change),alpha=0.8,shape=16,size=1) +
  labs(title = plot_title,
       x = "Log2 Fold Change",
       y = "-Log10 FDR")+
  scale_color_manual(values=c("Up"="#7FC97F","Down"="#BEAED4","NS"="grey"))+
  geom_hline(yintercept = -log10(0.01),
             linetype = "dashed") + 
  geom_vline(xintercept = c(-1, 1),
             linetype = "dashed")+
  theme_bw()+
  theme(panel.border = element_rect(colour = "black", fill = NA, linewidth= 1),    
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        axis.title = element_text(size=15,color="black"),
        axis.text = element_text(size=15,color="black"),
        legend.position=c(0.075,0.875),
        legend.background=element_rect(size=0.5,linetype="solid",color="black"),
        legend.title = element_text(size=12),
        legend.text = element_text(size=12)) 
