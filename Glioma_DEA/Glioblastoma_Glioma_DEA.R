# ----- Step 1: Load Required Libraries -----
library(DESeq2)
library(dplyr)
library(ggplot2)
library(plotly)
library(readr)
library(ggrepel)

#Loading in Count Data
count_path <- "C:/Users/jw2278/OneDrive - University of Cambridge/PhD Year 3/GSL_Metabolism_Project/Datasets/Raw_Data/Glioblastoma_Glioma_Data/Glioblastoma_Glioma_counts.csv"
counts <- read.csv(count_path, header = TRUE, row.names=1,stringsAsFactors = FALSE)

# Load metadata CSV.
metadata_path <- "C:/Users/jw2278/OneDrive - University of Cambridge/PhD Year 3/GSL_Metabolism_Project/Datasets/Raw_Data/Glioblastoma_Glioma_Data/Glioblastoma_Glioma_metadata.csv"
metadata <- read.csv(metadata_path, header = TRUE, stringsAsFactors = TRUE)
metadata$sample <- gsub("\\-", ".", metadata$sample)

# Set the sample column as row names for alignment.
rownames(metadata) <- metadata$sample

#Checking the Data
head(counts[, 1:5])
head(metadata)

#Ensuring all count sample columns are in metadata
common_samples <- intersect(colnames(counts), rownames(metadata))

#Subsetting both datasets to common samples
counts_filtered <- counts[, common_samples]
metadata_filtered <- metadata[common_samples, ]

#Checking order matches
all(colnames(counts_filtered) == rownames(metadata_filtered))  # should return TRUE

# ----- Step 5: Create a DESeqDataSet and Run Differential Expression Analysis -----
dds <- DESeqDataSetFromMatrix(countData = counts_filtered,
                              colData = metadata_filtered,
                              design = ~ primary.disease.or.tissue + X_gender)

#Filter out genes with low counts.
n_samps <- ncol(dds)
min_samps <- ceiling(0.9 * n_samps)

keep <- (rowSums(counts(dds) > 0) >= min_samps) & rowMeans(counts(dds)) >= 10
dds <- dds[keep, ]

# Run the DESeq2 pipeline.
dds <- DESeq(dds)


#Preparing the comparisons we want to analyse
res_cancer <- results(dds, contrast = c("primary.disease.or.tissue","Glioblastoma Multiforme","Brain Lower Grade Glioma"))


#Converting the comparisons to dataframes
res_cancer_df <- as.data.frame(res_cancer)

###Converting ensembl ids to gene names###
#making an ensembl id column to join by
res_cancer_df$ensembl_id <- rownames(res_cancer_df)

#loading in gene map
gene_map <- read.csv("C:/Users/jw2278/OneDrive - University of Cambridge/PhD Year 3/GSL_Metabolism_Project/Datasets/Raw_Data/TCGA_TARGET_GTEX_gene_map.csv", stringsAsFactors = FALSE)

#merging gene column from gene map into the results dataframes
res_cancer_df <- merge(res_cancer_df, gene_map[,c("id","gene")],
                       by.x="ensembl_id",by.y="id",
                       all.x=TRUE)



# Flag significant genes.
res_cancer_df <- res_cancer_df %>% mutate(Change = ifelse(padj < 0.01 & log2FoldChange > log2(1.5), "Up", ifelse(padj < 0.01 & log2FoldChange < -log2(1.5), "Down", "NS")))
#Creating simpler dataframes with only the genes of interest for ease of visualisation
res_cancer_df_mini <- res_cancer_df %>% filter(gene %in% c('A4GALT', 'ABO', 'B3GALNT1', 'B3GALT1', 'B3GALT4', 'B3GALT5', 
                                                           'B3GNT2', 'B3GNT5', 'B4GALNT1', 'B4GALT5', 'B4GALT6', 
                                                           'FUT1', 'FUT2', 'FUT3', 'FUT5', 'FUT6', 'FUT9', 
                                                           'GAL3ST1', 'GCNT2', 'ST3GAL1', 'ST3GAL2', 'ST3GAL3', 'ST3GAL4', 
                                                           'ST3GAL5', 'ST3GAL6', 'ST6GALNAC2', 'ST6GALNAC3', 'ST6GALNAC4', 'ST6GALNAC5', 
                                                           'ST6GALNAC6', 'ST8SIA1', 'ST8SIA5', 'UGCG', 'UGT8'))

#Generating Volcano Plots
plot_title <- "Volcano Plot: Glioblastoma vs Lower Grade Glioma"
p1 <- ggplot(res_cancer_df_mini, aes(x = log2FoldChange, y = -log10(padj), label = gene,text=gene)) +
  geom_point(shape=21,size=4,alpha=0.8,color="black",fill="grey") +
  labs(x = "Log2(Fold Change)",
       y = "-Log10(FDR)")+
  scale_color_manual(values=c("Up"="#7FC97F","Down"="#BEAED4","NS"="grey"))+
  geom_hline(yintercept = -log10(0.01),
             linetype = "dashed") + 
  geom_vline(xintercept = c(-log2(1.5), log2(1.5)),
             linetype = "dashed")+
  geom_point(data = subset(res_cancer_df_mini, Change=="Up"),col="black",size=5,shape=21,fill="#7FC97F")+
  geom_point(data = subset(res_cancer_df_mini, Change=="Down"),col="black",size=5,shape=21,fill="#BEAED4")+
  geom_label_repel(data = subset(res_cancer_df_mini, Change!="NS"),max.overlaps = 20,size=7)+
  scale_x_continuous(breaks = seq(-2,2,1),limits=c(-2.2,2.2))+
  ylim(0,100)+
  theme_bw()+
  theme(panel.border = element_rect(colour = "black", fill = NA, linewidth= 1),    
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        axis.title = element_text(size=26,color="black"),
        axis.text = element_text(size=24,color="black"),
        axis.ticks = element_line(linewidth=1),
        axis.ticks.length=unit(0.25,"cm"),
        legend.position="none") 


# #Saving Results Dataframes
save_name <- "C:/Users/jw2278/OneDrive - University of Cambridge/PhD Year 3/GSL_Metabolism_Project/Datasets/Raw_Data/Glioblastoma_Glioma_Data/Glioblastoma_v_Glioma_Full.csv"
write.csv(as.data.frame(res_cancer_df), save_name, quote = FALSE, row.names = TRUE)
# 
save_name_mini <- "C:/Users/jw2278/OneDrive - University of Cambridge/PhD Year 3/GSL_Metabolism_Project/Datasets/Raw_Data/Glioblastoma_Glioma_Data/Glioblastoma_v_Glioma_GSL.csv"
write.csv(as.data.frame(res_cancer_df_mini), save_name_mini, quote = FALSE, row.names = TRUE)
# 
#Saving the Volcano Plot
fig_name <- "Glioblastoma_Glioma Volcano Plot.png"
ggsave(p1, filename = fig_name,device="png",path="C:/Users/jw2278/OneDrive - University of Cambridge/PhD Year 3/GSL_Metabolism_Project/Datasets/Raw_Data/Glioblastoma_Glioma_Data",height=6,width=8)
