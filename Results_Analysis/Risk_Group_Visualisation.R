library(dplyr)
library(ggplot2)
library(plotly)
library(readr)
library(ggrepel)

res_df <- read.csv("C:/Python_Projects/LGG_Prognosis_Prediction/Results_Analysis/CGGA_Ensemble_DEG_RS-20.csv")
names(res_df)[names(res_df)=='X'] <- 'gene'

# Flag significant genes.
res_cancer_df <- res_df %>% mutate(Change = ifelse(padj < 0.01 & log2FoldChange > log2(1.5), "Up", ifelse(padj < 0.01 & log2FoldChange < -log2(1.5), "Down", "NS")))
#Creating simpler dataframes with only the genes of interest for ease of visualisation
res_cancer_df_mini <- res_df %>% filter(gene %in% c('A4GALT', 'ABO', 'B3GALNT1', 'B3GALT1', 'B3GALT4', 'B3GALT5', 
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
  scale_x_continuous(breaks = seq(-1.5,1.5,0.5),limits=c(-1.6,1.6))+
  ylim(0,25)+
  theme_bw()+
  theme(panel.border = element_rect(colour = "black", fill = NA, linewidth= 1),    
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        axis.title = element_text(size=26,color="black"),
        axis.text = element_text(size=24,color="black"),
        axis.ticks = element_line(linewidth=1),
        axis.ticks.length=unit(0.25,"cm"),
        legend.position="none")

#Saving the Volcano Plot
fig_name <- "RS-20_CGGA-Val_Volcano_RiskGroups.png"
ggsave(p1, filename = fig_name,device="png",path="C:/Python_Projects/LGG_Prognosis_Prediction/Results_Analysis",height=6,width=8)