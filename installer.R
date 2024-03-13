if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")


BiocManager::install(c("graph", "RBGL", "Rgraphviz"))