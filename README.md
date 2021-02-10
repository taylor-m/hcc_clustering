# hcc_clustering
 Hematocellular Carcinoma Clustering Model

The Hepatocellular Carcinoma dataset (HCC dataset) was collected at a University Hospital in Portugal. It contains real clinical data of 165 patients diagnosed with HCC. The purpose of my clustering model will be to create clusters that have a distribution of the mortality class average that will allow for cluster analysis to identify features of interest that may possibly be of interest in determining effect on mortality in patients with HCC.

End User Value:
The value in the clustering analysis of the  HCC dataset is providing additional insight into the ideal characteristics that compose the clusters having a lower overall mortality rate relative to the higher mortality groups.

Quantifiable Results:
The results will be quantified based on the distribution of mortality rates across cluster groups. The dataset proivides a target variable however the clusters will be made without introduction of the target variable to retain efficacy for clustering of future data. The metric will be primarily the mortality class distribution as the silhouette scores and other clustering metrics are ineffective with this particular dataset.

Visuals:
Clustering visuals used are FAMD dminesionality reduction due to the relatively balanced nature of the datas categorical and quantitative variables.

Results:
The objective will be answering the question of what distinguishing markers make up the higher and lower mortality groups that is easily distinguished from the rest of the clusters. The ideal number of cluster ranges from 3-6 due to the need for a clear range of values by mortality averages for each cluster that shows a clear pattern and distinction across groups.