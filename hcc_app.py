# %reload_ext nb_black
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN, OPTICS, AffinityPropagation, AgglomerativeClustering
from pyclustering.cluster.kmedoids import kmedoids
import prince
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist, squareform
from umap import UMAP
from sklearn.metrics import silhouette_score

%matplotlib inline
from scipy import stats

plt.style.use(["dark_background"])
# %matplotlib ipympl


def main():
    st.title("Hematocellular Carcinoma Clustering Model")



if __name__ == '__main__':
    main()