# TODO: cluster notes dict, plot analysis
# TODO: Var Info feature for normal range info
# TODO: pattern finding function for cluster vars
# TODO: cluster model tab

# %reload_ext nb_black
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN, OPTICS, AffinityPropagation, AgglomerativeClustering
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.impute import KNNImputer
import prince
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from scipy import stats
from pyclustering.cluster import cluster_visualizer_multidim

def main():
    st.title("Hepatocellular Carcinoma Clustering Model")
    url = "https://raw.githubusercontent.com/taylor-m/hcc_clustering/main/hcc_data/hcc-data.csv"
    raw = pd.read_csv(url)
    cols = [
            "Gender",
            "Symptoms",
            "Alcohol",
            "HBsAg",
            "HBeAg",
            "HBcAb",
            "HCVAb",
            "Cirrhosis",
            "Endemic",
            "Smoking",
            "Diabetes",
            "Obesity",
            "Hemochro",
            "AHT",
            "CRI",
            "HIV",
            "NASH",
            "Varices",
            "Spleno",
            "PHT",
            "PVT",
            "Metastasis",
            "Hallmark",
            "Age",
            "Grams_day",
            "Packs_year",
            "PS",
            "Encephalopathy",
            "Ascites",
            "INR",
            "AFP",
            "Hemoglobin",
            "MCV",
            "Leucocytes",
            "Platelets",
            "Albumin",
            "Total_Bil",
            "ALT",
            "AST",
            "GGT",
            "ALP",
            "TP",
            "Creatinine",
            "Nodule",
            "Major_Dim",
            "Dir_Bil",
            "Iron",
            "Sat",
            "Ferritin",
            "Class",
            ]
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    #=========================================================================
    # DATA FUNCTIONS
    def load_data():
        df = pd.read_csv(url, names=cols)
        
        # changing the ? input values to np.nan for the imputer
        df[df == "?"] = np.nan

        # using the KNN imputer to impute the missing values
        imputer = KNNImputer(missing_values=np.nan)
        imputed = imputer.fit_transform(df)
        
        
        # creating a new df from the imputed array
        df = pd.DataFrame(imputed, columns=cols)
        cats = [
            "Gender",
            "Symptoms",
            "Alcohol",
            "HBsAg",
            "HBeAg",
            "HBcAb",
            "HCVAb",
            "Cirrhosis",
            "Endemic",
            "Smoking",
            "Diabetes",
            "Obesity",
            "Hemochro",
            "AHT",
            "CRI",
            "HIV",
            "NASH",
            "Varices",
            "Spleno",
            "PHT",
            "PVT",
            "Metastasis",
            "Hallmark",
            "Class",
            "PS",
            "Encephalopathy",
            "Ascites",
            "Nodule",
            ]
        
        # rounding all the values in cat columns because imputed values weren't binary
        for cat in cats:
            df[cat] = round(df[cat]).astype(int)
        return df
    
    # CLUSTERING FUNCTIONS
    def kmed_cluster(df, k):
        # generate k random indices from distance matrix
        df2 = df.drop(columns="Class") 
        
        X_scaled = scale_df(df2)
        dmat = create_dmat(X_scaled)

        np.random.seed(42)
        n_rows = dmat.shape[0]
        init_medoids = np.random.randint(0, n_rows, k)
        
        # init_medoids
        kmed = kmedoids(dmat, initial_index_medoids=init_medoids, data_type="distance_matrix")
        kmed.process()
        clusters = kmed.get_clusters()
        medoid_idxs = kmed.get_medoids()
        # medoid_idxs

        labels = kmed.predict(dmat)
        df["kmed"] = labels
        # print(df.kmed.value_counts())
        # group_df = df.groupby("kmed").mean().sort_values("Class").style.background_gradient()

        # casting kmed clusters to strings
        df.kmed = df.kmed.astype(str)

        # reordering cluster numbers by mortality rate
        df.loc[(df.kmed == "3"), "kmed"] = 4
        df.loc[(df.kmed == "0"), "kmed"] = 2
        df.loc[(df.kmed == "1"), "kmed"] = 3
        df.loc[(df.kmed == "4"), "kmed"] = 1
        df.loc[(df.kmed == "2"), "kmed"] = 0

        group_df = df.groupby("kmed").mean().style.background_gradient()
        # counts = df.kmed.value_counts().index.sort_values(ascending=False)
        # group_df["count"] = counts
        return df, group_df, clusters, dmat
    def kmed_predict(df, X, k=5):
        # generate k random indices from distance matrix
        # df = df.drop(columns="Class") 
        # X = X.drop(columns="kmed")
        # X = pd.DataFrame(X, columns=df.columns)
        df = df.append(X)
        X_scaled = scale_df(df)
        dmat = create_dmat(X_scaled)

        np.random.seed(42)
        n_rows = dmat.shape[0]
        init_medoids = np.random.randint(0, n_rows, k)
        
        # init_medoids
        kmed = kmedoids(dmat, initial_index_medoids=init_medoids, data_type="distance_matrix")
        kmed.process()

        medoid_idxs = kmed.get_medoids()
        # medoid_idxs

        labels = kmed.predict(dmat)
        df["kmed"] = labels
        # print(df.kmed.value_counts())
        # group_df = df.groupby("kmed").mean().sort_values("Class").style.background_gradient()

        # casting kmed clusters to strings
        df.kmed = df.kmed.astype(str)

        # reordering cluster numbers by mortality rate
        df.loc[(df.kmed == "3"), "kmed"] = 4
        df.loc[(df.kmed == "0"), "kmed"] = 2
        df.loc[(df.kmed == "1"), "kmed"] = 3
        df.loc[(df.kmed == "4"), "kmed"] = 1
        df.loc[(df.kmed == "2"), "kmed"] = 0

        group_df = df.groupby("kmed").mean().style.background_gradient()
        # counts = df.kmed.value_counts().index.sort_values(ascending=False)
        # group_df["count"] = counts
        output = df.head(-1)
        return df
    def scale_df(df):
        # scale data for cluster
        scaler = StandardScaler()
        # looking at df with target var first
        X_scaled = scaler.fit_transform(df)
        X_scaled = pd.DataFrame(X_scaled, columns=df.columns, index=df.index)
        return X_scaled
    def create_dmat(scaled):
        # distance matrix
        dist = pdist(scaled, metric="cosine")
        dmat = squareform(dist)
        return dmat
    
    # PLOTTING FUNCTIONS
    def plot_boxplot(var):
        fig = go.Figure()
        # for i in range(df.kmed.unique()):
        fig.add_trace(
            go.Box(
                y=df[var],
                x=df.kmed,
                boxpoints=False,  # no data points
                #     marker_color='rgb(9,56,125)',
                #     line_color='rgb(9,56,125)'
            )
        )
        st.plotly_chart(fig)
    def plot_violin(var):
        fig = go.Figure()
        fig.add_trace(
            go.Violin(
                y=df[var],
                x=df.kmed,
            )
        )
        fig.update_traces(meanline_visible=True)

        st.plotly_chart(fig)
    def plot_barplot(var):
        fig = go.Figure(data=[
            go.Bar(name="Alive @ 1 year", x=df.kmed, y=df[df.Class == 1][var]),
            go.Bar(name="Deceeased @ 1 year", x=df.kmed, y=df[df.Class == 0][var]),
            ])
        fig.update_layout(barmode='stack')
        st.plotly_chart(fig)
    def plot_hist(var, cluster_num):
        cluster_df = df[df.kmed == cluster_num]
        fig = px.histogram(cluster_df, x=var)
        st.plotly_chart(fig)
    #=========================================================================
    cols = [
        "Gender",
        "Symptoms",
        "Alcohol",
        "HBsAg",
        "HBeAg",
        "HBcAb",
        "HCVAb",
        "Cirrhosis",
        "Endemic",
        "Smoking",
        "Diabetes",
        "Obesity",
        "Hemochro",
        "AHT",
        "CRI",
        "HIV",
        "NASH",
        "Varices",
        "Spleno",
        "PHT",
        "PVT",
        "Metastasis",
        "Hallmark",
        "Age",
        "Grams_day",
        "Packs_year",
        "PS",
        "Encephalopathy",
        "Ascites",
        "INR",
        "AFP",
        "Hemoglobin",
        "MCV",
        "Leucocytes",
        "Platelets",
        "Albumin",
        "Total_Bil",
        "ALT",
        "AST",
        "GGT",
        "ALP",
        "TP",
        "Creatinine",
        "Nodule",
        "Major_Dim",
        "Dir_Bil",
        "Iron",
        "Sat",
        "Ferritin",
        "Class",
        ]
    bin_cols = [
        "Gender",
        "Symptoms",
        "Alcohol",
        "HBsAg",
        "HBeAg",
        "HBcAb",
        "HCVAb",
        "Cirrhosis",
        "Endemic",
        "Smoking",
        "Diabetes",
        "Obesity",
        "Hemochro",
        "AHT",
        "CRI",
        "HIV",
        "NASH",
        "Varices",
        "Spleno",
        "PHT",
        "PVT",
        "Metastasis",
        "Hallmark",
        "Class",
        ]
    yes_no = [
        "Symptoms",
        "Alcohol",
        "HBsAg",
        "HBeAg",
        "HBcAb",
        "HCVAb",
        "Cirrhosis",
        "Endemic",
        "Smoking",
        "Diabetes",
        "Obesity",
        "Hemochro",
        "AHT",
        "CRI",
        "HIV",
        "NASH",
        "Varices",
        "Spleno",
        "PHT",
        "PVT",
        "Metastasis",
        "Hallmark",
        ]
    num_cols = [
        "Age",
        "Grams_day",
        "Packs_year",
        "INR",
        "AFP",
        "Hemoglobin",
        "MCV",
        "Leucocytes",
        "Platelets",
        "Albumin",
        "Total_Bil",
        "ALT",
        "AST",
        "GGT",
        "ALP",
        "TP",
        "Creatinine",
        "Major_Dim",
        "Dir_Bil",
        "Iron",
        "Sat",
        "Ferritin",
        ]
    cat_cols = ["PS", "Encephalopathy", "Ascites", "Nodule"]
    #=========================================================================
    analysis_vars = [
            "Iron",
            "Ferritin",
            "Dir_Bil",
            "Nodule",
            "GGT",
            "ALP",
            "Total_Bil",
            "Albumin",
            "Platelets",
            "Hemoglobin",
            "Gender",
            "Symptoms",
            "Alcohol",
            "HBsAg",
            "HBeAg",
            "HBcAb",
            "HCVAb",
            "Cirrhosis",
            "Endemic",
            "Smoking",
            "Diabetes",
            "Obesity",
            "Hemochro",
            "AHT",
            "CRI",
            "HIV",
            "NASH",
            "Varices",
            "Spleno",
            "PHT",
            "PVT",
            "Metastasis",
            "Hallmark",
            "Age",
            "Grams_day",
            "Packs_year",
            "PS",
            "Encephalopathy",
            "Ascites",
            "INR",
            "AFP",
            "MCV",
            "Leucocytes",
            "ALT",
            "AST",
            "TP",
            "Creatinine",
            "Major_Dim",
            "Sat",
            "Class",
        ]
    #=========================================================================
    num_vars = [
        "Age",
        "Grams_day",
        "Packs_year",
        "INR",
        "AFP",
        "Hemoglobin",
        "MCV",
        "Leucocytes",
        "Platelets",
        "Albumin",
        "Total_Bil",
        "ALT",
        "AST",
        "GGT",
        "ALP",
        "TP",
        "Creatinine",
        "Major_Dim",
        "Dir_Bil",
        "Iron",
        "Sat",
        "Ferritin",
        "Nodule*",
        "PS", 
        "Encephalopathy", 
        "Ascites",
        ]
    #=========================================================================
    cat_vars = [
        "Gender",
        "Symptoms",
        "Alcohol",
        "HBsAg",
        "HBeAg",
        "HBcAb",
        "HCVAb",
        "Cirrhosis",
        "Endemic",
        "Smoking",
        "Diabetes",
        "Obesity",
        "Hemochro",
        "AHT",
        "CRI",
        "HIV",
        "NASH",
        "Varices",
        "Spleno",
        "PHT",
        "PVT",
        "Metastasis",
        "Hallmark",
        "Class",
        
        ]
    #=========================================================================
    var_dict = {
        "Gender": "0=female|1=male",
        "HBsAg": "Hepatitis B surface Antigen",
        "HBeAg": "Hepatitis B e Antigen",
        "HBcAb": "Hep B core Antibody",
        "HCVAb" : "Hep C Virus Antibody",
        "Endemic" : "Endemic Countries",
        "Hemochro" : "Hemochromatosis",
        "AHT" : "Arterial Hypertension",
        "CRI" : "Chronic Renal Insufficiency",
        "HIV" : "Human Immunodeficiency Virus",
        "NASH" : "Nonalcoholic Steatohepatitis",
        "Varices" : "Esophageal Varices",
        "Spleno" :  "Splenomegaly",
        "PHT" : "Portal Hypertension",
        "PVT" : "Portal Vein Thrombosis",
        "Metastasis" : "Liver Metastasis",
        "Hallmark" : "Radiological Hallmark",
        "Age" : "Age at diagnosis",
        "Grams_day" : "Grams of Alcohol per day",
        "Packs_year" : "Packs of cigarets per year",
        "PS" : "Performance Status:\n0=Active\n1=Restricted\n2=Ambulatory\n3=Selfcare\n4=Disabled\n5=Dead",
        "Encephalopathy" : "Encephalopathy degree:\n1=None\n2=Grade I/II\n3=Grade III/IV",
        "Ascites" : "Ascites degree:\n1=None\n2=Mild\n3=Moderate to Severe",
        "INR" : "International Normalised Ratio",
        "AFP" : "Alpha-Fetoprotein (ng/mL)",
        "Hemoglobin" : "(g/dL)",
        "MCV" : "Mean Corpuscular Volume (fl)",
        "Leucocytes" : "(G/L)",
        "Platelets" : "(G/L)",
        "Albumin" : "(mg/dL)",
        "Total_Bil" : "Total Bilirubin (mg/dL)",
        "ALT" : "Alanine transaminase (U/L)",
        "AST" : "Aspartate transaminase (U/L)",
        "GGT" : "Gamma glutamyl transferase (U/L)",
        "ALP" : "Alkaline phosphatase (U/L)",
        "TP" : "Total Proteins (g/dL)",
        "Creatinine" : "(mg/dL)",
        "Nodule" : "Number of Nodules",
        "Major_Dim" : "Major dimension of nodule (cm)",
        "Dir_Bil" : "Direct Bilirubin (mg/dL)",
        "Iron" : "(mcg/dL)",
        "Sat" : "Oxygen Saturation (%)",
        "Ferritin" : "(ng/mL)",
        "Class" : "1=lives\n0=dies\n@ 1 year",
        }
    #=========================================================================
    plot_dict = {
        "Gender": "0=female|1=male",
        "HBsAg": "Hepatitis B surface Antigen",
        "HBeAg": "Hepatitis B e Antigen",
        "HBcAb": "Hep B core Antibody",
        "HCVAb" : "Hep C Virus Antibody",
        "Endemic" : "Endemic Countries",
        "Hemochro" : "Hemochromatosis",
        "AHT" : "Arterial Hypertension",
        "CRI" : "Chronic Renal Insufficiency",
        "HIV" : "Human Immunodeficiency Virus",
        "NASH" : "Nonalcoholic Steatohepatitis",
        "Varices" : "Esophageal Varices",
        "Spleno" :  "Splenomegaly",
        "PHT" : "Portal Hypertension",
        "PVT" : "Portal Vein Thrombosis",
        "Metastasis" : "Liver Metastasis",
        "Hallmark" : "Radiological Hallmark",
        "Age" : "Age at diagnosis",
        "Grams_day" : "Grams of Alcohol per day",
        "Packs_year" : "Packs of cigarets per year",
        "PS" : "Performance Status:\n0=Active\n1=Restricted\n2=Ambulatory\n3=Selfcare\n4=Disabled\n5=Dead",
        "Encephalopathy" : "Encephalopathy degree:\n1=None\n2=Grade I/II\n3=Grade III/IV",
        "Ascites" : "Ascites degree:\n1=None\n2=Mild\n3=Moderate to Severe",
        "INR" : "International Normalised Ratio",
        "AFP" : "Alpha-Fetoprotein (ng/mL)",
        "Hemoglobin" : "(g/dL)",
        "MCV" : "Mean Corpuscular Volume (fl)",
        "Leukocytes" : "(G/L)",
        "Platelets" : "(G/L)",
        "Albumin" : "(mg/dL)",
        "Total_Bil" : "Total Bilirubin (mg/dL)",
        "ALT" : "Alanine transaminase (U/L)",
        "AST" : "Aspartate transaminase (U/L)",
        "GGT" : "Gamma glutamyl transferase (U/L)",
        "ALP" : "Alkaline phosphatase (U/L)",
        "TP" : "Total Proteins (g/dL)",
        "Creatinine" : "(mg/dL)",
        "Nodules" : "Number of Nodules",
        "Major Dim" : "Major dimension of nodule (cm)",
        "Dir Bil" : "Direct Bilirubin (mg/dL)",
        "Iron" : "(mcg/dL)",
        "Sat" : "Oxygen Saturation (%)",
        "Ferritin" : "(ng/mL)",
        "Class" : "1=lives\n0=dies\n@ 1 year",
        }
    #=========================================================================
    cats = [
            "Gender",
            "Symptoms",
            "Alcohol",
            "HBsAg",
            "HBeAg",
            "HBcAb",
            "HCVAb",
            "Cirrhosis",
            "Endemic",
            "Smoking",
            "Diabetes",
            "Obesity",
            "Hemochro",
            "AHT",
            "CRI",
            "HIV",
            "NASH",
            "Varices",
            "Spleno",
            "PHT",
            "PVT",
            "Metastasis",
            "Hallmark",
            "Class",
            "PS",
            "Encephalopathy",
            "Ascites",
            "Nodule",
        ]
    #=========================================================================
    # load data and impute missing values
    df = load_data()
    # run K-medoid cluster model w/ K = 5
    df, group_df, clusters, dmat = kmed_cluster(df, 5)
    st.sidebar.title("Model")
    
    option = st.sidebar.selectbox("Model Options", ("Objective", "Data", "Cluster Analysis", "Cluster Predict", "Source"))
    #=========================================================================
    if option == "Data":
        st.subheader("Dataset")
        st.write(df.head(10))
        st.write(f"Number of samples: {df.shape[0]}")
        # st.write()
        st.subheader("Variables")
        st.write(
            """
            \n1. Gender 
            \n\t\t0=female|1=male
            \n2. Symptoms
            \n3. Alcohol
            \n4. HBsAg - Hep B surface Antigen
            \n5. HBeAg - Hep B e Antigen
            \n6. HBcAb - Hep B core Antibody
            \n7. HCVAb - Hep C Virus Antibody
            \n8. Cirrhosis
            \n9. Endemic Countries
            \n10. Smoking
            \n11. Diabetes
            \n12. Obesity
            \n13. Hemochromatosis
            \n14. AHT - Arterial Hypertension
            \n15. CRI - Chronic Renal Insufficiency
            \n16. HIV - Human Immunodeficiency Virus
            \n17. NASH - Nonalcoholic Steatohepatitis
            \n18. Varices - Esophageal Varices
            \n19. Spleno - Splenomegaly
            \n20. PHT - Portal Hypertension
            \n21. PVT - Portal Vein Thrombosis
            \n22. Metastasis - Liver Metastasis
            \n23. Hallmark - Radiological Hallmark
            \n24. Age - Age at diagnosis
            \n25. Grams/day - Grams of Alcohol per day
            \n26. Packs/year - Packs of cigarets per year
            \n27. PS - Performance Status 
                    \n\t\t[0=Active;1=Restricted;2=Ambulatory;3=Selfcare;4=Disabled;5=Dead]
            \n28. Encephalopathy - Encephalopathy degree
                    \n\t\t[1=None;2=Grade I/II; 3=Grade III/IV]
            \n29. Ascites - Ascites degree
                    \n\t\t[1=None;2=Mild;3=Moderate to Severe]
            \n30. INR - International Normalised Ratio
            \n31. AFP - Alpha-Fetoprotein (ng/mL)
            \n32. Hemoglobin (g/dL)
            \n33. MCV - Mean Corpuscular Volume (fl)
            \n34. Leukocytes(G/L)	
            \n35. Platelets	(G/L)
            \n36. Albumin (mg/dL)
            \n37. Total Bilirubin(mg/dL)
            \n38. ALT - Alanine transaminase (U/L)
            \n39. AST - Aspartate transaminase (U/L)
            \n40. GGT - Gamma glutamyl transferase (U/L)
            \n41. ALP - Alkaline phosphatase (U/L)
            \n42. TP - Total Proteins (g/dL)
            \n43. Creatinine (mg/dL)
            \n44. Nodules - Number of Nodules
            \n45. Major Dim - Major dimension of nodule (cm)
            \n46. Dir Bil - Direct Bilirubin (mg/dL)
            \n47. Iron	(mcg/dL)
            \n48. Sat - Oxygen Saturation (%)
            \n49. Ferritin (ng/mL)
            \n50. Class (1=lives;0=dies) @ 1 year
            """
            )
    #=========================================================================
    if option == "Source":
        st.write(
            """
            Data Set Name: 
            \nHepatocellular Carcinoma Dataset (HCC dataset)

            \n\nAbstract: 
            \nHepatocellular Carcinoma dataset (HCC dataset) was collected at a University Hospital in Portugal. It contains real clinical data of 165 patients diagnosed with HCC.

            \n\nDonors:
            \nMiriam Seoane Santos (miriams@student.dei.uc.pt) and Pedro Henriques Abreu (pha@dei.uc.pt), Department of Informatics Engineering, Faculty of Sciences and Technology, University of Coimbra
            . Armando Carvalho (aspcarvalho@gmail.com) and Adélia Simão (adeliasimao@gmail.com), Internal Medicine Service, Hospital and University Centre of Coimbra

            \n\nData Type: Multivariate
            \nTask: Classification, Regression, Clustering, Casual Discovery
            \nAttribute Type: Categorical, Integer and Real

            \n\nArea: Life Sciences
            \n\nFormat Type: Matrix
            \n\nMissing values: Yes

            \n\nInstances and Attributes:
            \nNumber of Instances (records in your data set): 165
            \nNumber of attributes (fields within each record): 49

            \n\nRelevant Information:
            \nHCC dataset was obtained at a University Hospital in Portugal and contais several demographic, risk factors, laboratory and overall survival features of 165 real patients diagnosed with HCC. The dataset contains 49 features selected according to the EASL-EORTC (European Association for the Study of the Liver - European Organisation for Research and Treatment of Cancer) Clinical Practice Guidelines, which are the current state-of-the-art on the management of HCC.

            \n\nThis is an heterogeneous dataset, with 23 quantitative variables, and 26 qualitative variables. Overall, missing data represents 10.22% of the whole dataset and only eight patients have complete information in all fields (4.85%). The target variables is the survival at 1 year, and was encoded as a binary variable: 0 (dies) and 1 (lives). A certain degree of class-imbalance is also present (63 cases labeled as “dies” and 102 as “lives”).

            \n\nA detailed description of the HCC dataset (feature’s type/scale, range, mean/mode and missing data percentages) is provided in Santos et al. “A new cluster-based oversampling method for improving survival prediction of hepatocellular carcinoma patients”, Journal of biomedical informatics, 58, 49-59, 2015.
            """
        )
    #=========================================================================
    analysis_dict = {
        "Ferritin" : "- ~12-112% lower mean levels\n- stratified level distirbution compared to other clusters",
        "Iron" : "- significantly higher iron level distribution in high risk cluster\n- lower iron level distributions in medium risk clusters",
        "Dir_Bil" : "- higher levels in risk clusters 0 and 1\n- similar levels in risk clusters 2-4",
        "Major_Dim" : "indistinct",
        "Gender": "",
        "HBsAg": "",
        "HBeAg": "",
        "HBcAb": "",
        "HCVAb" : "",
        "Endemic" : "",
        "Hemochro" : "",
        "AHT" : "",
        "CRI" : "",
        "HIV" : "",
        "NASH" : "",
        "Varices" : "",
        "Spleno" :  "",
        "PHT" : "",
        "PVT" : "",
        "Metastasis" : "",
        "Hallmark" : "",
        "Age" : "",
        "Grams_day" : "",
        "Packs_year" : "",
        "PS" : "",
        "Encephalopathy" : "risk clusters 0,1 have even distributions throughout; 2-4 heavy grouping around 1",
        "Ascites" : "risk clusters 0,1 have even distributions throughout; 2-4 heavy grouping around 1",
        "INR" : "risk cluster 4 has a distribution between the others; 2,3 being lower, 0,1 being higher",
        "AFP" : "cluster 4, 2 have different distributions from other clusters",
        "Hemoglobin" : "- elevated in high risk groups",
        "MCV" : "inconclusive",
        "Leucocytes" : "- same relative pattern as platelet var",
        "Platelets" : "- clusters 0,3 are low/high risk clusters respectively yet have similar platelet levels\n- cluster 2 has highest platelet levels",
        "Albumin" : "- elevated levels in high risk groups",
        "Total_Bil" : "- risk clusters 2-4 all show bottom heavy total bilirubin levels\n- clusters 2 has similar levels yet is a lower risk group",
        "ALT" : "inconclusive",
        "AST" : "inconclusive",
        "GGT" : "- higher risk groups have bottom heavy level distributions",
        "ALP" : "- higher risk groups have bottom heavy level distributions",
        "TP" : "inconclusive",
        "Creatinine" : "indistinct",
        "Nodule" : "- higher risk cluster, bottom heavy distribution",
        }
    #=========================================================================
    if option == "Cluster Analysis":
        st.subheader("Cluster Analysis")
        st.write(f"Data Class Mean: {df.Class.mean()}")
        # plot_type = st.sidebar.radio("Plot Type", ["Boxplot", "Violin"])

        # options for overall chart options
        st.sidebar.subheader("Chart Options:")
        if st.sidebar.checkbox("View Dataframe", False):
            st.write(df)
        if st.sidebar.checkbox("Gradient", False):
            group_df
        if st.sidebar.checkbox("Counts", False):
            st.write(df.kmed.value_counts())
        if st.sidebar.checkbox("Plot Clusters", False):
            df_copy = df.copy()

        for cat in cats:
            df_copy[cat] = df_copy[cat].astype(str)
        # fig, ax = plt.subplots()
        model = prince.FAMD()
        famd = model.fit(df_copy)
        coordinates = famd.transform(df_copy)

        famd.plot_row_coordinates(df_copy, color_labels=df_copy.kmed)
        st.pyplot()

        # option for var cluster visualization
        plot_var = st.sidebar.selectbox("Variable", analysis_vars)
        # option for viewing information from analysis var dict about var analysis notes
        notes = st.sidebar.checkbox("Cluster Notes")
        
        if plot_var in num_vars:
            st.sidebar.header("Plot Type:")
            box = st.sidebar.checkbox("Boxplot")
            violin = st.sidebar.checkbox("Violin Plot")
            hist = st.sidebar.checkbox("Histogram")
            if hist:
                cluster_num = st.sidebar.selectbox("Cluster #", df.kmed.unique())
            if st.sidebar.button("Plot", False):
                st.subheader(plot_var)
                st.write(var_dict[plot_var])
                if notes:
                    st.subheader("Analysis Notes")
                    st.write(analysis_dict[plot_var])
                if violin:
                    plot_violin(plot_var)
                if box:
                    plot_boxplot(plot_var)
                if hist:
                    st.write(f"Cluster {cluster_num}, {plot_var}")
                    plot_hist(plot_var, cluster_num)

                # if plot_var == "AFP":
                    # st.image(\U'c:\Users\tayma\github\hcc_clustering\afp_table.png')
                
        else:
            if st.sidebar.button("Plot", False):
                plot_barplot(plot_var)
        info_var = st.sidebar.selectbox("Var Info", cols)
    #=========================================================================
    if option == "Objective":
        st.subheader("Clustering Model Objective")
        st.write(
            '''
            The Hepatocellular Carcinoma dataset (HCC dataset) was collected at a University Hospital in Portugal. It contains real clinical data of 165 patients diagnosed with HCC. 
            The purpose of my clustering model will be to create clusters that have a distribution of the mortality class average that will allow for cluster analysis to identify 
            features of interest used in determining the effect on mortality rates in patients with HCC.
            '''
        )
        st.subheader("End User Value:")
        st.write(
            '''
            The value in the clustering analysis of the HCC dataset is providing additional insight into the ideal characteristics that compose the clusters having a lower overall 
            mortality rate relative to the higher mortality groups.
            '''
        )
        st.subheader("Quantifiable Results:")
        st.write(
            '''
            Results are assessed based on the distribution of mortality rates across cluster groups. The dataset provides a target variable; however, clustering is done without 
            introducing the target variable to retain efficacy for clustering of future data. The metric will be primarily the mortality class distribution as the silhouette scores, 
            and other clustering metrics are ineffective with this particular dataset.
            '''
        )
        st.subheader("Visuals:")
        st.write(
            '''
            I use FAMD dimensionality reduction to visualize clusters. I chose FAMD because of the balance of categorical and quantitative variables in the dataset.
            '''
        )
        st.subheader("Results:")
        st.write(
            '''
            The objective is to find what distinguishing markers make up the higher and lower mortality groups easily distinguished from the rest of the clusters. The ideal 
            number of groups (3-6) because of the need for a distinguishable range of mortality averages for each cluster showing a distinctive pattern across groups.
            '''
        )
    #=========================================================================
    if option == "Cluster Predict":
        st.subheader("Cluster Predictor")
        df = df.drop(columns=["Class", "kmed"])
        cols = df.columns.to_list()
        cols_list = []
        input = []
        for i in range(len(cols)):
            # cols[i]
            cols_list.append(cols[i])
            if cols[i] in yes_no:
                col = st.radio(cols[i], ["yes", "no"])
                if col == "yes":
                    col = 1
                else:
                    col = 0
                input.append(col)
                
            elif cols[i] in cat_cols:
                st.write(var_dict[cols[i]])
                col = st.slider(cols[i], min_value=df[cols[i]].min(), max_value=df[cols[i]].max())
                input.append(col)

            # for col in num_cols:
            elif cols[i] in num_cols:
                col = st.number_input(cols[i])
                input.append(col)
            else:
                col = st.radio(cols[i], ["male", "female"])
                if col == "male":
                    col = 1
                else:
                    col = 0
                input.append(col)
        # input
        # input = [
            # 1,
            # 0,
            # 0,
            # 0,
            # 0,
            # 0,
            # 0,
            # 0,
            # 0,
            # 0,
            # 0,
            # 0,
            # 0,
            # 0,
            # 0,
            # 0,
            # 0,
            # 0,
            # 0,
            # 0,
            # 0,
            # 0,
            # 0,
            # 26,
            # 13,
            # 0,
            # 1,
            # 2,
            # 1,
            # 1.15,
            # 165,
            # 12.5,
            # 94,
            # 950,
            # 73000,
            # 3.64,
            # 1.86,
            # 49,
            # 59,
            # 142,
            # 154,
            # 10.45,
            # 0.97,
            # 2,
            # 5.2,
            # 0.58,
            # 99,
            # 47,
            # 340
            # ]
        input_df = pd.DataFrame(input)
        input_df = input_df.T
        input_df.columns = cols_list
        # input_df
        
        # input
        X = input_df
        k = 5
        if st.button("Predict Cluster", False):
            # output = kmed_predict(df, input_df)
            # output.head(-1)
            # def kmed_predict(df, X, k=5):
            # generate k random indices from distance matrix
            # df = df.drop(columns="Class") 
            # X = X.drop(columns="kmed")
            # X = pd.DataFrame(X, columns=df.columns)
            # X
            # df
            df = df.append(X, ignore_index=True)
            # df
            X_scaled = scale_df(df)
            dmat = create_dmat(X_scaled)

            np.random.seed(42)
            n_rows = dmat.shape[0]
            init_medoids = np.random.randint(0, n_rows, k)
            
            # init_medoids
            kmed = kmedoids(dmat, initial_index_medoids=init_medoids, data_type="distance_matrix")
            kmed.process()

            medoid_idxs = kmed.get_medoids()
            # medoid_idxs

            labels = kmed.predict(dmat)
            df["kmed"] = labels
            # print(df.kmed.value_counts())
            # group_df = df.groupby("kmed").mean().sort_values("Class").style.background_gradient()

            # casting kmed clusters to strings
            df.kmed = df.kmed.astype(str)

            # reordering cluster numbers by mortality rate
            df.loc[(df.kmed == "3"), "kmed"] = 4
            df.loc[(df.kmed == "0"), "kmed"] = 2
            df.loc[(df.kmed == "1"), "kmed"] = 3
            df.loc[(df.kmed == "4"), "kmed"] = 1
            df.loc[(df.kmed == "2"), "kmed"] = 0
            # clust = [2, 4, 0, 1, 3]
            # label = kmed.predict(X)
            # X['kmed'] = clust.index(label)
            group_df = df.groupby("kmed").mean().style.background_gradient()
            # counts = df.kmed.value_counts().index.sort_values(ascending=False)
            # group_df["count"] = counts
            output = df.iloc[165]
            st.write(f"Risk Cluster: {output['kmed']}")
            st.dataframe(output)
            
            # return df



if __name__ == '__main__':
    main()