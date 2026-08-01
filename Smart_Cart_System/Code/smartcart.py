import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("smartcart_customers.csv")

#print(df.isnull().sum())

'''             Handleing missing value             '''

df["Income"] = df["Income"].fillna(df["Income"].median())


'''             Feature engineering             '''

# Age

df["Age"] = 2026 - df["Year_Birth"]

# Customer Joining Date

df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"],dayfirst=True)

refrernce_date  = df["Dt_Customer"].max()

df["Customer_Tenuer_Days"] = (refrernce_date - df["Dt_Customer"]).dt.days

# Spending

df["Total_Spending"] = df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"]  + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]

# Children

df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

# Educaton

df["Education"] = df["Education"].replace(
    {
    "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
    "Graduation": "Graduate",
    "Master": "Postgraduate", "PhD": "Postgraduate"
}
)

# Marital Status

df["Living_With"] = df["Marital_Status"].replace(
    {
    "Married": "Partner", "Together": "Partner",
    "Single": "Alone", "Divorced": "Alone",
    "Widow": "Alone", "Absurd": "Alone", "YOLO": "Alone"
}
)

'''             Drop Columns            '''

cols = ["ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", "Dt_Customer"]
spending_cols = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]

cols_to_drop = cols + spending_cols

df_cleaned = df.drop(columns=cols_to_drop)


'''             Outliers            '''

cols = ["Income", "Recency", "Response", "Age", "Total_Spending", "Total_Children"]

# relative plots of some features-  pair plots

# sns.pairplot(df_cleaned[cols])

# plt.show()

# Remove outliers

# print("data size with outliers:", len(df_cleaned))

df_cleaned = df_cleaned[ (df_cleaned["Age"] < 90) ]
df_cleaned = df_cleaned[ (df_cleaned["Income"] < 600_000) ]

# print("data size without outliers:", len(df_cleaned))

'''             Heatmap             '''

corr = df_cleaned.corr(numeric_only=True)

# plt.figure(figsize=(8, 6))

# sns.heatmap(
#     corr,
#     annot=True,
#     annot_kws={"size": 6},
#     cmap="coolwarm"
# )

'''             OneHotEncoding              '''

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()

cat_cols = ["Education", "Living_With"]

enc_cols = ohe.fit_transform(df_cleaned[cat_cols])

enc_df = pd.DataFrame(enc_cols.toarray(), columns=ohe.get_feature_names_out(cat_cols), index=df_cleaned.index)

df_encoded = pd.concat([df_cleaned.drop(columns=cat_cols),enc_df], axis=1)


'''             Scaling             '''

from sklearn.preprocessing import StandardScaler

X = df_encoded

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


'''             PCA             '''

# 2D 
from sklearn.decomposition import PCA

pca = PCA(n_components=3)

X_pca = pca.fit_transform(X_scaled)

# print(pca.explained_variance_ratio_)

# plot
# fig = plt.figure(figsize=(8, 6))

# ax = fig.add_subplot(111, projection="3d")

# ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2])

# ax.set_xlabel("PCA1")
# ax.set_ylabel("PCA2")
# ax.set_zlabel("PCA3")
# ax.set_title("3d projection")

# plt.show()

'''             Analyze K value             '''

# 1. Elbow Method

from sklearn.cluster import KMeans
from kneed import KneeLocator

wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit_predict(X_pca)
    wcss.append(kmeans.inertia_)

knee = KneeLocator(range(1, 11), wcss, curve="convex", direction="decreasing")
optimal_k = knee.elbow

# print("best k =", optimal_k)

# plot

# plt.plot(range(1, 11), wcss, marker='o')
# plt.xlabel("K")
# plt.ylabel("WCSS")
# plt.show()



# 2. Silhouette Score

from sklearn.metrics import silhouette_score

scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    scores.append(score)

# plot
# plt.plot(range(2, 11), scores, marker='o')
# plt.xlabel("K")
# plt.ylabel("Silhouette score")
# plt.show()


# combined plot

# k_range = range(2, 11)

# fig, ax1 = plt.subplots(figsize=(8, 6))

# ax1.plot(k_range, wcss[:len(k_range)], marker="o", color="blue") 
# ax1.set_xlabel("K")
# ax1.set_ylabel("WCSS")

# ax2 = ax1.twinx()
# ax2.plot(k_range, scores[:len(k_range)], marker="x", color="red", linestyle="--")
# ax2.set_ylabel("SS")

# plt.show()


'''                 clustering                '''

# K_means

kmeans = KMeans(n_clusters=4, random_state=42)
labels_kmeans = kmeans.fit_predict(X_pca)

# fig = plt.figure(figsize=(8, 6))

# ax = fig.add_subplot(111, projection="3d")

# ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels_kmeans)

# plt.show()

# Agglomerative Clustering

from sklearn.cluster import AgglomerativeClustering

agg_clf = AgglomerativeClustering(n_clusters=4, linkage="ward")
labels_agg = agg_clf.fit_predict(X_pca)

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels_agg)

# plt.show()


'''             Characterization of Clusters                '''

X["cluster"] = labels_agg

pal = ["red", "blue", "yellow", "green"]

# sns.countplot(x=X["cluster"], palette=pal, hue=X["cluster"])

# plt.show()

# Income & Spending patterns

# sns.scatterplot(x=X["Total_Spending"], y=X["Income"], hue=X["cluster"], palette=pal)

# plt.show()

# Cluster Summary

cluster_summary = X.groupby("cluster").mean()
print(cluster_summary)