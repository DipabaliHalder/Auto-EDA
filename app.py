from sklearn.cluster import KMeans
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.cm as cm
from wordcloud import WordCloud
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score

# Set page configuration
st.set_page_config(page_title="Auto EDA App", layout="wide")

# Function to load data
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
        df.dropna(inplace=True)
        df.drop_duplicates()
    elif file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
        df.dropna(inplace=True)
        df.drop_duplicates()
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None
    return df

# Main app
def main():
    st.sidebar.title("Auto EDA App")
    st.sidebar.markdown("### Upload Your Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            options = st.sidebar.radio("Select the option:", [
                "Dataset Info", "Summary Statistics", 
                "Univariate Analysis", "Bivariate Analysis", "Correlation Analysis", 
                "Feature Importance", "Word Cloud", "3D Scatter Plot", "PCA and Clustering"
            ])

            # Call the respective function based on the selected option
            if options == "Dataset Info":
                data_overview(df)
            elif options == "Summary Statistics":
                summary_statistics(df)
            elif options == "Univariate Analysis":
                univariate_analysis(df)
            elif options == "Bivariate Analysis":
                bivariate_analysis(df)
            elif options == "Correlation Analysis":
                correlation_analysis(df)
            elif options == "Feature Importance":
                feature_importance(df)
            elif options == "Word Cloud":
                word_cloud(df)
            elif options == "3D Scatter Plot":
                scatter_3d(df)
            elif options == "PCA and Clustering":
                pca_and_clustering(df)
    else:
        st.info("Please upload a CSV or Excel file to begin the analysis.")
        st.subheader("****Automated EDA provides a comprehensive overview of the dataset with key statistics and visualizations.****")
        st.markdown("""
        **Below are key highlights of this app...**

        - **Dataset Info**: Provides an overview of the dataset, including the first few rows, data types, and shape.

        - **Missing & Duplicate Value Check**: Identifies missing values and checks for duplicate entries in the dataset.

        - **Summary Statistics**: Displays summary statistics for numeric columns, including count, mean, median, and standard deviation.

        - **Univariate Analysis**: Analyzes individual columns through histograms and box plots, providing insights into their distributions.

        - **Bivariate Analysis**: Explores relationships between two variables using scatter plots and bar plots.

        - **Correlation Analysis**: Shows the correlation between numeric variables with a heatmap and lists the top correlations.

        - **Feature Importance**: Utilizes a Random Forest model to determine the importance of features in predicting the target variable.

        - **Word Cloud**: Generates a word cloud from a selected text column to visualize the frequency of words.

        - **3D Scatter Plot**: Creates a 3D scatter plot to visualize the relationship between three numeric variables.
        """)


# Data Overview
def data_overview(df):
    st.header("Dataset Info")
    st.write(df.head())
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Types")
        st.write(df.dtypes)
        
    with col2:
        st.subheader("Dataset Shape:")
        st.subheader(df.shape)

# Summary Statistics
def summary_statistics(df):
    st.header("Summary Statistics")
    
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        st.write(numeric_df.describe())
        
        st.subheader("Detailed Statistics")
        column = st.selectbox("Select a numeric column:", numeric_df.columns)
        col_data = numeric_df[column].dropna()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Median: {col_data.median():.2f}")
            st.write(f"Mode: {col_data.mode().values[0]:.2f}")
        with col2:
            st.write(f"Skewness: {col_data.skew():.2f}")
            st.write(f"Kurtosis: {col_data.kurtosis():.2f}")
    else:
        st.warning("No numeric columns found in the dataset.")

# Word Cloud
def word_cloud(df):
    try:
        st.subheader("Word Cloud")
        # Select the column you want to create the word cloud for
        column_name = st.selectbox('Select a string column:', [col for col in df.columns if df[col].dtype == 'object'],index=None)
        if column_name!=None:
            # Create a word cloud
            wordcloud = WordCloud(colormap="Reds",background_color="White").generate(' '.join(df[column_name].values))

            # Display the word cloud
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
    except Exception as e:
        st.warning(e)

# Univariate Analysis
def univariate_analysis(df):
    st.header("Univariate Analysis")
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        column = st.selectbox("Select a column for analysis:", numeric_columns)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Histogram")
            fig, ax = plt.subplots()
            sns.histplot(df[column].dropna(), kde=True, ax=ax, color='skyblue')
            ax.set_title(f"Histogram of {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            
            # Add insights
            mean_val = df[column].mean()
            median_val = df[column].median()
            st.write(f"Mean: {mean_val:.2f}")
            st.write(f"Median: {median_val:.2f}")
            if mean_val > median_val:
                st.write("The distribution is right-skewed.")
            elif mean_val < median_val:
                st.write("The distribution is left-skewed.")
            else:
                st.write("The distribution appears to be symmetrical.")
        
        with col2:
            st.subheader("Box Plot")
            fig, ax = plt.subplots()
            sns.boxplot(y=df[column].dropna(), ax=ax, color='lightgreen')
            ax.set_title(f"Box Plot of {column}")
            ax.set_ylabel(column)
            st.pyplot(fig)
            
            # Add insights
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            st.write(f"Interquartile Range (IQR): {iqr:.2f}")
            st.write(f"Lower Whisker: {df[column].min():.2f}")
            st.write(f"Upper Whisker: {df[column].max():.2f}")
    else:
        st.warning("No numeric columns found in the dataset.")

# Bivariate Analysis
def bivariate_analysis(df):
    st.header("Bivariate Analysis")
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    if len(numeric_columns) >= 2 and len(categorical_columns) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Scatter Plot")
            x_col = st.selectbox("Select X-axis column:", numeric_columns, key="scatter_x",index=None)
            y_col = st.selectbox("Select Y-axis column:", numeric_columns, key="scatter_y",index=None)
            

            if x_col!=None and y_col!=None:
                fig, ax = plt.subplots()
                sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, alpha=0.6,legend=False)
                ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                st.pyplot(fig)
                
                # Add insights
                correlation = df[x_col].corr(df[y_col])
                st.write(f"Correlation coefficient: {correlation:.2f}")
                if abs(correlation) < 0.3:
                    st.write("There appears to be a weak correlation between the variables.")
                elif abs(correlation) < 0.7:
                    st.write("There appears to be a moderate correlation between the variables.")
                else:
                    st.write("There appears to be a strong correlation between the variables.")
        
        with col2:
            st.subheader("Bar Plot")
            x_col = st.selectbox("Select categorical column for x-axis:", categorical_columns, key="bar_x",index=None)
            y_col = st.selectbox("Select numeric column for y-axis:", numeric_columns, key="bar_y",index=None)
            
            if x_col!=None and y_col!=None:
                fig, ax = plt.subplots()
                sns.barplot(data=df, x=x_col, y=y_col, ax=ax, palette="bright")  # Using a brighter color palette
                ax.set_title(f"Bar Plot: Average {y_col} by {x_col}")
                ax.set_xlabel(x_col)
                ax.set_ylabel(f"Average {y_col}")
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Add insights
                group_means = df.groupby(x_col)[y_col].mean().sort_values(ascending=False)
                st.write(f"Highest average {y_col}: {group_means.index[0]} ({group_means.iloc[0]:.2f})")
                st.write(f"Lowest average {y_col}: {group_means.index[-1]} ({group_means.iloc[-1]:.2f})")
                st.write(f"Range of averages: {group_means.iloc[0] - group_means.iloc[-1]:.2f}")
    else:
        st.warning("Insufficient column types for bivariate analysis. Please ensure you have at least two numeric columns and one categorical column.")

# Correlation Analysis
def correlation_analysis(df):
    st.header("Correlation Analysis")
    
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 1:
        corr_matrix = numeric_df.corr()

        col1,col2=st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
        with col2:
            st.write("**Correlation Matrix:**")
            st.write(corr_matrix)
        
        # Add insights and bar chart for top 5 correlations
        high_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = high_corr.stack().sort_values(ascending=False)[:5]
        st.subheader("Top 5 highest correlations:")
        
        col1,col2=st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(range(len(high_corr)), high_corr.values)
            cmap = cm.get_cmap('viridis')
            colors = cmap(np.linspace(0, 1, len(high_corr)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            ax.set_xticks(range(len(high_corr)))
            ax.set_xticklabels([f"{var1}\n{var2}" for (var1, var2) in high_corr.index], rotation=45, ha='right')
            ax.set_ylabel("Correlation Coefficient")
            ax.set_title("Top 5 Highest Correlations")
            
            # Add value labels on top of each bar
            for i, v in enumerate(high_corr.values):
                ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
            
            st.pyplot(fig)
        with col2:
            for (var1, var2), corr in high_corr.items():
                st.write(f"{var1} - {var2}: {corr:.2f}")
    else:
        st.warning("Insufficient numeric columns for correlation analysis.")

# Feature Importance
def feature_importance(df):
    st.header("Feature Importance")
    st.markdown("Using Random Forest model")
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    if len(numeric_columns) > 0:
        target_column = st.selectbox("Select the target variable:", numeric_columns,index=None)
        try:
            if target_column!=None:
            
                feature_columns = [col for col in numeric_columns if col != target_column]
                
                if categorical_columns.size > 0:
                    st.write("Categorical columns will be encoded for feature importance calculation.")
                
                # Prepare the data
                X = df[feature_columns]
                y = df[target_column]
                
                # Encode categorical variables
                for col in categorical_columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(df[col].astype(str))
                
                # Train a Random Forest model
                if len(np.unique(y)) > 10:  # Regression task
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                else:  # Classification task
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                
                model.fit(X, y)
                
                # Get feature importances
                importances = model.feature_importances_
                feature_imp = pd.DataFrame({'feature': X.columns, 'importance': importances})
                feature_imp = feature_imp.sort_values('importance', ascending=False)
                
                # Plot feature importances
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='importance', y='feature', data=feature_imp, ax=ax, palette='rocket')
                ax.set_title("Feature Importance")
                ax.set_xlabel("Importance")
                ax.set_ylabel("Feature")
                st.pyplot(fig)
                
                
                st.write("Feature Importance Table:")
                st.write(feature_imp)
                
                st.write("**Insights:**")
                st.write(f"The most important feature is '{feature_imp.iloc[0]['feature']}' with an importance of {feature_imp.iloc[0]['importance']:.4f}.")
                st.write(f"The least important feature is '{feature_imp.iloc[-1]['feature']}' with an importance of {feature_imp.iloc[-1]['importance']:.4f}.")
                st.write("Consider focusing on the top features for model development or further analysis.")
        except Exception as e:
            st.warning(e)
    else:
        st.warning("Insufficient numeric columns for feature importance analysis.")

def scatter_3d(df):
    st.header("3D Scatter Plot")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    x_column = st.selectbox("Select X-axis Column", numeric_columns, key='3d_scatter_x')
    y_column = st.selectbox("Select Y-axis Column", numeric_columns, key='3d_scatter_y')
    z_column = st.selectbox("Select Z-axis Column", numeric_columns, key='3d_scatter_z')
    color_column = st.selectbox("Select Color Column", ["None"] + numeric_columns, key='3d_scatter_color')
            
    if color_column == "None":
        fig = px.scatter_3d(df, x=x_column, y=y_column, z=z_column)
    else:
        fig = px.scatter_3d(df, x=x_column, y=y_column, z=z_column, color=color_column)
            
    st.plotly_chart(fig, use_container_width=True)    

def pca_and_clustering(df):
    try:
        st.header("PCA and Clustering Analysis")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) < 3:
            st.warning("Insufficient numeric columns for PCA and clustering analysis. Please ensure you have at least three numeric columns.")
            return

        st.markdown("**Feature Selection**")
        selected_features = st.multiselect("Select features for analysis:", numeric_columns, default=list(numeric_columns)[:3])
        
        if len(selected_features) < 3:
            st.warning("Please select at least three features for PCA and clustering analysis.")
            return

        X = df[selected_features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)

        # Explained variance ratio
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        # Plot explained variance ratio
        fig_var = go.Figure()
        fig_var.add_trace(go.Bar(x=list(range(1, len(explained_variance_ratio) + 1)),
                                y=explained_variance_ratio,
                                name='Individual',
                                marker_color='rgba(255, 75, 75, 0.7)'))  # Streamlit red with transparency
        fig_var.add_trace(go.Scatter(x=list(range(1, len(cumulative_variance_ratio) + 1)),
                                    y=cumulative_variance_ratio,
                                    name='Cumulative',
                                    line=dict(color='rgb(255, 0, 0)')))  # Solid Streamlit red
        fig_var.update_layout(title='Explained Variance Ratio',
                            xaxis_title='Principal Components',
                            yaxis_title='Explained Variance Ratio')
        st.plotly_chart(fig_var)

        # PCA Insights
        st.subheader("PCA Insights:")
        st.write("The explained variance ratio tells us how much of the original information in our dataset is captured by each principal component.")
        st.write(f"1. The first principal component explains {explained_variance_ratio[0]:.2%} of the total variance in the data.")
        st.write(f"2. To capture 80% of the variance in the data, we need {np.argmax(cumulative_variance_ratio >= 0.8) + 1} principal components.")
        st.write(f"3. To capture 95% of the variance, we need {np.argmax(cumulative_variance_ratio >= 0.95) + 1} principal components.")

        # Select number of components
        suggested_components = np.argmax(cumulative_variance_ratio >= 0.8) + 1
        n_components = st.slider("Select number of principal components:", min_value=2, max_value=len(selected_features), value=min(suggested_components, 10))
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        st.write(f"You've selected {n_components} principal components, which together explain {cumulative_variance_ratio[n_components-1]:.2%} of the total variance in the data.")

        # Elbow method for K-means
        max_clusters = min(10, X_pca.shape[0] - 1)
        inertias = []
        silhouette_scores = []
        K = range(2, max_clusters + 1)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_pca)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_pca, kmeans.labels_))

        # Calculate elbow point
        elbow_point = K[np.argmax(np.diff(np.diff(inertias))) + 1]

        # Plot Elbow curve and Silhouette scores
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(x=list(K), y=inertias, mode='lines+markers', name='Inertia',
                                    line=dict(color='rgb(255, 75, 75)')))  # Light Streamlit red
        fig_elbow.add_trace(go.Scatter(x=list(K), y=silhouette_scores, mode='lines+markers', name='Silhouette Score',
                                    line=dict(color='rgb(255, 0, 0)'), yaxis='y2'))  # Dark Streamlit red
        fig_elbow.update_layout(title='Elbow Method and Silhouette Scores',
                                xaxis_title='Number of Clusters (k)',
                                yaxis_title='Inertia',
                                yaxis2=dict(title='Silhouette Score', overlaying='y', side='right'))
        st.plotly_chart(fig_elbow)

        # Clustering Insights
        st.subheader("Clustering Insights:")
        max_silhouette = K[np.argmax(silhouette_scores)]
        st.write(f"1. The elbow method suggests {elbow_point} as an optimal number of clusters. This is where adding more clusters doesn't significantly reduce the inertia (within-cluster sum of squares).")
        st.write(f"2. The highest silhouette score is achieved with {max_silhouette} clusters. The silhouette score measures how similar an object is to its own cluster compared to other clusters.")
        st.write(f"3. Consider using {elbow_point} or {max_silhouette} clusters for your analysis, or experiment with values in between. The best choice may depend on your specific needs and domain knowledge.")

        # K-means clustering
        suggested_clusters = (elbow_point + max_silhouette) // 2
        n_clusters = st.slider("Select number of clusters (K):", min_value=2, max_value=max_clusters, value=suggested_clusters)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_pca)

        # 3D PCA plot
        if n_components >= 3:
            fig_3d = px.scatter_3d(x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
                                color=cluster_labels,
                                labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'},
                                title="3D PCA Plot with Clusters",
                                color_continuous_scale='Reds')  # Using shades of red
            st.plotly_chart(fig_3d)

        # 2D PCA scatter plot
        fig_2d = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=cluster_labels,
                            labels={'x': 'First Principal Component', 'y': 'Second Principal Component'},
                            title="2D PCA Scatter Plot with Clusters",
                            color_continuous_scale='Reds')  # Using shades of red
        st.plotly_chart(fig_2d)

        # Cluster size information
        cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
        fig_sizes = px.bar(x=cluster_sizes.index, y=cluster_sizes.values, 
                        labels={'x': 'Cluster', 'y': 'Number of Data Points'},
                        title="Cluster Sizes",
                        color=cluster_sizes.index,
                        color_continuous_scale='Reds')  # Using shades of red
        st.plotly_chart(fig_sizes)

        st.subheader("Cluster Size Insights:")
        st.write(f"1. The largest cluster contains {cluster_sizes.max()} data points.")
        st.write(f"2. The smallest cluster contains {cluster_sizes.min()} data points.")
        if cluster_sizes.min() / cluster_sizes.max() < 0.1:
            st.write("3. There's a significant imbalance in cluster sizes. This could indicate outliers, or that some clusters represent rare but important subgroups in your data.")
            st.write("   Consider investigating why some clusters are much smaller, or try adjusting the number of clusters.")
        else:
            st.write("3. The cluster sizes are relatively balanced, which is often a good sign in clustering analysis.")

        # Display cluster centers
        pca_components = pca.components_
        cluster_centers = kmeans.cluster_centers_
        feature_importance = pd.DataFrame(
            np.abs(pca_components.T.dot(cluster_centers.T)),
            columns=[f'Cluster {i+1}' for i in range(n_clusters)],
            index=selected_features
        )
        st.markdown("**Feature Importance in Clusters**")
        st.write(feature_importance)
        # Heatmap of cluster centers
        fig_heatmap = px.imshow(feature_importance, 
                                labels=dict(x="Clusters", y="Features", color="Absolute Importance"),
                                title="Heatmap of Feature Importance in Clusters",color_continuous_scale='Reds')
        st.plotly_chart(fig_heatmap)

        st.markdown("**Feature Importance Insights**")
        for cluster in range(n_clusters):
            top_feature = feature_importance[f'Cluster {cluster+1}'].idxmax()
            st.write(f"Cluster {cluster+1} is most influenced by the feature: {top_feature}")

        st.subheader("Overall Insights and Recommendations:")
        st.write(f"1. Dimensionality Reduction: The PCA analysis has reduced the dimensionality of the data from {len(selected_features)} to {n_components} principal components, while still explaining {cumulative_variance_ratio[n_components-1]:.2%} of the total variance.")
        st.write(f"2. Clustering: The K-means algorithm has divided the data into {n_clusters} clusters based on these principal components.")
        st.write(f"3. Optimal Clusters: The analysis suggests that the optimal number of clusters is between {elbow_point} (elbow method) and {max_silhouette} (silhouette score). You're currently using {n_clusters} clusters.")
    except Exception as e:
        st.warning("PCA does not accept missing values encoded as NaN natively. Please remove those entries and retry.")

if __name__ == "__main__":
    main()