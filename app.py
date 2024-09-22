import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from wordcloud import WordCloud
import matplotlib.cm as cm

# Set page configuration
st.set_page_config(page_title="Auto EDA App", layout="wide")

# Custom color palette
color_palette = px.colors.qualitative.Plotly

# Function to load data
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None
    return df

# Function to handle missing values
def handle_missing_values(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Impute numeric columns with median
    numeric_imputer = SimpleImputer(strategy='median')
    df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])
    
    # Impute categorical columns with mode
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])
    
    return df

# Function to encode categorical variables
def encode_categorical(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

# Data Overview
def data_overview(df):
    st.header("Dataset Info")
    
    # Display basic information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Rows", df.shape[0], delta=None, delta_color="normal")
    with col2:
        st.metric("Number of Columns", df.shape[1], delta=None, delta_color="normal")
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum(), delta=None, delta_color="inverse")
    
    # Display first few rows
    st.subheader("First Few Rows")
    st.dataframe(df.head())
    
    # Display column info
    st.subheader("Column Information")
    col_info = pd.DataFrame({
        'Data Type': df.dtypes,
        'Non-Null Count': df.notnull().sum(),
        'Null Count': df.isnull().sum(),
        'Unique Values': df.nunique()
    })
    st.dataframe(col_info.style)

# Summary Statistics
def summary_statistics(df):
    st.header("Summary Statistics")
    
    # Separate numeric and categorical columns
    numeric_df = df.select_dtypes(include=[np.number])
    categorical_df = df.select_dtypes(exclude=[np.number])
    
    if not numeric_df.empty:
        st.subheader("Numeric Columns")
        st.dataframe(numeric_df.describe())
        
        # Distribution plots for numeric columns
        st.subheader("Distribution of Numeric Columns")
        num_cols = st.multiselect("Select columns for distribution plot:", numeric_df.columns)
        if num_cols:
            fig = px.box(df, y=num_cols, points="all")
            st.plotly_chart(fig)
    
    if not categorical_df.empty:
        st.subheader("Categorical Columns")
        for col in categorical_df.columns:
            st.write(f"**{col}**")
            st.dataframe(categorical_df[col].value_counts().reset_index())
            
            # Bar plot for categorical columns
            fig = px.bar(categorical_df[col].value_counts().reset_index(), 
                         x='index', y=col, title=f'Distribution of {col}',
                         color=col, color_continuous_scale=px.colors.sequential.Reds)
            st.plotly_chart(fig)

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
            x_col = st.selectbox("Select X-axis column:", numeric_columns, key="scatter_x")
            y_col = st.selectbox("Select Y-axis column:", numeric_columns, key="scatter_y")
            
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, alpha=0.6)
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
            x_col = st.selectbox("Select categorical column for x-axis:", categorical_columns, key="bar_x")
            y_col = st.selectbox("Select numeric column for y-axis:", numeric_columns, key="bar_y")
            
            fig, ax = plt.subplots()
            sns.barplot(data=df, x=x_col, y=y_col, ax=ax, palette="bright")
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

        fig = px.imshow(corr_matrix, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig.update_layout(title_text="Correlation Heatmap", title_x=0.5)
        st.plotly_chart(fig)
        
        # Add insights and bar chart for top 5 correlations
        high_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = high_corr.stack().sort_values(ascending=False)[:5]
        st.subheader("Top 5 highest correlations:")
        
        fig = px.bar(x=[f"{var1}\n{var2}" for (var1, var2) in high_corr.index], y=high_corr.values,
                     color=high_corr.values, color_continuous_scale="Viridis")
        fig.update_layout(title_text="Top 5 Highest Correlations", title_x=0.5,
                          xaxis_title="Variable Pairs", yaxis_title="Correlation Coefficient")
        st.plotly_chart(fig)
        
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
        target_column = st.selectbox("Select the target variable:", numeric_columns)
        
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
        fig = px.bar(feature_imp, x='importance', y='feature', orientation='h',
                     color='importance', color_continuous_scale="Viridis")
        fig.update_layout(title_text="Feature Importance", title_x=0.5,
                          xaxis_title="Importance", yaxis_title="Feature")
        st.plotly_chart(fig)
        
        st.write("Feature Importance Table:")
        st.dataframe(feature_imp.style.background_gradient(cmap='viridis', subset=['importance']))
        
        st.write("**Insights:**")
        st.write(f"The most important feature is '{feature_imp.iloc[0]['feature']}' with an importance of {feature_imp.iloc[0]['importance']:.4f}.")
        st.write(f"The least important feature is '{feature_imp.iloc[-1]['feature']}' with an importance of {feature_imp.iloc[-1]['importance']:.4f}.")
        st.write("Consider focusing on the top features for model development or further analysis.")
    else:
        st.warning("Insufficient numeric columns for feature importance analysis.")

# Word Cloud
def word_cloud(df):
    st.subheader("Word Cloud")
    # Select the column you want to create the word cloud for
    column_name = st.selectbox('Select a string column:', [col for col in df.columns if df[col].dtype == 'object'])
    if column_name:
        # Create a word cloud
        wordcloud = WordCloud(width=800, height=400, colormap="viridis", background_color="white").generate(' '.join(df[column_name].astype(str).values))

        # Display the word cloud
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

# 3D Scatter Plot
def scatter_3d(df):
    st.header("3D Scatter Plot")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    x_column = st.selectbox("Select X-axis Column", numeric_columns, key='3d_scatter_x')
    y_column = st.selectbox("Select Y-axis Column", numeric_columns, key='3d_scatter_y')
    z_column = st.selectbox("Select Z-axis Column", numeric_columns, key='3d_scatter_z')
    color_column = st.selectbox("Select Color Column", ["None"] + numeric_columns, key='3d_scatter_color')
            
    if color_column == "None":
        fig = px.scatter_3d(df, x=x_column, y=y_column, z=z_column, color_discrete_sequence=color_palette)
    else:
        fig = px.scatter_3d(df, x=x_column, y=y_column, z=z_column, color=color_column, color_continuous_scale="Viridis")
            
    fig.update_layout(title_text="3D Scatter Plot", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)    

# PCA and Clustering
def pca_and_clustering(df):
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
                            marker_color=color_palette[0]))
    fig_var.add_trace(go.Scatter(x=list(range(1, len(cumulative_variance_ratio) + 1)),
                                y=cumulative_variance_ratio,
                                name='Cumulative',
                                marker_color=color_palette[1]))
    fig_var.update_layout(title='Explained Variance Ratio',
                        xaxis_title='Principal Components',
                        yaxis_title='Explained Variance Ratio',
                        title_x=0.5)
    st.plotly_chart(fig_var)

    # PCA Insights
    st.subheader("PCA Insights:")
    st.write(f"1. The first principal component explains {explained_variance_ratio[0]:.2%} of the total variance in the data.")
    st.write(f"2. To capture 80% of the variance in the data, we need {np.argmax(cumulative_variance_ratio >= 0.8) + 1} principal components.")
    st.write(f"3. To capture 95% of the variance, we need {np.argmax(cumulative_variance_ratio >= 0.95) + 1} principal components.")

    # Select number of components
    n_components = st.slider("Select number of principal components:", min_value=2, max_value=len(selected_features), value=min(3, len(selected_features)))
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

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

    # Plot Elbow curve and Silhouette scores
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(x=list(K), y=inertias, mode='lines+markers', name='Inertia', marker_color=color_palette[0]))
    fig_elbow.add_trace(go.Scatter(x=list(K), y=silhouette_scores, mode='lines+markers', name='Silhouette Score', yaxis='y2', marker_color=color_palette[1]))
    fig_elbow.update_layout(title='Elbow Method and Silhouette Scores',
                            xaxis_title='Number of Clusters (k)',
                            yaxis_title='Inertia',
                            yaxis2=dict(title='Silhouette Score', overlaying='y', side='right'),
                            title_x=0.5)
    st.plotly_chart(fig_elbow)

    # Clustering Insights
    st.subheader("Clustering Insights:")
    elbow_point = K[np.argmax(np.diff(np.diff(inertias))) + 1]
    max_silhouette = K[np.argmax(silhouette_scores)]
    st.write(f"1. The elbow method suggests {elbow_point} as an optimal number of clusters.")
    st.write(f"2. The highest silhouette score is achieved with {max_silhouette} clusters.")

    # K-means clustering
    n_clusters = st.slider("Select number of clusters (K):", min_value=2, max_value=max_clusters, value=min(elbow_point, max_silhouette))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca)

    # 3D PCA plot
    if n_components >= 3:
        fig_3d = px.scatter_3d(x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
                            color=cluster_labels,
                            labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'},
                            title="3D PCA Plot with Clusters")
        fig_3d.update_layout(title_x=0.5)
        st.plotly_chart(fig_3d)

    # 2D PCA scatter plot
    fig_2d = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=cluster_labels,
                        labels={'x': 'First Principal Component', 'y': 'Second Principal Component'},
                        title="2D PCA Scatter Plot with Clusters",
                        color_continuous_scale="Viridis")
    fig_2d.update_layout(title_x=0.5)
    st.plotly_chart(fig_2d)

    # Cluster size information
    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
    fig_sizes = px.bar(x=cluster_sizes.index, y=cluster_sizes.values, 
                    labels={'x': 'Cluster', 'y': 'Number of Data Points'},
                    title="Cluster Sizes",
                    color=cluster_sizes.index,
                    color_discrete_sequence=color_palette)
    fig_sizes.update_layout(title_x=0.5)
    st.plotly_chart(fig_sizes)

    st.subheader("Cluster Size Insights:")
    st.write(f"1. The largest cluster contains {cluster_sizes.max()} data points.")
    st.write(f"2. The smallest cluster contains {cluster_sizes.min()} data points.")
    if cluster_sizes.min() / cluster_sizes.max() < 0.1:
        st.write("3. There's a significant imbalance in cluster sizes. This could indicate outliers or important subgroups in your data.")
    else:
        st.write("3. The cluster sizes are relatively balanced, which is often a good sign in clustering analysis.")

# Main app
def main():
    st.sidebar.title("Auto EDA App")
    st.sidebar.markdown("### Upload Your Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            # Handle missing values
            df = handle_missing_values(df)
            
            # Encode categorical variables
            df_encoded = encode_categorical(df.copy())
            
            options = st.sidebar.radio("Select the option:", [
                "Dataset Info", "Summary Statistics", 
                "Univariate Analysis", "Bivariate Analysis", "Correlation Analysis", 
                "Feature Importance", "Word Cloud", "3D Scatter Plot", "PCA and Clustering"
            ])

            if options == "Dataset Info":
                data_overview(df)
            elif options == "Summary Statistics":
                summary_statistics(df_encoded)
            elif options == "Univariate Analysis":
                univariate_analysis(df_encoded)
            elif options == "Bivariate Analysis":
                bivariate_analysis(df)
            elif options == "Correlation Analysis":
                correlation_analysis(df_encoded)
            elif options == "Feature Importance":
                feature_importance(df_encoded)
            elif options == "Word Cloud":
                word_cloud(df)
            elif options == "3D Scatter Plot":
                scatter_3d(df_encoded)
            elif options == "PCA and Clustering":
                pca_and_clustering(df_encoded)
    else:
        st.info("Please upload a CSV or Excel file to begin the analysis.")
        st.markdown("""
        ### Welcome to the Auto EDA App!

        This app provides a comprehensive overview of your dataset with key statistics and visualizations.

        #### Key Features:
        1. **Dataset Info**: Get an overview of your data, including shape, data types, and memory usage.
        2. **Summary Statistics**: View detailed statistics for both numeric and categorical columns.
        3. **Univariate Analysis**: Analyze individual columns through histograms and box plots.
        4. **Bivariate Analysis**: Explore relationships between two variables using scatter plots and bar plots.
        5. **Correlation Analysis**: Visualize correlations between numeric variables with a heatmap.
        6. **Feature Importance**: Determine the importance of features using a Random Forest model.
        7. **Word Cloud**: Generate a word cloud from text data to visualize frequent terms.
        8. **3D Scatter Plot**: Create a 3D visualization of relationships between three numeric variables.
        9. **PCA and Clustering**: Perform dimensionality reduction and clustering analysis on your data.

        To get started, please upload a CSV or Excel file using the sidebar on the left.
        """)

if __name__ == "__main__":
    main()