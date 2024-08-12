import numpy as np
import configparser
from glob import glob
from pandas import DataFrame
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load configuration from the config.ini file
config = configparser.ConfigParser()
config.read("../config.ini")


def add_file_path(data: DataFrame) -> DataFrame:
    """
    Adds the file path of images to the DataFrame based on the 'Image Index' column.

    Args:
        data (DataFrame): The input DataFrame containing at least the 'Image Index' column.

    Returns:
        DataFrame: The DataFrame with an additional 'Path' column containing the full image file paths.
    """
    all_image_paths = {
        os.path.basename(x): x for x in glob(f"{config['nih']['data_dir']}/images_*/images/*.png")
    }
    data["Path"] = data["Image Index"].map(all_image_paths.get)
    return data


def clean_data(data: DataFrame, dataset_name: str = "nih") -> DataFrame:
    """
    Cleans the dataset by filtering rows based on specific conditions for each dataset.

    Args:
        data (DataFrame): The input DataFrame containing the dataset.
        dataset_name (str, optional): The name of the dataset. Supports 'nih', 'chexpert', and 'brax'. Defaults to 'nih'.

    Returns:
        DataFrame: The cleaned DataFrame.
    """
    if dataset_name == "nih":
        data = data[data['Finding Labels'].str.contains('Atelectasis|Cardiomegaly|Edema|Consolidation|Effusion|No Finding')]
        data = data[(data['Patient Age'] > 0) & (data["Patient Age"] <= 100)]
    elif dataset_name == "chexpert":
        data = data[(data['Frontal/Lateral'] == 'Frontal') & 
                    (data[['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']].notna().any(axis=1)) &
                    (data['Sex'] != 'Unknown')]
    elif dataset_name == "brax":
        data = data[(data['ViewPosition'].isin(['PA', 'AP'])) & 
                    (data[['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']].notna().any(axis=1)) &
                    (data['PatientAge'] > 0) & (data["PatientAge"] <= 100)]
    return data


def sample(data: DataFrame, fraction_samples: float) -> DataFrame:
    """
    Randomly samples a fraction of the DataFrame.

    Args:
        data (DataFrame): The input DataFrame to be sampled.
        fraction_samples (float): The fraction of the data to sample.

    Returns:
        DataFrame: A sampled DataFrame.
    """
    return data.sample(frac=fraction_samples)


def remove_duplicates_patients(data: DataFrame, dataset_name: str = "nih") -> DataFrame:
    """
    Removes duplicate patient entries based on 'Patient ID' or 'PatientID' columns.

    Args:
        data (DataFrame): The input DataFrame.
        dataset_name (str, optional): The name of the dataset. Defaults to 'nih'.

    Returns:
        DataFrame: The DataFrame with duplicate patients removed.
    """
    if dataset_name in ["nih", "chexpert"]:
        return data.drop_duplicates(subset="Patient ID", keep='last')
    elif dataset_name == "brax":
        return data.drop_duplicates(subset="PatientID", keep='last')


def process_data(data: DataFrame, fraction_samples: int = 0, dataset_name: str = "nih") -> DataFrame:
    """
    Processes the dataset by removing duplicates, adding file paths, cleaning data, and sampling.

    Args:
        data (DataFrame): The input DataFrame.
        fraction_samples (int, optional): The fraction of data to sample. Defaults to 0.
        dataset_name (str, optional): The name of the dataset. Defaults to 'nih'.

    Returns:
        DataFrame: The processed DataFrame.
    """
    if dataset_name == "nih":
        data = remove_duplicates_patients(data, dataset_name=dataset_name)
        data = add_file_path(data)
        data = clean_data(data, dataset_name)
        if fraction_samples > 0:
            data = sample(data, fraction_samples)
    elif dataset_name == "chexpert":
        data['Patient ID'] = data['Path'].str.split('/').str[2]
        data = remove_duplicates_patients(data, dataset_name=dataset_name)
        data = clean_data(data, dataset_name)
        data['Path'] = config['chexpert']['data_dir'] + '/' + data['Path']
        if fraction_samples > 0:
            data = sample(data, fraction_samples)
    elif dataset_name == "brax":
        data = remove_duplicates_patients(data, dataset_name)
        data['PatientAge'] = data['PatientAge'].replace('85 or more', 85).astype(int)
        data = clean_data(data, dataset_name)
        data.rename(columns={"PngPath": "Path"}, inplace=True)
        data['Path'] = config['brax']['data_dir'] + '/' + data['Path']
        if fraction_samples > 0:
            data = sample(data, fraction_samples)
    return data


def select_index(data: DataFrame, id: str, dataset_name: str = "nih", tsne: np.ndarray = None) -> DataFrame:
    """
    Selects specific rows from the DataFrame based on a predefined index and optionally adds t-SNE dimensions.

    Args:
        data (DataFrame): The input DataFrame.
        id (str): The experiment ID used to load the index file.
        dataset_name (str, optional): The name of the dataset. Defaults to 'nih'.
        tsne (np.ndarray, optional): The t-SNE data to be added to the DataFrame. Defaults to None.

    Returns:
        DataFrame: The DataFrame with selected indices and optional t-SNE dimensions.
    """
    index = np.load(f"{config['general']['experiment_dir']}/{id}/index_{dataset_name}_{id}.npy")
    data = data.loc[index]

    if tsne is not None:
        dim_tsne = tsne.shape[1]
        for i in range(dim_tsne):
            data[f'tsne_{i}'] = tsne[:, i]

    return data

def cluster_data(data: DataFrame, id: str, dataset_name: str = "nih", pca: np.ndarray = None,
                 tsne: np.ndarray = None, cluster_type: str = 'kmeans',
                 labels_pca: np.ndarray = None, labels_tsne: np.ndarray = None) -> DataFrame:
    """
    Clusters data using PCA and/or t-SNE and calculates distances to centroids.

    This function loads a pre-defined index for selecting relevant rows in the data. It clusters the data 
    using PCA and/or t-SNE, and can optionally calculate distances to the cluster centroids if k-means clustering is used.

    Args:
        data (DataFrame): The input DataFrame containing data to be clustered.
        id (str): The experiment ID used to load relevant index and centroid files.
        dataset_name (str, optional): The name of the dataset. Defaults to 'nih'.
        pca (np.ndarray, optional): The PCA-transformed data. Defaults to None.
        tsne (np.ndarray, optional): The t-SNE-transformed data. Defaults to None.
        cluster_type (str, optional): The clustering method to use. Currently only 'kmeans' is supported. Defaults to 'kmeans'.
        labels_pca (np.ndarray, optional): Precomputed PCA cluster labels. If provided, these labels will be used directly. Defaults to None.
        labels_tsne (np.ndarray, optional): Precomputed t-SNE cluster labels. If provided, these labels will be used directly. Defaults to None.

    Returns:
        DataFrame: The DataFrame with added columns for cluster IDs, centroid coordinates, and distances to centroids.
    """
    # Load the predefined index and filter the data accordingly
    index = np.load(f"{config['general']['experiment_dir']}/{id}/index_{dataset_name}_{id}.npy")
    data = data.loc[index]

    # Handle PCA clustering, either using precomputed labels or computing them
    if labels_pca is not None:
        data['cluster_id_pca'] = labels_pca
    elif pca is not None:
        pca_data = np.load(f"{config['general']['experiment_dir']}/{id}/labels_{dataset_name}_pca_{id}.npy")
        data['cluster_id_pca'] = pca_data
        dim_pca = pca.shape[1]
        for i in range(dim_pca):
            data[f'pca_{i}'] = pca[:, i]

    # Handle t-SNE clustering, either using precomputed labels or computing them
    if labels_tsne is not None:
        data['cluster_id_tsne'] = labels_tsne
    elif tsne is not None:
        tsne_data = np.load(f"{config['general']['experiment_dir']}/{id}/labels_{dataset_name}_tsne_{id}.npy")
        data['cluster_id_tsne'] = tsne_data
        dim_tsne = tsne.shape[1]
        for i in range(dim_tsne):
            data[f'tsne_{i}'] = tsne[:, i]

    # If k-means clustering is specified, calculate centroids and distances to centroids
    if cluster_type == 'kmeans':
        centroids_pca = np.load(f"{config['general']['experiment_dir']}/{id}/centroids_{dataset_name}_pca_{id}.npy")
        centroids_tsne = np.load(f"{config['general']['experiment_dir']}/{id}/centroids_{dataset_name}_tsne_{id}.npy")

        # Map centroids to the data for PCA dimensions
        centroids_dict_pca = dict(zip(range(len(centroids_pca)), centroids_pca))
        for i in range(dim_pca):
            data[f'centroid_pca_{i}'] = data['cluster_id_pca'].map(centroids_dict_pca).str[i]

        # Map centroids to the data for t-SNE dimensions
        centroids_dict_tsne = dict(zip(range(len(centroids_tsne)), centroids_tsne))
        for i in range(dim_tsne):
            data[f'centroid_tsne_{i}'] = data['cluster_id_tsne'].map(centroids_dict_tsne).str[i]

        # Calculate Euclidean distances to PCA centroids
        pca_columns = [f'pca_{i}' for i in range(dim_pca)]
        centroid_pca_columns = [f'centroid_pca_{i}' for i in range(dim_pca)]
        data['distance_pca'] = np.sqrt(np.sum((data[pca_columns] - data[centroid_pca_columns]) ** 2, axis=1))

        # Calculate Euclidean distances to t-SNE centroids
        tsne_columns = [f'tsne_{i}' for i in range(dim_tsne)]
        centroid_tsne_columns = [f'centroid_tsne_{i}' for i in range(dim_tsne)]
        data['distance_tsne'] = np.sqrt(np.sum((data[tsne_columns] - data[centroid_tsne_columns]) ** 2, axis=1))

    return data

def plot_difference_gender_age(data: DataFrame, save: bool = False, dir: str = "", title: str = "", 
                               palette: str = 'mako', column_age: str = "Patient Age", column_gender: str = "Patient Gender") -> None:
    """
    Plots the age distribution of patients separated by gender using Kernel Density Estimation (KDE).

    This function generates a KDE plot that visualizes the distribution of patient ages, separated by gender.
    The plot is customizable with options to save the plot, specify a title, and adjust the color palette.

    Args:
        data (DataFrame): The input DataFrame containing at least the specified age and gender columns.
        save (bool, optional): Whether to save the plot to a file. Defaults to False.
        dir (str, optional): The directory path where the plot will be saved, if `save` is True. Defaults to an empty string.
        title (str, optional): The title of the plot. Defaults to an empty string.
        palette (str, optional): The color palette to use for the plot. Defaults to 'mako'.
        column_age (str, optional): The name of the column in the DataFrame containing patient age data. Defaults to 'Patient Age'.
        column_gender (str, optional): The name of the column in the DataFrame containing patient gender data. Defaults to 'Patient Gender'.

    Returns:
        None: The function does not return any value. It either displays the plot or saves it to a file.
    """
    # Create the KDE plot, splitting the data by gender
    ax = sns.kdeplot(
        data,
        x=column_age, hue=column_gender,
        hue_order=['Male', 'Female'],
        palette=palette,
        fill=True,
    )

    # Customize the plot appearance
    ax.set_xticks(range(0, 100, 5))  # Set the x-axis ticks for age range
    plt.title(title)  # Set the plot title
    ax.set_xlabel('Patient Age')  # Set the x-axis label
    plt.ylim(0, 0.013)  # Set the y-axis limits for the density

    # Save the plot if required
    if save:
        plt.savefig(dir)
        plt.clf()  # Clear the plot after saving
        
        

def plot_features_dimension(data: DataFrame, dimensions: int, hue: str, cluster: str, 
                            palette: str = 'mako', sickness: list = False) -> None:
    """
    Plots KDE distributions of feature dimensions, optionally separated by specific conditions (e.g., sickness).

    This function generates a series of Kernel Density Estimation (KDE) plots to visualize the distribution
    of features (e.g., PCA or t-SNE dimensions) across a specified number of dimensions. It can optionally 
    separate the data by specific conditions (e.g., sickness types) and plot these distributions side by side.

    Args:
        data (DataFrame): The input DataFrame containing the data to be plotted.
        dimensions (int): The number of dimensions to plot (e.g., PCA or t-SNE dimensions).
        hue (str): The column name in the DataFrame used to separate the data in the plot (e.g., by gender or sickness type).
        cluster (str): The prefix of the feature columns to be plotted (e.g., 'pca' or 'tsne').
        palette (str, optional): The color palette to use for the plots. Defaults to 'mako'.
        sickness (list, optional): A list of specific conditions to filter the data by (e.g., ['Atelectasis', 'Cardiomegaly']). 
                                   If provided, the function will create subplots for each condition. Defaults to False.

    Returns:
        None: The function does not return any value. It displays the generated KDE plots.
    """
    if sickness:
        # Create subplots with rows equal to the number of dimensions and columns equal to the number of sickness types
        fig, axes = plt.subplots(nrows=dimensions, ncols=len(sickness), figsize=(15, 30))

        for num, s in enumerate(sickness):
            # Create a boolean mask for the presence of the sickness in 'Finding Labels'
            data[s] = data['Finding Labels'].str.contains(s, regex=False)
            for i in range(dimensions):
                plot_data = data[data[s]]  # Filter data based on the current sickness
                ax = axes[i][num] if dimensions > 1 else axes  # Handle case when there is only one dimension
                sns.kdeplot(
                    data=plot_data,
                    x=f'{cluster}_{i}',
                    hue=s,
                    palette=palette,
                    fill=True,
                    ax=ax,
                )
                ax.set_xlabel(f'{cluster} Dimension {i}')  # Label x-axis with the dimension name
                ax.set_ylabel('Density')  # Label y-axis as Density
    else:
        # Create subplots with rows equal to the number of dimensions and one column
        fig, axes = plt.subplots(nrows=dimensions, ncols=1, figsize=(15, 30))
        
        # Loop through each dimension and create a KDE plot
        for i in range(dimensions):
            ax = axes[i] if dimensions > 1 else axes  # Handle case when there is only one dimension
            sns.kdeplot(
                data=data,
                x=f'{cluster}_{i}',
                hue=hue,
                palette=palette,
                fill=True,
                ax=ax,
            )
            ax.set_xlabel(f'{cluster} Dimension {i}')  # Label x-axis with the dimension name
            ax.set_ylabel('Density')  # Label y-axis as Density

    # Adjust the layout to prevent overlapping elements
    plt.tight_layout()
    plt.show()

def sample_data(data: DataFrame, fraction_samples: float, number_clusters: int, type: str = 'random', type_cluster: str = 'tsne') -> tuple[int, DataFrame]:
    """
    Samples data from the DataFrame either randomly or based on clusters.

    This function allows for sampling a subset of the data, either by selecting a random sample from the entire dataset or by sampling within each cluster. The sampling can be based on a specified fraction of the total data and is adjustable for different clustering methods.

    Args:
        data (DataFrame): The input DataFrame containing the data to sample.
        fraction_samples (float): The fraction of the total data to sample.
        number_clusters (int): The number of clusters in the data. This is used to determine the sample size per cluster when sampling by cluster.
        type (str, optional): The sampling method to use. Can be 'random' for random sampling or 'cluster' for sampling within each cluster. Defaults to 'random'.
        type_cluster (str, optional): The type of cluster used in the DataFrame (e.g., 'tsne' or 'pca'). This is relevant only if sampling by cluster. Defaults to 'tsne'.

    Returns:
        tuple[int, DataFrame]: A tuple where the first element is the number of samples per cluster (if applicable), and the second element is the sampled DataFrame.
    """
    if type == 'random':
        # Calculate the total number of samples to draw based on the fraction provided
        image_per_cluster = int(len(data) * fraction_samples)
        # Randomly sample the calculated number of rows from the entire DataFrame
        sampled_data = data.sample(n=image_per_cluster)
    elif type == 'cluster':
        # Calculate the number of samples to draw per cluster
        image_per_cluster = int(len(data) * fraction_samples) // number_clusters
        # Group the data by the specified cluster ID and sample within each group
        sampled_data = data.groupby(f'cluster_id_{type_cluster}').apply(
            lambda x: x.sample(len(x)) if len(x) <= image_per_cluster else x.sample(n=image_per_cluster)
        )

    return image_per_cluster, sampled_data

def cluster_analysis(data: DataFrame, save: bool = False, dir: str = "", title: str = "", dataset: str = "nih") -> None:
    """
    Generates and optionally saves a boxplot for analyzing clusters based on patient age.

    This function creates a boxplot to visualize the distribution of patient ages within each t-SNE cluster. 
    The specific columns used for plotting are based on the dataset being analyzed. The plot can be optionally 
    saved to a specified directory.

    Args:
        data (DataFrame): The input DataFrame containing the data for cluster analysis.
        save (bool, optional): Whether to save the generated plot to a file. Defaults to False.
        dir (str, optional): The directory path where the plot will be saved, if `save` is True. Defaults to an empty string.
        title (str, optional): The title of the plot. Defaults to an empty string.
        dataset (str, optional): The name of the dataset being analyzed. Supports 'nih' and 'chexpert'. Defaults to 'nih'.

    Returns:
        None: The function does not return any value. It either displays the boxplot or saves it to a file.
    """
    # Generate the boxplot based on the specified dataset
    if dataset == "nih":
        sns.boxplot(x="cluster_id_tsne", y="Patient Age", data=data)
    elif dataset == "chexpert":
        sns.boxplot(x="cluster_id_tsne", y="Age", data=data)
    
    # Set the title of the plot
    plt.title(title)

    # Save the plot to the specified directory if required
    if save:
        plt.savefig(dir)

def plot_kde(data: DataFrame, fraction_samples: float, number_clusters: int, type: str = 'random', type_cluster: str = 'tsne') -> pd.Series:
    """
    Generates KDE plots by sampling the data multiple times and returns the mean of the sampled data.

    This function samples the data multiple times according to the specified sampling method (either random or cluster-based).
    After generating samples, it calculates the mean across all sampled DataFrames. The function is useful for understanding
    the distribution of a specific column by averaging across multiple samples.

    Args:
        data (DataFrame): The input DataFrame containing the data to be sampled and plotted.
        fraction_samples (float): The fraction of the total data to sample in each iteration.
        number_clusters (int): The number of clusters in the data. Used when sampling by clusters.
        type (str, optional): The sampling method to use. Can be 'random' for random sampling or 'cluster' for cluster-based sampling. Defaults to 'random'.
        type_cluster (str, optional): The type of cluster used for grouping when sampling by clusters (e.g., 'tsne' or 'pca'). Defaults to 'tsne'.

    Returns:
        pd.Series: A Series containing the mean of the sampled data across the specified column(s).
    """
    all_sampled_data = []

    # Perform the sampling process 100 times and collect the sampled data
    for _ in range(100):
        _, sampled_data = sample_data(data, fraction_samples, number_clusters, type, type_cluster)
        all_sampled_data.append(sampled_data)

    # Calculate the mean of the sampled data across all samples
    # This assumes that the samples contain a specific column of interest (e.g., 'Age')
    mean_samples = pd.concat(all_sampled_data).mean()

    return mean_samples

def categorize_age(data: DataFrame, column_age: str = 'Patient Age', new_column: str = 'Patient Age Categorized') -> DataFrame:
    """
    Categorizes the patient age into predefined age groups.

    This function creates a new column in the DataFrame where the patient ages are categorized into specific age groups.
    The age groups are: '0-15', '15-30', '30-45', '45-60', '60-75', '75-90', and '90+'. The function applies this categorization
    to each entry in the specified age column.

    Args:
        data (DataFrame): The input DataFrame containing the age data to be categorized.
        column_age (str, optional): The name of the column containing patient age data. Defaults to 'Patient Age'.
        new_column (str, optional): The name of the new column to be created for the categorized age groups. Defaults to 'Patient Age Categorized'.

    Returns:
        DataFrame: The DataFrame with the newly added column containing the categorized age groups.
    """
    def _categorize_age(age):
        """Helper function to categorize an individual age."""
        if age < 15:
            return '0-15'
        elif 15 <= age < 30:
            return '15-30'
        elif 30 <= age < 45:
            return '30-45'
        elif 45 <= age < 60:
            return '45-60'
        elif 60 <= age < 75:
            return '60-75'
        elif 75 <= age < 90:
            return '75-90'
        else:
            return '90+'
    
    # Apply the categorization to the specified age column and create a new column for the categorized ages
    data[new_column] = data[column_age].apply(_categorize_age)
    
    return data


def plot_cluster_analysis(data: DataFrame, column_age: str, palette: str = "Set3") -> None:
    """
    Generates a boxplot to analyze the distribution of patient ages within t-SNE clusters.

    This function creates a boxplot that visualizes the distribution of patient ages across different t-SNE clusters.
    The plot is customizable with options for the color palette and the column representing patient age.

    Args:
        data (DataFrame): The input DataFrame containing the data to be plotted.
        column_age (str): The name of the column in the DataFrame that contains patient age data.
        palette (str, optional): The color palette to use for the plot. Defaults to "Set3".

    Returns:
        None: The function does not return any value. It displays the generated boxplot.
    """
    plt.figure(figsize=(10, 6))  # Set the size of the plot

    # Create a boxplot of patient age distribution across t-SNE clusters
    sns.boxplot(x="cluster_id_tsne", y=column_age, data=data, palette=palette)

    plt.ylim(0, 100)  # Set the limits for the y-axis (patient age)
    plt.xlabel("Cluster ID")  # Label the x-axis as "Cluster ID"
    plt.ylabel("Patient Age")  # Label the y-axis as "Patient Age"
    plt.show()  # Display the plot