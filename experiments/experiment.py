import os
import argparse
import uuid
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.decomposition import PCA
import configparser
import pandas as pd
from yellowbrick.cluster import KElbowVisualizer

import sys
sys.path.append('..')

from data.data_util import cluster_analysis, cluster_data, plot_difference_gender_age, plot_kde, process_data, sample_data

class FeatureExtractorAndClustering:
    def __init__(self, fraction_samples=1, save_feature_extraction=False, pca=False, tsne=False, dataset='nih', n_components=2, cluster=None, config_path="../config.ini", n_clusters=0):
        """
        FeatureExtractorAndClustering class for extracting features, performing dimensionality reduction,
        and clustering on a given dataset.

        Args:
            fraction_samples (float): Fraction of samples to use, ranging from 0 to 1.
            pca (bool): Flag indicating whether to use Principal Component Analysis (PCA).
            tsne (bool): Flag indicating whether to use t-Distributed Stochastic Neighbor Embedding (t-SNE).
            dataset (str): Name of the dataset to use.
            n_components (int): Number of components to use in dimensionality reduction.
            n_clusters (int): Number of clusters to use in clustering.
            cluster (str): Cluster algorithm to use (either 'kmeans' or 'spectral').
            config_path (str): Path to the configuration file (default is "../config.ini").
        """
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.id_experiment = str(uuid.uuid4())

        self.fraction_samples = fraction_samples
        self.save_feature_extraction = save_feature_extraction
        self.pca = pca
        self.tsne = tsne
        self.dataset = dataset
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.cluster = cluster
        self.kelbow_tsne = 0
        self.kelbow_pca = 0
        self.pca_explained_varience = 0
        self.pca_explained_varience_ratio = 0
        self.run_time = 0

    def _load_model(self):
        """
        Load the TensorFlow Hub model.

        Returns:
            tf.keras.layers.Layer: TensorFlow Hub model.
        """
        print(f"Loading model from {self.config['model']['model_dir']}")
        return hub.KerasLayer(self.config['model']['model_dir'])

    def _load_data(self, analysis=True):
        """
        Load and process the dataset.

        Returns:
            pd.DataFrame: Processed dataset.
        """
        print(f"Loading data from {self.config[self.dataset]['labels']}")
        data = pd.read_csv(self.config[self.dataset]['labels'])
        if analysis:
            return process_data(data, fraction_samples=1, dataset_name=self.dataset) 
        return process_data(data, fraction_samples=self.fraction_samples, dataset_name=self.dataset)
    
    def _create_experiment_folder(self):
        """
        Create an experiment folder.
        """
        print("Creating experiment folder")
        os.mkdir(f"../data/dataset/experiment/{self.id_experiment}")

    def _verify_gpu(self):
        """
        Verify that a GPU is available.
        """
        if tf.config.list_physical_devices('GPU'):
            print("GPU is available for TensorFlow")
        else:
            print("GPU is not available, using CPU")

    def _save_index_data(self, data_index):
        """
        Save the index data.

        Args:
            data_index (numpy.ndarray): Index data.
        """
        print("Saving index data")
        np.save(f"../data/dataset/experiment/{self.id_experiment}/index_{self.dataset}_{self.id_experiment}", data_index)

    def _save_info_txt(self, data):
        """
        Save information in a text file.

        Args:
            data: Data for which information is saved.
        """
        print("Saving information in txt file")
        with open(f"../data/dataset/experiment/{self.id_experiment}/info_{self.dataset}_{self.cluster}_{self.id_experiment}.txt", "w") as f:
            f.write(f"Experiment ID: {self.id_experiment}\n")
            f.write(f"Run time: {self.run_time}\n")
            f.write(f"Dataset: {self.dataset}\n")
            f.write(f"Number of samples: {len(data)}\n")
            f.write(f"Fraction of samples: {self.fraction_samples}\n")
            f.write(f"Number of components: {self.n_components}\n")
            f.write(f"Cluster algorithm: {self.cluster}\n")
            f.write(f"TSNE: {self.tsne}\n")
            f.write(f"PCA: {self.pca}\n")
            f.write(f"TSNE Kelbow: {self.kelbow_tsne}\n")
            f.write(f"PCA Kelbow: {self.kelbow_pca}\n")
            f.write(f"Number of clusters: {self.n_clusters}\n")
            f.write(f"Cluster explained varience ratio: {np.sum(self.pca_explained_varience_ratio)}\n")
            
    def _feature_extraction(self, model, data):
        """
        Extract features from the dataset using the given model.

        Args:
            model: TensorFlow Hub model.
            data: Processed dataset.

        Returns:
            tf.Tensor: Extracted features.
        """
        processed_data = []

        for x in range(len(data)):
            img = tf.keras.preprocessing.image.load_img(data.iloc[x]['Path'], target_size=(448, 448))
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = img / 255
            img = tf.clip_by_value(img, 0., 1.)
            img = tf.expand_dims(img, 0)
            feature = model(img)
            processed_data.append(feature)

        processed_data = tf.concat(processed_data, axis=0)
        processed_data = tf.reshape(processed_data, (len(data), -1))

        if self.save_feature_extraction:
            print("Saving feature extraction")
            np.save(f"../data/dataset/experiment/{self.id_experiment}/features_{self.dataset}_{self.id_experiment}.npy", processed_data)

        return processed_data

    def _feature_extraction_fast(self, model, data, batch_size=32):
        """
        Extract features from the dataset using the given model.

        Args:
            model: TensorFlow Hub model.
            data: Processed dataset.
            batch_size: The size of each batch for processing.

        Returns:
            tf.Tensor: Extracted features.
        """

        def load_and_preprocess_image(path):
            img = tf.keras.preprocessing.image.load_img(path, target_size=(448, 448))
            img = tf.keras.preprocessing.image.img_to_array(img)
            return img / 255

        num_samples = len(data)
        all_features = []

        # Process in batches
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_images = [load_and_preprocess_image(data.iloc[i]['Path']) for i in range(start_idx, end_idx)]
            batch_images = tf.stack(batch_images)  # Convert list of images to a batch tensor
            batch_features = model(batch_images)
            all_features.append(batch_features)

        processed_data = tf.concat(all_features, axis=0)
        processed_data = tf.reshape(processed_data, (num_samples, -1))

        if self.save_feature_extraction:
            print("Saving feature extraction")
            np.save(f"../data/dataset/experiment/{self.id_experiment}/features_{self.dataset}_{self.id_experiment}.npy", processed_data)

        return processed_data

    def _kmeans_clustering(self, reduced_data, method):
        """
        Perform k-means clustering.

        Args:
            reduced_data: Data after dimensionality reduction.
            method (str): Dimensionality reduction method ('tsne' or 'pca').

        Returns:
            numpy.ndarray: Cluster labels.
            numpy.ndarray: Cluster centroids.
        """
        print(f"{self.cluster} Clustering")

        # Determine the number of clusters
        n_clusters = self.n_clusters
        if n_clusters == 0:
            model_cluster = KMeans()
            kelbow = KElbowVisualizer(model_cluster, k=(10,50))
            kelbow.fit(reduced_data)

            n_clusters = kelbow.elbow_value_
            if method == 'tsne':
                self.kelbow_tsne = n_clusters
                self.n_clusters = n_clusters
            elif method == 'pca':
                self.kelbow_pca = n_clusters
                self.n_clusters = n_clusters

        # Perform KMeans clustering
        model_cluster = KMeans(n_clusters=n_clusters)
        model_cluster.fit(reduced_data)

        return model_cluster.labels_, model_cluster.cluster_centers_

        
    def _spectral_clustering(self, reduced_data):
        """
        Perform spectral clustering.
        
        Args:
            reduced_data: Data after dimensionality reduction.
        
        Returns:
            numpy.ndarray: Cluster labels.
            None: Cluster centroids.
        """
        print(f"{self.cluster} Clustering")
        model = SpectralClustering(n_clusters=self.n_clusters)
        model.fit(reduced_data)
        labels = model.labels_
        return labels, None

    def _dbscan_clustering(self, reduced_data, eps=0.5, min_samples=5):
        """
        Perform DBSCAN clustering.

        Args:
            reduced_data: Data after dimensionality reduction.
            eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

        Returns:
            numpy.ndarray: Cluster labels.
            None: Cluster centroids (DBSCAN does not compute centroids).
        """
        print(f"{self.cluster} DBSCAN Clustering")
        model = DBSCAN(eps=eps, min_samples=min_samples)
        model.fit(reduced_data)
        labels = model.labels_

        return labels, None

    def _perform_dim_reduction_and_clustering(self, method, features):
        """
        Perform dimensionality reduction and clustering.

        Args:
            method (str): Dimensionality reduction method ('tsne' or 'pca').
            features: Extracted features.
        """
        print(f"Performing {method}")
        if method == 'tsne':
            reducer = TSNE(n_components=self.n_components, init='random', learning_rate='auto', method='exact')
        elif method == 'pca':
            reducer = PCA(n_components=self.n_components)
        
        reduced_data = reducer.fit_transform(features)
        
        if method == 'pca':
            self.pca_explained_varience = reducer.explained_variance_
            self.pca_explained_varience_ratio = reducer.explained_variance_ratio_
        
        np.save(f"../data/dataset/experiment/{self.id_experiment}/{method}_{self.dataset}_{self.id_experiment}.npy", reduced_data)

        if self.cluster == 'kmeans':
            labels, centroids = self._kmeans_clustering(reduced_data, method)
        elif self.cluster == 'spectral':
            labels, centroids = self._spectral_clustering(reduced_data)
        elif self.cluster == 'dbscan':
            labels, centroids = self._dbscan_clustering(reduced_data)
        
        if self.cluster:
            print("Saving")
            np.save(f"../data/dataset/experiment/{self.id_experiment}/labels_{self.dataset}_{method}_{self.id_experiment}.npy", labels)
        
            if centroids is not None:
                np.save(f"../data/dataset/experiment/{self.id_experiment}/centroids_{self.dataset}_{method}_{self.id_experiment}.npy", centroids)

    def analysis(self):
        """
        Perform analysis on the experiment.

        Args:
            id_experiment (str): Experiment ID.
        """
        data = self._load_data(analysis=True)
        data = cluster_data(data, self.id_experiment, dataset_name=self.dataset, pca=np.load(f"../data/dataset/experiment/{self.id_experiment}/tsne_{self.dataset}_{self.id_experiment}.npy"), tsne=np.load(f"../data/dataset/experiment/{self.id_experiment}/pca_{self.dataset}_{self.id_experiment}.npy"), cluster_type=self.cluster)
        
        _, sampled_pca = sample_data(data, 0.3, self.n_clusters, type='cluster', type_cluster='pca')
        _, sampled_tsne = sample_data(data, 0.3, self.n_clusters, type='cluster')
        _, sampled_random = sample_data(data, 0.3, 1, type='random')

        plot_difference_gender_age(sampled_pca, dataset=self.dataset, dir=f"../data/dataset/experiment/{self.id_experiment}/gender_age_pca.jpg", title=f"PCA {self.dataset} {self.cluster}", save=True)
        plot_difference_gender_age(sampled_tsne, dataset=self.dataset, dir=f"../data/dataset/experiment/{self.id_experiment}/gender_age_tsne.jpg", title=f"TSNE {self.dataset} {self.cluster}", save=True)
        plot_difference_gender_age(sampled_random, dataset=self.dataset, dir=f"../data/dataset/experiment/{self.id_experiment}/gender_age_random.jpg", title=f"RANDOM {self.dataset} {self.cluster}", save=True)

        cluster_analysis(data, save=True, dir=f"../data/dataset/experiment/{self.id_experiment}/cluster_analysis_{self.dataset}_{self.cluster}_{self.id_experiment}.jpg", title=f"{self.dataset} {self.cluster}", dataset=self.dataset)

    def run(self):
        """
        Run the experiment.
        """
        time = tf.timestamp()
        
        print(f"Running experiment with id: {self.id_experiment}")
        
        model = self._load_model()
        data = self._load_data(analysis=False)
        self._create_experiment_folder()
        self._save_index_data(data.index.values)

        features = self._feature_extraction_fast(model, data)

        if self.tsne:
            self._perform_dim_reduction_and_clustering('tsne', features)
        if self.pca:
            self._perform_dim_reduction_and_clustering('pca', features)

        self.run_time = tf.timestamp() - time
        self._save_info_txt(features)
        self.analysis()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fraction_samples', type=float, default=0, help='number of samples to use, 0 to 1')
    parser.add_argument('--save_feature_extraction', action='store_true', help='save feature extraction')
    parser.add_argument('--pca', action='store_true', help='use pca')
    parser.add_argument('--tsne', action='store_true', help='use tsne')
    parser.add_argument('--dataset', type=str, default='nih', help='dataset to use')
    parser.add_argument('--n_components', type=int, default=2, help='number of components to use in dimensionality reduction')
    parser.add_argument('--n_clusters', type=int, default=0, help='number of clusters')
    parser.add_argument('--cluster', type=str, help='cluster algorithm to use')
    args = parser.parse_args()

    feature_extractor_and_clustering = FeatureExtractorAndClustering(
        fraction_samples=args.fraction_samples,
        save_feature_extraction=args.save_feature_extraction,
        pca=args.pca,
        tsne=args.tsne,
        dataset=args.dataset,
        n_components=args.n_components,
        n_clusters=args.n_clusters,
        cluster=args.cluster
    )

    feature_extractor_and_clustering.run()
