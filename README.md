# Veg-NPP-Spatiotemporal-Graph-Scenario-Pred-XiongAn
Overview

This repository provides the source code and processed datasets for a graph-based spatiotemporal framework for township-scale vegetation net primary productivity (NPP) prediction under multiple future environmental scenarios.

The repository includes historical vegetation NPP data, historical and future environmental drivers, township-level spatial features, graph topology information, and implementations of the sequence neural network (SeqNN) and graph neural network (GNN) combinations investigated in this study, including ablation models.

The framework integrates historical NPP dynamics, spatial topological relationships, and future environmental guidance to support historical prediction experiments and long-term scenario-based forecasting.

Repository Structure

The repository is organized into three main directories:

Township-NPP-Prediction/
│
├── Feature_Class/
│
├── Graph_Data/
│
├── Code/
│
└── README.md

1. Feature_Class

This directory contains the spatial vector datasets used to construct the township-level spatial graph.

Township boundary (Polygon): Administrative boundaries of the townships within the study area.

Township centroid (Point): Geometric centroids of the township polygons.

TIN (Polyline): Triangulated irregular network (TIN) constructed from the geometric centroids.

The attribute tables of these shapefiles also contain several preliminary geometric attributes and derived spatial properties, which can be used for subsequent graph construction and edge attribute calculation.

2. Graph_Data

This directory contains the processed township-level datasets in CSV format. The time-series data were primarily derived by applying the Zonal Statistics as Table tool in ArcMap to calculate the mean raster values within each township polygon.

The datasets are organized as follows:

Historical vegetation NPP

Historical township-level NPP time series.

Historical and future environmental drivers

Historical temperature, precipitation, and potential evapotranspiration (PET).

Future temperature, precipitation, and PET projections from CMIP6 under three Shared Socioeconomic Pathways (SSPs):

SSP1-1.9: Low-emission scenario.

SSP2-4.5: Intermediate scenario.

SSP5-8.5: High-emission scenario.

Historical and future vegetation fraction

Historical township-level vegetation fractions derived from the China Land Cover Dataset (CLCD).

Future annual township-level vegetation fractions obtained from simulations using the Patch-generating Land Use Simulation (PLUS) model.

Spatial graph information

County_Edges.csv: Contains the graph edge indices and corresponding edge attributes. The first two columns represent the source and target node indices, respectively, while the remaining columns contain the associated edge attributes calculated from the geometric properties of township centroids and the TIN structure.

The graph was explicitly undirected by adding reverse edges, and self-loops were also added to the edge list.

Data processing notes:

All time-series datasets are aggregated at the township scale.

The graph nodes correspond to individual townships.

The node indexing follows the FID order of the original township polygon shapefile in Feature_Class.

The edge indices and edge attributes in County_Edges.csv follow the same row ordering.

3. Code

This directory contains the implementation of the proposed framework, including data processing, dataset construction, model implementation, training, evaluation, and future scenario prediction.

File

Description

Main.py

Main entry point for executing the experimental workflow.

DataProcess.py

Data loading and preprocessing procedures.

MakeDataset.py

Construction of training, validation, and test datasets.

GetModel.py

Model definition, initialization, loading, and saving. It includes all SeqNN-GNN combinations evaluated in this study and the corresponding ablation models.

TrainModel.py

Model training, including loss calculation, metric evaluation, and training workflow visualization.

DrawLog.py

Recording and visualization of training and validation losses and evaluation metrics throughout the training process.

ModelPred.py

Long-term rolling prediction under future environmental scenarios, including scenario data loading, preprocessing, and dataset construction. Historical training data are used as the reference for preprocessing to minimize potential data leakage.

CONFIG.py

Configuration of data preprocessing parameters and model hyperparameters.

PATH.py

Definition of file paths for data loading and result saving.

Custom layers

The Code directory also contains a separate folder with implementations of several customized neural network layers used in this study, including:

Positional embeddings for temporal feature processing in Transformer architectures.

Weighted aggregation of multiple temporal outputs into a single-step representation.

Shared N-BEATS layers across different temporal feature dimensions.

These implementations are provided to facilitate the reproduction and further development of the investigated model architectures.

Environment and Dependencies

The code was developed and tested using the following main Python packages. Similar versions are generally expected to work, although compatibility may depend on the installed PyTorch and CUDA environments.

Component

Package

Tested version

Data preprocessing

scikit-learn

1.9.1

Data preprocessing

statsmodels

0.14.5

Data preprocessing

vmdpy

0.2

Model development

PyTorch

2.7.1+cu128

Model development

PyTorch Geometric

2.5.3

Visualization

matplotlib

3.10.0

Utilities

tqdm

4.66.4

Other standard Python packages may also be required depending on the specific modules being executed.

Notes

The original remote sensing, meteorological, land-cover, and climate projection datasets are not redistributed in this repository because they are publicly accessible from their respective data providers. Their sources and acquisition details are documented in the associated manuscript.

The repository provides processed township-level datasets and spatial graph information to facilitate the reproduction of the proposed framework.

Model-specific prediction outputs for the historical test period and future SSP scenarios are not included to avoid redundant storage of extensive experimental results.

Please ensure that the file paths defined in PATH.py are correctly configured before running the scripts.

The correspondence between township identifiers, polygon FIDs, graph node indices, and edge indices should be maintained when using or modifying the provided spatial datasets.

Citation

If you use this code or the processed datasets in your research, please cite the associated publication:

Yao, S., and Jing, C. [Manuscript title]. Environmental Modelling & Software. [Publication details to be added.]

A permanent repository link and DOI will be provided once the software and datasets have been archived.

Contact

For questions regarding the code, datasets, or implementation details, please contact the corresponding author of the associated publication.
