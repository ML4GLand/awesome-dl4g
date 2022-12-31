# awesome-dl4g
Community-curated list of software packages and data resources for techniques in deep learning for genomics (DL4G). Modeled after [awesome-single-cell](https://github.com/seandavi/awesome-single-cell). [Contributions welcome!](https://github.com/adamklie/awesome-dl4g/blob/main/CONTRIBUTING.md)

## Contents
- [Software packages](#software-packages)
    - [Deep learning frameworks](#deep-learning-frameworks)
    - [DL4G packages](#dl4g-packages)
    - [Data wrangling](#data-wrangling)
    - [Model zoos](#model-zoos)
    - [Visualizations](#visualizations)
    - [Interpretability](#interpretability)
- [Models](#models)
    - [Convolutional](#convolutional)
    - [Recurrent](#recurrent)
    - [Hybrid](#hybrid)
    - [Autoencoder](#autoencoder)
    - [Transformer](#transformer)
    - [Generative](#generative)
- [Datasets and databases](#datasets-and-databases)
  - [Transcriptomic]
  - [Epigenomic]
- Interpetation methods(#intepretation-methods)
  - [Neuron visualizations]
  - [Feature attributions]
  - [*In-silico*]
- [Tutorials and workflows](#tutorials-and-workflows)
- [Journal articles of general interest](#journal-articles-of-general-interest)
    - [Paper collections](#paper-collections)
    - [Experimental design](#experimental-design)
    - [Methods comparisons](#methods-comparisons)
- [Similar lists and collections](#similar-lists-and-collections)
- [People](#people)

## Software packages

### Deep learning frameworks
- [Tensorflow](https://www.tensorflow.org/) - Developed by the Google Brain team (released in 2015), has a reputation as a well-documented framework with powerful visualization tools (TensorBoard) and an abundance of trained models (TensorFlow Hub). Also known to be complex and have a steep learning curve. Often used for deploying trained models to production (TensforFlow Server). Version 2.0 was released in 2019.

- [Keras](https://keras.io/) - An API written in Python to simplify training models. Passes low-level computations to Backend library, which is often Tensorflow.

- [PyTorch](https://pytorch.org/) - Developed by Facebook AI (released in 2017), has a reputation for simplicity, ease of use, flexibility, efficient memory usage and dynamic computational graphs. Often used for prototyping models and for research.

- [PyTorch Lightning](https://www.pytorchlightning.ai/) - An API for PyTorch dsigned to reduce boilerplate PyTorch code and speed up the prototyping of models.

- [JAX](https://jax.readthedocs.io/en/latest/index.html) - JAX is Autograd and XLA, brought together for high-performance numerical computing and machine learning research

### DL4G Packages
- [DragoNN](https://kundajelab.github.io/dragonn/) - [TensorFlow] - Predictive modeling of regulatory genomics, nucleotide-resolution feature discovery, and simulations for systematic development and benchmarking. (2016)

- [pysster](https://github.com/budach/pysster) - [TensorFlow] - A Python package for training and interpretation of convolutional neural networks on biological sequence data. (2018)

- [DeepChem](https://deepchem.readthedocs.io/en/latest/index.html) - [PyTorch, TensorFlow, jax] - Open-source toolchain that democratizes the use of deep-learning in drug discovery, materials science, quantum chemistry, and biology. (2019)

- [Kipoi](https://kipoi.org/) - [PyTorch, TensorFlow] - An API and a repository of ready-to-use trained models for genomics. Also allows for usage via the command line or R. (2019)

- [Selene](https://selene.flatironinstitute.org/master/index.html) - [PyTorch] - Python library and command line interface for training deep neural networks from biological sequence data such as genomes. (2019)

- [DeepAccess](https://cgs.csail.mit.edu/deepaccess-package/) - [TensorFlow] -  Training and interpreting CNNs for predicting cell type-specific accessibility. (2021)

- [Janggu](https://janggu.readthedocs.io/en/latest/readme.html) - [Keras] - Package that facilitates deep learning in the context of genomics. Janggu provides special Genomics datasets and compatibiltity with NumPy, sklearn, and Keras. (2021)

- [GOPHER](https://github.com/shtoneyan/gopher) - [TensorFlow] - scripts for data preprocessing, training deep learning models for DNA sequence to epigenetic function prediction and evaluation of models. (2022)

- [ENNGene](https://github.com/ML-Bioinfo-CEITEC/ENNGene) - [TensorFlow] - An application that simplifies the local training of custom Convolutional Neural Network models on Genomic data via an easy to use Graphical User Interface. (2022)

- [EUGENe](https://eugene-tools.readthedocs.io/en/latest/) - [PyTorch Lightning] - An API for running DL4G workflows with sequence-to-function models. Uses SeqData to containerize sequence data and integrates functions for data loading, model training and model intereptation from several libraries (2022)

### Data wrangling
- [Nucleus](https://github.com/google/nucleus) - Library of Python and C++ code designed to make it easy to read, write and analyze data in common genomics file formats like SAM and VCF.

- [BioPython](https://biopython.org/) - Biopython is a set of freely available tools for biological computation written in Python by an international team of developers.

- [scikit-bio](http://scikit-bio.org/) - An open-source, BSD-licensed, python package providing data structures, algorithms, and educational resources for bioinformatics.

- [BioNumPy](https://bionumpy.github.io/bionumpy/) - A Python library for easy and efficient representation and analysis of biological data. (2022)

- [seqgra](https://kkrismer.github.io/seqgra/) - A deep learning pipeline that incorporates the rule-based simulation of biological sequence data and the training and evaluation of models

- [kipoiseq](https://kipoi.org/kipoiseq/) - Standard set of data-loaders for training and making predictions for DNA sequence-based models.

- [simdna](https://github.com/kundajelab/simdna) - This is a tool for generating simulated regulatory sequence for use in experiments/analyses.

- [genome-loader](https://github.com/mcvickerlab/genome-loader) - Pipeline for efficient genomic data processing.

- [PyRanges](https://pyranges.readthedocs.io/en/latest/index.html) - GenomicRanges and genomic Rle-objects for Python.

- [BedTools](https://bedtools.readthedocs.io/en/latest/) - Swiss-army knife of tools for a wide-range of genomics analysis tasks


### Model zoos
- [kipoi models](https://github.com/kipoi/models) - repository hosts predictive models for genomics and serves as a model source for [Kipoi](https://kipoi.org/groups/)

- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) - [PyTorch, TensorFlow, JAX] - Transformers provides APIs and tools to easily download and train state-of-the-art pretrained models. (2021)

### Visualizations
- [vizsequence](https://github.com/kundajelab/vizsequence) - Collecting commonly-repeated sequence visualization code here. (2019)

- [logomaker](https://logomaker.readthedocs.io/en/latest/) - a Python package for generating publication-quality sequence logos. (2019)

- [seqlogo](https://github.com/betteridiot/seqlogo) - Python port of Bioconductor's seqLogo served by WebLogo. (2020)

- [TensorBoard](https://www.tensorflow.org/tensorboard) - TensorFlow's visualization toolkit

### Interpretability
- [Captum](https://captum.ai/) - [PyTorch] - General library for model interpretability in PyTorch

- [SHAP](https://shap.readthedocs.io/en/latest/) - SHapley Additive exPlanations game theoretic approach to explain the output of any machine learning model

- [TF-MoDISco](https://github.com/jmschrei/tfmodisco-lite/) - Biological motif discovery algorithm that differentiates itself by using attribution scores from a machine learning model,

- [fastISM](https://github.com/kundajelab/fastISM) - [Keras] - Keras implementation for fast in-silico saturated mutagenesis (ISM) for convolution-based architectures

- [yuzu](https://github.com/kundajelab/yuzu) - [PyTorch] - a compressed sensing-based approach that can make in-silico saturation mutagenesis calculations on DNA, RNA, and proteins an order of magnitude faster

- [ExpectedPatternEffect](https://cgs.csail.mit.edu/deepaccess-package/interpret) - [TensorFlow] - interpretation of trained DeepAccess models

- [Global importanace analysis](https://github.com/p-koo/residualbind/blob/master/global_importance_analysis.py) - model interpretability with global importance analysis

- [Scrambler](https://github.com/johli/scrambler) - Interpretation method for sequence-predictive models based on deep generative masking

- [DFIM](https://github.com/kundajelab/dfim) - Epistatic feature interactions from neural network models of regulatory DNA sequence

### Utilities
- [MEME suite](https://meme-suite.org/meme/) - Motif-based sequence analysis tools

- [HOMER](http://homer.ucsd.edu/homer/index.html) - suite of tools for Motif Discovery and next-gen sequencing analysis

- [RayTune](https://docs.ray.io/en/latest/tune/index.html) - Python library for experiment execution and hyperparameter tuning at any scale

## Models

### Convolutional
- [DeepBind]()
- [DeepSEA]()
- [Basset]()
- [Basenji]()
- [ResidualBind]()

### Recurrent
- 

### Hybrid
- [DanQ]()
- [DeepMEL]()
- [DeepFlyBrain]()

### Autoencoder
- []()

### Transformer
- [Enformer]()

### Generative
- []()

## Datasets and databases

### Transcriptomic
- (GTEX)[]
- FANTOM5[]

### Epigenomic
- ENCODE
- Roadmap

### Chemoinformatics
- 

### Single cell
- 

## Tutorials and workflows

## Journal articles of general interest

### Paper collections

### Experimental design

### Methods comparisons

## Similar lists and collections

- [deeplearning-biology](https://github.com/hussius/deeplearning-biology)
- [awesome-deepbio](https://github.com/gokceneraslan/awesome-deepbio)
- [awesome-deep-learning-4-life-sciences](https://github.com/virtualramblas/awesome-deep-learning-4-life-sciences)

## People
