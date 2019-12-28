# *ADnEV*: Cross-Domain Schema Matching using Deep Similarity Matrix Adjustment and Evaluation
<p align="center">
<img src ="/fig.JPG">
</p>

## Prerequisites:
1. [Anaconda 3](https://www.anaconda.com/download/)
2. [Tensorflow (or tensorflow-gpu)](https://www.tensorflow.org/install/)
3. [Keras](https://keras.io/#installation)
4. [Surprise](http://surpriselib.com/)
5. [pyFM](https://github.com/coreylynch/pyFM)

## Getting Started

### Installation:
1. Create a dataset using [ORE](https://bitbucket.org/tomers77/ontobuilder-research-environment/src) by [running VectorPrinting experiment](https://bitbucket.org/tomers77/ontobuilder-research-environment/wiki/cmd) with respect to the selected [domain of interest](https://bitbucket.org/tomers77/ontobuilder-research-environment/wiki/Datasets) and [schema matchers](https://bitbucket.org/tomers77/ontobuilder-research-environment/wiki/MatchingSystems).  
1.1 An example dataset is available for download: [Beta Dataset](https://github.com/shraga89/DSMA/blob/master/VectorsBeta.csv)
2. Clone the [SMAnE repository](https://github.com/shraga89/DSMA/)
3. Update [Config](https://github.com/shraga89/DSMA/blob/master/Config.py) with your configuration details.

### Running
1. Run [mainSaver](https://github.com/shraga89/DSMA/blob/master/mainSaver.py) to train and test your dataset using a 5-fold cross validation.  
1.1 You can also run a pre-trained model using [mainLoader](https://github.com/shraga89/DSMA/blob/master/mainLoader.py).
2. The results will appear in the [results](https://github.com/shraga89/DSMA/blob/master/results) folder, there you will find a [notebook](https://github.com/shraga89/DSMA/blob/master/results/Analyzer.ipynb) to help you analyze the results.
3. Your models will appear in the [models](https://github.com/shraga89/DSMA/blob/master/models) folder, there you will find a [notebook](https://github.com/shraga89/DSMA/blob/master/models/Visualizer.ipynb) to help you visualize the models.

## The Paper
The paper is under review.

## The Team
*Deep SMAnE* was developed at the Technion - Israel Institute of Technology by [Roee Shraga](https://sites.google.com/view/roee-shraga/) under the supervision of [Prof. Avigdor Gal](https://agp.iem.technion.ac.il/avigal/) in collaboration with Haggai Roitman from IBM Research - AI.