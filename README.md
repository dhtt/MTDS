*Author: Trang Do, Naajil Aamir* 

*This project was developed under the course of Machine Learning Seminar held by Universität des Saarlandes (SoSe2019) and was submitted to the Lehrstuhl Bioinformatik - Saarbrücken on July 2019.*


# Multitask Learning for Drug Sensitivity Prediction


Anticancer drugs resistance is an enormous challenge in cancer management and research. Being of intrinsic nature or induced after a period of treatment, the unresponsiveness of tumour cells could lead to failure to eradicate neoplastic cells or disease recurrence and relapse in cured patients.
In this project, we developed a model to predict the sensitivity of tumour cell lines to multiple drugs with gene expression profiles, copy number alterations and mutation events by building neural networks on these information separately and integratively. The obtained results of single models and ensemble model were compared and hopefully would facilitate further studies on the same topic.


![Image of Input](https://github.com/dhtt/images/blob/master/input.png)
**Figure 1.** Data manipulation overflow. All the datasets went through NaN removal using the developed algorithm, feature selection from literature, and filtering based on mutual cell lines. Dimensions of initial and final output files are included.


![Image of Multitask Model](https://github.com/dhtt/images/blob/master/model.png)
**Figure 2.** Structure of Ensemble model. First, second and third branch are structure of GE, CNV and MUT best single model, respectively. Branches were concatenated and the resulting layer is followed by two more dense layers. The main output layer has 101 neurons, equal to the number of drugs to be predicted.


![Image of Result](https://github.com/dhtt/images/blob/master/result.png)
**Figure 3.** Train and validation losses of Single and Ensemble models. MSE from the training and validation of 5 best single models constructed and trained over 200 epochs on gene expression dataset (GE), copy number variation dataset (CNV), mutation dataset (MUT) and of Ensemble model (ENS) with 5 different batch sizes over 200 epochs. The training and validation loss is optimized in Ensemble model using batch size of 100. Comparison between GE and MUT shows that single models constructed on gene expression data are more robust than those on mutation
data.
