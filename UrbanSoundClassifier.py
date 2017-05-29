import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA, PCA
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  make_scorer
from sklearn.metrics import confusion_matrix
import python_speech_features as psf
import soundfile as sf

randomState = 27


def LoadMetadata(address):
    #load metadata and return current classes
    metadata = pd.read_csv(address)
    classes = metadata['class'].unique()
    return classes, metadata


def CheckFolderConsistency(classes):
    # Check whether the data folder exists and is in the expected format
    import os
    directory = os.path.curdir + "/data/"
    for label in classes:
        if not os.path.isdir(directory + label):
            os.mkdir(directory + label)
    return


def CheckWavFiles(download):
    # check if raw wav files are present, if not may download
    if download is True:
        return DownloadFiles()
    else:
        return False


def DownloadFiles():
    # Download database files
    print "Not implemented, sorry"
    return True


def RearrangeFolders(classes, metadata, address):
    #rearrange folders on required structure
    import os
    from shutil import copyfile

    directory = os.path.dirname("F:/Drive/IA004/FinalProject/UrbanSound8K/audio/")
    print   directory
    fold = np.arange(1, 11)
    for n_fold in fold:
        foldItems = metadata.loc[metadata['fold'] == n_fold]
        source = directory + "/" + "fold" + str(n_fold) + "/"
        for index, entry in foldItems.iterrows():
            destination = address + "/" + entry['class'] + "/" + entry['slice_file_name']
            if not os.path.isfile(destination):
                print destination
                copyfile((source + entry['slice_file_name']), destination)


def ShowClassDistribution(metadata):
    #show class distributions for the current dataset
    classDistribution = metadata['class'].value_counts()
    classDistribution.plot.barh(title="Class Distribution", legend=True, colormap='hot')
    plt.show()


def TrainOnDefaultClassifier(trainData, trainLabels, testData, testLabels):
    #Used to train the data using a standard SVM classifier
    defaultClassifier = svm.SVC(random_state=randomState, kernel='linear')
    #workaround to wrong first elements
    trainLabels[0] = 0
    testLabels[0] = 0
    defaultClassifier = defaultClassifier.fit(trainData, trainLabels)
    prediction = defaultClassifier.predict(testData)
    joblib.dump( defaultClassifier, 'default_MFCC.pkl')
    return accuracy_score(testLabels, prediction), defaultClassifier


def TrainOnClassifier(trainData, trainLabels, testData, testLabels, classifier):
    #Used to train the data using a standard classifier
    trainLabels[0] = 0
    testLabels[0] = 0
    classifier = classifier.fit(trainData[:-1], trainLabels[:-1])
    prediction = classifier.predict(testData)
    joblib.dump( classifier, 'MLP_MFCC.pkl')
    return accuracy_score(testLabels, prediction), classifier


def FindBestClassifier(trainData, trainLabels, testData, testLabels):
#find the best SVM classifier using extensive grid search
    clf_svm = svm.SVC(random_state=randomState, kernel='linear')
    parameters = {'kernel': ['linear' ],
                  'C': [100, 1, .01, .001],
                  'gamma': ['auto', .01, 1, .001],
                  }
    scorer = make_scorer(accuracy_score)
    grid_obj = GridSearchCV(clf_svm, parameters, scorer, n_jobs=3)
    grid_obj = grid_obj.fit(trainData[:-1], trainLabels[:-1])
    clf_svm = grid_obj.best_estimator_
    prediction = clf_svm.predict(testData)
    joblib.dump(clf_svm, 'best_MFCC.pkl')
    print clf_svm
    return accuracy_score(testLabels, prediction), clf_svm

def FindBestMLPClassifier(trainData, trainLabels, testData, testLabels):
#find the best MLP classifier using extendive grid search. May take much time to run
    classifier = MLPClassifier(hidden_layer_sizes=(1500, 750, 500, 250, 125, 75, 50, 25), solver='adam', activation='tanh', max_iter=200  ,
                               random_state=randomState, verbose=True)

    parameters = {'hidden_layer_sizes': [(1500, 750, 500, 250, 125, 75, 50, 25) ],
                  'activation': ['relu', 'tanh', 'logistic'],
                  'solver': ['sgd', 'adam', 'lbfgs'],
                  'learning_rate': ['constant', 'invscaling', 'adaptive']
                  }
    scorer = make_scorer(accuracy_score)
    grid_obj = GridSearchCV(classifier, parameters, scorer, n_jobs=1, verbose=True)
    grid_obj = grid_obj.fit(trainData[:-1], trainLabels[:-1])
    classifier = grid_obj.best_estimator_
    classifier = classifier.fit(trainData[:-1], trainLabels[:-1])
    prediction = classifier.predict(testData)
    joblib.dump(classifier, 'bestMLP_MFCC.pkl')
    print classifier
	print "f1 score for test: {0}".format(f1_score(testLabels, prediction, average='weighted'))

    return accuracy_score(testLabels, prediction), classifier


def CalculateICAModel(trainData, n_components=5):
#Calculate ICA model with specified number of components
    fastICA = FastICA(n_components=n_components, random_state=randomState, algorithm='deflation', whiten=True, max_iter=100)
    defaultClassifier = fastICA.fit(trainData)
    joblib.dump( defaultClassifier, 'ICA_model_{0}.pkl'.format(n_components))
    return accuracy_score(testLabels, prediction)


def CalculatePCAModel(trainData, n_components):
#Calculate PCA model with specified number of components
    PCAmodel = PCA(n_components=n_components, random_state=randomState)
    PCAmodel = PCAmodel.fit(trainData)
    fileName = 'PCA_model_{0}.pkl'.format(n_components)
    joblib.dump( PCAmodel, fileName)
    return PCAmodel


def TransformFeatures(featureExtractor, dataToBeExtracted):
    print "Transforming Features"
    xExtracted = featureExtractor.transform(dataToBeExtracted)
    return xExtracted


def TrainTestValidationSeparation(metadata, train, test, validation):
#Separates the data on the metadata following the train, test and validation proportion
    trainMetadata, validationMetadata = train_test_split(metadata, test_size=validation, random_state=randomState)
#as train, test and validation values are related to all data, we need to adjust the first two on this new split
#  as they need to sum 1
    test = test / (1-validation)
    trainMetadata, testMetadata = train_test_split(trainMetadata, test_size=test)
    return trainMetadata, testMetadata, validationMetadata


def CreateFileBlock(address, blockMeta):
    fileBlock = np.empty([1, 176400])
    labelBlock = np.empty([1,1])
    currentFile = 0
    currentBlock = 1
    print blockMeta

    for index, entry in blockMeta.iterrows():
        fileLocation = address + "/" + entry['class'] + "/" + entry['slice_file_name']
        audioFile = sf.SoundFile(fileLocation)
        audioRead = audioFile.read(frames=176400, fill_value=0)

        if audioFile.channels == 2:
            audioRead = np.mean(audioRead, axis=1)
        fileBlock = np.insert(fileBlock, 0, audioRead, axis=0)
        audioFile.close()
        labelBlock = np.insert(labelBlock,0, int(entry['classID']), axis=0)
        conclusion = currentFile / float(blockMeta.shape[0])
        currentFile += 1

        if conclusion >= (2 * currentBlock * .01):
            currentBlock += 1
            print "{0} % completed".format(int(conclusion * 100))

    print "fileBlock Shape: {0}".format(np.shape(fileBlock))
    return fileBlock, labelBlock


def ExtractMFCC(address, meta):
#Extract MFCC for each file in the dataset
    currentFile = 0
    currentBlock = 1
    for index, entry in meta.iterrows():
        fileLocation = address + "/" + entry['class'] + "/" + entry['slice_file_name']
        audioFile = sf.SoundFile(fileLocation)
        audioRead = audioFile.read(frames=176400, fill_value=0)
        if audioFile.channels == 2:
            audioRead = np.mean(audioRead, axis=1)
        mfcc = psf.mfcc(audioRead, winlen=0.0025,winstep=0.001, numcep=26 )
        np.save(file=fileLocation, arr=mfcc.flatten())
        audioFile.close()
        conclusion = currentFile / float(meta.shape[0])
        currentFile += 1

        if conclusion >= (2 * currentBlock * .01):
            currentBlock += 1
            print "{0} % completed".format(int(conclusion * 100))


def CreateNpyFileBlock(address, blockMeta):
#read all numpy files and group them as a block
    fileBlock = np.empty([1, 14313])
    labelBlock = np.empty([1,1])
    currentFile = 0
    currentBlock = 1

    for index, entry in blockMeta.iterrows():
        fileLocation = address + "/" + entry['class'] + "/" + entry['slice_file_name'] + ".npy"
        mfcc_data = np.load(fileLocation)
        reading = np.zeros([1, 5187])
        fileBlock = np.insert(fileBlock, 0, mfcc_data, axis=0)
        labelBlock = np.insert(labelBlock, 0, int(entry['classID']))
        conclusion = currentFile / float(blockMeta.shape[0])
        currentFile += 1

        if conclusion >= (2 * currentBlock * .01):
            currentBlock += 1
            print "{0} % completed".format(int(conclusion * 100))

    print "fileBlock Shape: {0}".format(np.shape(fileBlock))
    return fileBlock, labelBlock

def CalculatePCAs():
    components = 300
    while components <= 1000:
        print "n_components: {0}".format( components)

        trainData = joblib.load('trainData.pkl')
        trainLabels = joblib.load('trainLabels.pkl')

        PCAmodel = CalculatePCAModel(trainData, components)
        PCAmodel = joblib.load ( 'PCA_model_{0}.pkl'.format(components))

        # print "PCA model loaded"
        train_PCA = TransformFeatures(PCAmodel, trainData)
        # print "train features extracted"
        trainData = np.array

        # testData, testLabels = CreateFileBlock(address=address, blockMeta=testMetadata)
        # joblib.dump( testData, 'testData.pkl')
        # joblib.dump( testLabels, 'testLabels.pkl')
        testData = joblib.load('testData.pkl')
        testLabels = joblib.load( 'testLabels.pkl')
        testLabels[0] = 0
        test_PCA = TransformFeatures(PCAmodel, testData)
        # print "test features extracted"

        testData = np.array
        print TrainOnDefaultClassifier(train_PCA, trainLabels, test_PCA, testLabels)
        components += 50


if __name__ == "__main__":

    address = "data/UrbanSound8K.csv"
 # load metadata and classes
    classes, metadata = LoadMetadata(address=address)

    address = "data"
    CheckFolderConsistency(classes)
    CheckWavFiles(download=True)
    RearrangeFolders(classes=classes,metadata=metadata, address="data")
    # ShowClassDistribution(metadata)

    trainMetadata, testMetadata, validationMetadata = TrainTestValidationSeparation(metadata=metadata,
                                                                                    train=0.5, test=0.3, validation=0.2)
#print data about the dataset which was just loaded
    print "Train Metadeata"
    print trainMetadata['class'].value_counts()
    print "Test Metadata"
    print testMetadata['class'].value_counts()
    print "Validation Metadata"
    print validationMetadata['class'].value_counts()

#load raw data for initial performance tests
    # trainData = np.array
    # print trainMetadata
    # trainData, trainLabels = CreateFileBlock(address=address, blockMeta=trainMetadata)
    # joblib.dump( trainData, 'trainData.pkl')
    # joblib.dump( trainLabels, 'trainLabels.pkl')

#Extract Mel-Frequency Cepstral Coefficients for the dataset
    ExtractMFCC(address, metadata)
    trainData, trainLabels = CreateNpyFileBlock(address=address, blockMeta=trainMetadata)
    joblib.dump( trainData, 'trainDataMFCC.pkl')
    joblib.dump( trainLabels, 'trainLabelsMFCC.pkl')

    testData, testLabels = CreateNpyFileBlock(address=address, blockMeta=testMetadata)
    joblib.dump( testData, 'testDataMFCC.pkl')
    joblib.dump( testLabels, 'testLabelsMFCC.pkl')

#load MFCC data if already calculated
    # trainData = joblib.load('trainDataMFCC.pkl')
    # trainLabels = joblib.load('trainLabelsMFCC.pkl')
    testData = joblib.load('testDataMFCC.pkl')
    testLabels = joblib.load( 'testLabelsMFCC.pkl')

#train on default SVM classifier
    score, model =  TrainOnDefaultClassifier(trainData, trainLabels, testData, testLabels)
    print "Train score on default SVM: {0}".format(score)
#use extensive grid search to find the best SVM model
    score, model = FindBestClassifier(trainData, trainLabels, testData, testLabels)
    print "Train score on best SVM: {0}".format(score)
#use extensive grid search to find the best MLP classifier
    score, model = FindBestMLPClassifier(trainData, trainLabels, testData, testLabels)
    print "Train score on best MLP: {0}".format(score)
#train on a default Decision tree classifier
    # classifier = tree.DecisionTreeClassifier(random_state=randomState)
    # score, model = TrainOnClassifier(trainData, trainLabels, testData, testLabels, classifier)
    # print "Train score on default Decision Tree Classifier: {0}".format(score)


#Group validation samples in one block to speed up reruns
    validationData, validationLabels = CreateNpyFileBlock(address=address, blockMeta=validationMetadata)
    joblib.dump( validationData, 'validationDataMFCC.pkl')
    joblib.dump( validationLabels, 'validationLabelsMFCC.pkl')
    validationData = joblib.load('validationDataMFCC.pkl')
    validationLabels = joblib.load('validationLabelsMFCC.pkl')
    # model = joblib.load('best_MFCC.pkl')
    validation_prediction = model.predict(validationData)
    print accuracy_score(validationLabels, validation_prediction)
    print "f1 score for validation: {0}".format(f1_score(validationLabels, validation_prediction, average='weighted'))

    confMat = confusion_matrix(validationLabels, validation_prediction)
    print confMat
    import seaborn as sns
    fig = plt.figure()
    ax = sns.heatmap(confMat)
    fig.add_axes(ax)
    plt.show()