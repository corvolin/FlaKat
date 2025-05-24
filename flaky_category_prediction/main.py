#from select import KQ_NOTE_LINK
import numpy as np
import pandas as pd
import os

from loadData import loadRaw
from loadData import loadCode2vecEmbedding
from loadData import loadTfidfEmbedding
from loadData import loadDoc2vecEmbedding

from reduction import pcaReduction
from reduction import ldaReduction
from reduction import isomapReduction
from reduction import tsneReduction
from reduction import umapReduction

from model import getKnnCategoryAccuracy
from model import getSvmCategoryAccuracy
from model import getRFCategoryAccuracy
from model import getRFCategoryAccuracyWithBO
from model import getGBDTCategoryAccuracy
from model import getGBDTCategoryAccuracyWithBO

from analysis import generateScatterPlot
from analysis import generateScatterPlotMatrix
from analysis import prettyPrintReductionQuality
from analysis import writeMatrixToCsv
from analysis import plotMatrix
from analysis import show3dScatterPlot

from analysis import getPredictionAccuracyByKnn
from analysis import getPredictionAccuracyBySVM
from analysis import getPredictionAccuracyByRandomForest
from analysis import getPredictionAccuracyByGBDT
from analysis import sequentialModelBasedOptimizationRandomForest




def gridSearchReductions(vectorDf, outputDf, outputDir, prefix, rawDf, reducedDimension,categoryNames, colors, size):

    embeddingList = []
    embeddingNames = []

    reduced = pcaReduction(vectorDf,reducedDimension)
    embeddingList.append(reduced)
    embeddingNames.append("PCA")
    if reducedDimension == 2: 
        generateScatterPlot(outputDir, prefix + '_PCA_2d.png',reduced, rawDf, categoryNames, colors, size)
    elif reducedDimension == 3: 
        show3dScatterPlot(prefix + '_PCA_3d', reduced, rawDf, categoryNames, colors, size)

    reduced = ldaReduction(vectorDf, outputDf, reducedDimension)
    embeddingList.append(reduced)
    embeddingNames.append("LDA")
    if reducedDimension == 2: 
        generateScatterPlot(outputDir, prefix + '_LDA_2d.png',reduced, rawDf, categoryNames, colors, size)
    elif reducedDimension == 3: 
        show3dScatterPlot(prefix + '_LDA_3d', reduced, rawDf, categoryNames, colors, size) 

    n_neighbour = [2,5,10,20,50]
    for n in n_neighbour:
        reduced = isomapReduction(vectorDf, outputDf, reducedDimension, n)
        embeddingList.append(reduced)
        tempName = "isomap_n_"+str(n)
        if reducedDimension == 2: 
            generateScatterPlot(outputDir, prefix + '_' + tempName + '_2d.png',reduced, rawDf, categoryNames, colors, size)
        elif reducedDimension == 3: 
            show3dScatterPlot(prefix + '_' + tempName + '_3d', reduced, rawDf, categoryNames, colors, size)
        embeddingNames.append(tempName)

    perp = [10,25,50,100]
    for p in perp:
        reduced = tsneReduction(vectorDf, reducedDimension, p, 1000)
        embeddingList.append(reduced)
        tempName = "tsne_p_"+str(p)
        if reducedDimension == 2: 
            generateScatterPlot(outputDir, prefix + '_' + tempName + '_2d.png',reduced, rawDf, categoryNames, colors, size)
        elif reducedDimension == 3: 
            show3dScatterPlot( prefix + '_' + tempName + '_3d', reduced, rawDf, categoryNames, colors, size)
        embeddingNames.append(tempName)

    n_neighbour = [5,20,80,320]
    min_dist = [0.05,0.2,0.5,0.8]
    for n in n_neighbour:
        for d in min_dist:
            reduced = umapReduction(vectorDf, reducedDimension, n, d)
            embeddingList.append(reduced)
            tempName = "umap_n_"+str(n)+"_d_"+str(d)
            if reducedDimension == 2: 
                generateScatterPlot(outputDir, prefix + '_' + tempName + '_2d.png',reduced, rawDf, categoryNames, colors, size)
            elif reducedDimension == 3: 
                show3dScatterPlot(prefix + '_' + tempName + '_3d', reduced, rawDf, categoryNames, colors, size)
            embeddingNames.append(tempName)

def generateScatterPlotForOptimal(doc2vecDf, code2vecDf, tfidfDf, outputDf, outputDir, rawDf, categoryNames, colors, size):
    doc2vecReducedList = []
    code2vecReducedList = []
    tfidf2vecReducedList = []

    doc2vecReducedList.append(pcaReduction(doc2vecDf,2))
    doc2vecReducedList.append(ldaReduction(doc2vecDf,outputDf,2))
    doc2vecReducedList.append(isomapReduction(doc2vecDf, outputDf, 2, 5))
    doc2vecReducedList.append(tsneReduction(doc2vecDf, 2, 50, 1000))
    doc2vecReducedList.append(umapReduction(doc2vecDf, 2, 320, 0.05))

    code2vecReducedList.append(pcaReduction(code2vecDf,2))
    code2vecReducedList.append(ldaReduction(code2vecDf,outputDf,2))
    code2vecReducedList.append(isomapReduction(code2vecDf, outputDf, 2, 2))
    code2vecReducedList.append(tsneReduction(code2vecDf, 2, 50, 1000))
    code2vecReducedList.append(umapReduction(code2vecDf, 2, 320, 0.05))
    
    tfidf2vecReducedList.append(pcaReduction(tfidfDf,2))
    tfidf2vecReducedList.append(ldaReduction(tfidfDf,outputDf,2))
    tfidf2vecReducedList.append(isomapReduction(tfidfDf, outputDf, 2, 2))
    tfidf2vecReducedList.append(tsneReduction(tfidfDf, 2, 50, 1000))
    tfidf2vecReducedList.append(umapReduction(tfidfDf, 2, 320, 0.05))

    reducedMatrix = []
    reducedMatrix.append(doc2vecReducedList)
    reducedMatrix.append(code2vecReducedList)
    reducedMatrix.append(tfidf2vecReducedList)

    generateScatterPlotMatrix(outputDir, 'multiple.png',reducedMatrix, None, rawDf, categoryNames, colors, size)

# generate the accuracy plots for both f1s and fdc using the optimal reduced emebdding
def generateAccuracyPlots(doc2vecDf, code2vecDf, tfidfDf, outputDf, outputDir):
    embeddingList = []
    embeddingNames = []

    doc2vecReduced = ldaReduction(doc2vecDf, outputDf,2)
    embeddingList.append(doc2vecReduced)
    embeddingNames.append("doc2vec")

    code2vecReduced = ldaReduction(code2vecDf, outputDf, 2)
    embeddingList.append(code2vecReduced)
    embeddingNames.append("code2vec")

    tfidfReduced = ldaReduction(tfidfDf, outputDf, 2)
    embeddingList.append(tfidfReduced)
    embeddingNames.append("tfidf")

    fold = 10

    kList = [2,5,10,20,50,100,200]

    knnResult = getPredictionAccuracyByKnn(kList, fold, embeddingList, outputDf.values, True)
    writeMatrixToCsv(knnResult, embeddingNames, outputDir+'knn_fdc.csv')
    plotMatrix(knnResult,embeddingNames,kList, "knn", "fdc", outputDir)

    knnResult = getPredictionAccuracyByKnn(kList, fold, embeddingList, outputDf.values, False)
    writeMatrixToCsv(knnResult, embeddingNames, outputDir+'knn_f1s.csv')
    plotMatrix(knnResult,embeddingNames,kList, "knn", "f1s", outputDir)

    kernelList = ['linear', 'poly', 'rbf', 'sigmoid']

    svmResult = getPredictionAccuracyBySVM(kernelList, fold, embeddingList, outputDf.values, True)
    writeMatrixToCsv(svmResult, embeddingNames, outputDir+'svm_fdc.csv')
    plotMatrix(svmResult,embeddingNames, kernelList, "svm", "fdc", outputDir)

    svmResult = getPredictionAccuracyBySVM(kernelList, fold, embeddingList, outputDf.values, False)
    writeMatrixToCsv(svmResult, embeddingNames, outputDir+'svm_f1s.csv')
    plotMatrix(svmResult,embeddingNames, kernelList, "svm", "f1s",outputDir)
    
    minLeafList = [1,2,5,10,20,50,100,200,300,500]

    rfResult = getPredictionAccuracyByRandomForest(minLeafList, fold, embeddingList, outputDf.values, True)
    writeMatrixToCsv(rfResult, embeddingNames, outputDir+'randomForest_fdc.csv')
    plotMatrix(rfResult,embeddingNames, minLeafList, "rf", "fdc", outputDir)

    rfResult = getPredictionAccuracyByRandomForest(minLeafList, fold, embeddingList, outputDf.values, False)
    writeMatrixToCsv(rfResult, embeddingNames, outputDir+'randomForest_f1s.csv')
    plotMatrix(rfResult,embeddingNames, minLeafList, "rf", "f1s",outputDir)

    minLeafList = [1,2,5,10,20,50,100,200,300,500]

    gbdtResult = getPredictionAccuracyByGBDT(minLeafList, fold, embeddingList, outputDf.values, True)
    writeMatrixToCsv(gbdtResult, embeddingNames, outputDir+'GBDT_fdc.csv')
    plotMatrix(gbdtResult,embeddingNames, minLeafList, "gbdt", "fdc", outputDir)

    gbdtResult = getPredictionAccuracyByGBDT(minLeafList, fold, embeddingList, outputDf.values, False)
    writeMatrixToCsv(gbdtResult, embeddingNames, outputDir+'GBDT_f1s.csv')
    plotMatrix(gbdtResult,embeddingNames, minLeafList, "gbdt", "f1s", outputDir)



def main():
    # location of raw data
    rawCsv = 'data/input/extracted-all-projects.csv'
    # location of vectorized result from code2vec
    code2vecCsv = 'data/input/vector.csv'
    outputDir = 'data/output/'

    features = []
    code2vec_default_dimension = 384
    for i in range(code2vec_default_dimension):
        features.append('feature_'+str(i))
    col = ['id','name'] + features

    #categoryNames = ['ID','OD','UD','OD-Vic','OD-Brit','NOD','NDOD','NDOI','NonFlaky']
    #colors = ["blue", "red", "indigo", "green", "yellow", "silver", "maroon", "peru", "aqua"]
    categoryNames = ['ID','OD','NOD']
    colors = ["green", "blue", "red"]

    rawDf = loadRaw(rawCsv, True)
    outputDf = rawDf['category']
    print(outputDf.value_counts())
    code2vecDf = loadCode2vecEmbedding(code2vecCsv, col, rawDf, features, True)
    tfidfDf = loadTfidfEmbedding(rawDf, 384, 5, 0.7)
    doc2vecDf = loadDoc2vecEmbedding(rawDf, 384, 10, 1)

    print(len(code2vecDf))
    print(len(tfidfDf))
    print(len(doc2vecDf))
    

    sampling = 0

    reductionDimention = 2
    IsomapN = 5
    tSNEp = 50
    tSNEi = 1000
    UMAPn = 320
    UMAPd = 0.05

    nameEmbedding = ""
    nameSampling = ""
    nameReduction = ""
    nameScatter = ""
    nameClassifier = ""
    nameOption = ""

    fold = 10
    metrics = ['precision', 'recall', 'f1s', 'mcc', 'fdc']

    knnK=[2,3,4,5,7,10,20,30,50,70,100,150,200,250,500]
    svmKernel=['linear', 'poly', 'rbf', 'sigmoid']
    svmC = [ 0.1, 1, 5, 10]

    rfConfig={'max_depth': 53, 'max_leaf_nodes': 324, 'max_samples': 0.5096968938687001, 'min_impurity_decrease': 0.020294613947313078, 'min_samples_leaf': 70, 'min_samples_split': 203, 'min_weight_fraction_leaf': 0.00522892296925322, 'n_estimators': 133}
    # doc2vec
    #{'max_depth': 116, 'max_leaf_nodes': 286, 'max_samples': 0.6008105922992728, 'min_impurity_decrease': 0.008335864754482336, 'min_samples_leaf': 34, 'min_samples_split': 217, 'min_weight_fraction_leaf': 0.041835274717841445, 'n_estimators': 146}
    # code2vec
    #{'max_depth': 30, 'max_leaf_nodes': 125, 'max_samples': 0.5578022123369274, 'min_impurity_decrease': 0.004788117051408303, 'min_samples_leaf': 41, 'min_samples_split': 227, 'min_weight_fraction_leaf': 0.0018379269493552176, 'n_estimators': 172}
    # tfidf
    #{'max_depth': 53, 'max_leaf_nodes': 324, 'max_samples': 0.5096968938687001, 'min_impurity_decrease': 0.020294613947313078, 'min_samples_leaf': 70, 'min_samples_split': 203, 'min_weight_fraction_leaf': 0.00522892296925322, 'n_estimators': 133}
   
    gbdtConfig = {
        'learning_rate': 0.5433990284529154, 
        'max_depth': 17, 
        'min_samples_leaf': 88, 
        'min_samples_split': 305, 
        'n_estimators': 187
        }
    
    #generateScatterPlotForOptimal(doc2vecDf, code2vecDf, tfidfDf, outputDf, outputDir, rawDf, categoryNames, colors, [15,10])
    
    while True:
        cmd = input('Enter command\n')
        
        if cmd == 'exit':
            print('Exit program')
            break

        elif len(cmd) != 6:
            print('Invalid command')
        
        else:
            """
            1. embedding to use 1-doc2vec 2-code2vec 3-tfidf
            2. sampling 0-none 1-TL 2-SMOTE 3-TL&SMOTE 4-SMOTE&TL
            3. reduction 1-PCA 2-LDA 3-Isomap 4-t-SNE 5-UMAP
            4. visualization of reduced embedding 0-none 1-2d 2-3d
            5. classifier 1-KNN 2-SVM 3-RF 4-RF with tuning 5-GBDT 6-GBDT with tuning
            6. result 0-csv 1-no log

            142010
            """
            for i, c in enumerate(cmd):
                if i == 0:
                    if c == '1':
                        vectorEmbedding = doc2vecDf
                        nameEmbedding = "doc2vec"
                    elif c == '2':
                        vectorEmbedding = code2vecDf
                        nameEmbedding = "code2vec"
                    elif c == '3':
                        vectorEmbedding = tfidfDf
                        nameEmbedding = "tfidf"
                
                elif i == 1:
                    vector_val, vector_cat = vectorEmbedding, outputDf.values
                    if c == '0':
                        sampling = 0
                        nameSampling = "NoSampling"
                    elif c == '1':
                        sampling = 1
                        nameSampling = "TL"
                    elif c == '2':
                        sampling = 2
                        nameSampling = "SMOTE"
                    elif c == '3':
                        sampling = 3
                        nameSampling = "TL+SMOTE"
                    elif c == '4':
                        sampling = 4
                        nameSampling = "SMOTE+TL"

                elif i == 2:
                    if int(cmd[3]) == 0:
                        reductionDimention = 2
                    else:
                        reductionDimention = int(cmd[3])+1
                        
                    if c =='1':
                        reducedEmbedding = pcaReduction(vector_val, reductionDimention)
                        nameReduction = "PCA"
                    elif c == '2':
                        reducedEmbedding = ldaReduction(vector_val, vector_cat, reductionDimention)
                        nameReduction = "LDA"
                    elif c == '3':
                        reducedEmbedding = isomapReduction(vector_val, vector_cat, reductionDimention, IsomapN)
                        nameReduction = "Isomap"
                    elif c == '4':
                        reducedEmbedding = tsneReduction(vector_val, reductionDimention, tSNEp, tSNEi)
                        nameReduction = "tSNE"
                    elif c == '5':
                        reducedEmbedding = umapReduction(vector_val, reductionDimention, UMAPn, UMAPd)
                        nameReduction = "UMAP"

                elif i == 3:
                    if c == '0':
                        pass
                        #print("No scatter plots")
                    elif c == '1':
                        generateScatterPlot(outputDir,nameEmbedding+"_"+nameReduction+"_2d.png", reducedEmbedding, vector_cat, categoryNames, colors, [10,10])
                        nameScatter = "2"
                    elif c == '2':
                        show3dScatterPlot(nameEmbedding+"_"+nameReduction+"_3d", reducedEmbedding, vector_cat, categoryNames, colors, [10,10])
                        nameScatter = "3"

                elif i == 4:
                    if c == '1':

                        for k in knnK:
                            result = getKnnCategoryAccuracy(fold, sampling, reducedEmbedding, vector_cat, metrics, k)
                            nameClassifier = "KNN"

                            resultCsv = outputDir + nameEmbedding+"_"+nameSampling+"_"+nameReduction+"_"+nameScatter+"_"+nameClassifier+"_k="+str(k)+".csv"
                            df = pd.DataFrame(result).transpose()
                            df.to_csv(resultCsv)
                    elif c == '2':
                        for k in svmKernel:
                            for c in svmC:
                                result = getSvmCategoryAccuracy(fold, sampling, reducedEmbedding, vector_cat, metrics, k, c)
                                nameClassifier = "SVM"
                                
                                resultCsv = outputDir + nameEmbedding+"_"+nameSampling+"_"+nameReduction+"_"+nameScatter+"_"+nameClassifier+"_k="+str(k)+"_c="+str(c)+".csv"
                                df = pd.DataFrame(result).transpose()
                                df.to_csv(resultCsv)
                    elif c == '3':
                        result = getRFCategoryAccuracy(fold, sampling, reducedEmbedding, vector_cat, metrics, rfConfig)
                        nameClassifier = "RF"
                    elif c == '4':
                        result = getRFCategoryAccuracyWithBO(fold, sampling, reducedEmbedding, vector_cat, metrics)
                        nameClassifier = "RF_BO"
                    elif c == '5':
                        result = getGBDTCategoryAccuracy(fold, sampling, reducedEmbedding, vector_cat, metrics, gbdtConfig)
                        nameClassifier = "GBDT"
                    elif c == '6':
                        result = getGBDTCategoryAccuracyWithBO(fold, sampling, reducedEmbedding, vector_cat, metrics)
                        nameClassifier = "GBDT_BO"
                elif i == 5:
                    if c == '0':
                        
                        resultCsv = outputDir + nameEmbedding+"_"+nameSampling+"_"+nameReduction+"_"+nameScatter+"_"+nameClassifier+".csv"
                        df = pd.DataFrame(result).transpose()
                        df.to_csv(resultCsv)
                    elif c == '1':
                        pass
                    
        
        """
        # grid search and save results for all combinations into files
        if cmd == '1':

            vectorDf = doc2vecDf
            prefix = 'doc2vec'
            gridSearchReductions(vectorDf, outputDf, outputDir, prefix, rawDf, 2, categoryNames, colors, [15,15])

            vectorDf = code2vecDf
            prefix = 'code2vec'
            gridSearchReductions(vectorDf, outputDf, outputDir, prefix, rawDf, 2, categoryNames, colors, [15,15])

            vectorDf = tfidfDf
            prefix = 'tfidf'
            gridSearchReductions(vectorDf, outputDf, outputDir, prefix, rawDf, 2, categoryNames, colors, [15,15])

        # generate multiple 2d scatter plot for visualing reduction between each embedding and each technique
        elif cmd == '2':
            generateScatterPlotForOptimal(doc2vecDf, code2vecDf, tfidfDf, outputDf, outputDir, rawDf, categoryNames, colors, [15,10])

        # generate prediction accuracy
        elif cmd == '3':
            generateAccuracyPlots(doc2vecDf, code2vecDf, tfidfDf, outputDf, outputDir)
        
        # generate 3d scatter plots
        elif cmd == '4':

            vectorDf = doc2vecDf
            prefix = 'doc2vec'
            gridSearchReductions(vectorDf, outputDf, outputDir, prefix, rawDf, 3, categoryNames, colors, [10,10])

            vectorDf = code2vecDf
            prefix = 'code2vec'
            gridSearchReductions(vectorDf, outputDf, outputDir, prefix, rawDf, 3, categoryNames, colors, [10,10])

            vectorDf = tfidfDf
            prefix = 'tfidf'
            gridSearchReductions(vectorDf, outputDf, outputDir, prefix, rawDf, 3, categoryNames, colors, [10,10])
        
        elif cmd == '5':

            doc2vecReduced = ldaReduction(doc2vecDf, outputDf,2)
            code2vecReduced = ldaReduction(code2vecDf, outputDf,2)
            tfidfReduced = ldaReduction(tfidfDf, outputDf,2)

            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)]
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(5, 100, num = 20)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10, 20, 50, 100, 200, 500]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 5, 10, 20, 50, 100, 200]

            random_grid = {'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf}
                        
            sequentialModelBasedOptimizationRandomForest(random_grid, 100, outputDir+"doc_SMDO_FDC_result.csv", 5, doc2vecReduced, outputDf.values, 1)
            sequentialModelBasedOptimizationRandomForest(random_grid, 100, outputDir+"code_SMDO_FDC_result.csv", 5, code2vecReduced, outputDf.values, 1)
            sequentialModelBasedOptimizationRandomForest(random_grid, 100, outputDir+"tfidf_SMDO_FDC_result.csv", 5, tfidfReduced, outputDf.values, 1)

        elif cmd == 'exit':
            print('Exit program')
            break

        else:
            print('Invalid command')
        """
    

if __name__ == "__main__":
    main()