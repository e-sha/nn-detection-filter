import os
import h5py
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

import sys
caffe_root = '/opt/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

def readFromFile(in_fileName):
    with open(in_fileName, "rb") as f:
        data = pickle.load(f)
    return data

def GetDatasetFileNames(in_datasetName, in_testName):
    curDir = os.getcwd()
    out_files = {"scaler": "result/" + in_datasetName + "_scaler_" + in_testName + ".p",
                 "snapshot": ""}
    for ds in ["train", "test"]:
        out_files[ds] = os.path.join(curDir, "result/", in_datasetName + "_" + ds +
                                     "_h5_list_" + in_testName + ".txt")
    return out_files

def GetNetFileNames(in_netName, in_testName):
    out_files = {}
    for ds in ["train", "test"]:
        out_files[ds] = "result/" + in_netName + "_" + ds + "_" + in_testName + ".prototxt"
    return out_files

def GetTrainLog(in_netName):
    return {"data": "result/" + in_netName + "_plot.p", "img": "result/" + in_netName + "_train.png"}

def GetNetParams(in_datasetName, in_testName, in_batchSize):
    netName = in_datasetName + "Net"
    solverFileName = "result/" + netName + "_solver_" + in_testName + ".prototxt"
    prefix = "result/" + netName + "_snapshots_" + in_testName + "/tmp"
    out_params = {"dataset": GetDatasetFileNames(in_datasetName, in_testName),
                  "netName": netName, "net": GetNetFileNames(netName, in_testName),
                  "batchSize": in_batchSize, "solver": solverFileName,
                  "snapshotPrefix": prefix, "trainLog": GetTrainLog(netName)}
    return out_params

def prefix2fileName(in_prefix, in_iter):
    return in_prefix + "_iter_" + str(in_iter) + ".caffemodel.h5"

def addSnapshotFileName(in_params, in_iter):
    in_params["net"]["snapshot"] = prefix2fileName(in_params["snapshotPrefix"], in_iter)
    return in_params["net"]["snapshot"]

def WriteNets(in_params, in_generator):
    for ds in ["train", "test"]:
        fileName = in_params["net"][ds]
        datasetFileName = in_params["dataset"][ds]
        batchSize = in_params["batchSize"][ds]
        with open(fileName, "w") as f:
            f.write(str(in_generator(datasetFileName, batchSize)))

def GetDataFiles(in_dataList):
    with open(in_dataList, "r") as f:
        fileNameArray = f.readlines()
    return [x[:-1] if x[-1] == "\n" else x for x in fileNameArray]

def getDataset(in_dataList, in_keyArray):
    fileNameArray = GetDataFiles(in_dataList)
    result = dict.fromkeys(in_keyArray)
    for fileName in fileNameArray:
        with h5py.File(fileName, "r") as H:
            for key in in_keyArray:
                curData = np.array(H.get(key))
                if (result[key] is None):
                    result[key] = curData
                else:
                    result[key] = np.vstack((result[key], curData))
    return result

def CountSamples(in_dataList):
    fileNameArray = GetDataFiles(in_dataList)
    nSamples = 0
    for fileName in fileNameArray:
        with h5py.File(fileName, "r") as H:
            label = np.array(H.get("label"))
        nSamples += label.shape[0]
    return nSamples

def CountStageSamples(in_params, in_stage):
    return CountSamples(in_params["dataset"][in_stage])

def iDivUp(in_nom, in_denom):
    return (in_nom - 1) / in_denom + 1

def CountStageBatches(in_params, in_stage):
    nSamples = CountStageSamples(in_params, in_stage)
    batchSize = in_params["batchSize"][in_stage]
    return iDivUp(nSamples, batchSize)

def WriteDatasetLists(in_params, in_fileArray, in_splitFactor):
    dataLists = {"train": [], "test": []}
    dataLists["train"], dataLists["test"] = split(in_fileArray, in_splitFactor)

    for ds in ["train", "test"]:
        outFileName = in_params["dataset"][ds]
        stageList = dataLists[ds]
        with open(outFileName, "w") as f:
            for fileName in stageList:
                f.write(fileName + "\n")

def GetLabelBounds(in_params):
    data = getDataset(in_params["dataset"]["train"], ["label"])["label"]
    scaler = readFromFile(in_params["dataset"]["scaler"])
    data = scaler["output"].inverse_transform(data)
    minArray = data.min(axis = 0).reshape(-1, 1)
    maxArray = data.max(axis = 0).reshape(-1, 1)
    return np.hstack((minArray, maxArray))

def getScalerAndNet(in_netParams):
    scaler = readFromFile(in_netParams["dataset"]["scaler"])
    net = caffe.Net(in_netParams["net"]["test"],
                    in_netParams["net"]["snapshot"], caffe.TEST)
    return scaler, net

def isScalar(in_net, in_key):
    return in_net.blobs[in_key].data.ndim == 0

def TestNet(in_net, in_data, in_outKeyArray, in_isCount = True):
    inKeyArray = in_data.keys()
    nSamples = in_data[inKeyArray[0]].shape[0]
    batchSize = in_net.blobs[inKeyArray[0]].data.shape[0]
    nBatches = iDivUp(nSamples, batchSize)

    idx = 0
    while (in_net.layers[idx].type == "HDF5Data" or in_net.layers[idx].type == "DummyData"):
        idx += 1
    startLayer = in_net._layer_names[idx]

    result = dict.fromkeys(in_outKeyArray)
    for key in in_outKeyArray:
        if (isScalar(in_net, key)):
            result[key] = np.empty(nBatches)
        else:
            result[key] = np.empty(np.hstack((nSamples, in_net.blobs[key].data.shape[1:])))

    if (in_isCount):
        for iBatch in tqdm(range(nBatches), "Net testing"):
            iStart = iBatch * batchSize
            iStop  = min(iStart + batchSize, nSamples)

            nCur = iStop - iStart
            for key in inKeyArray:
                in_net.blobs[key].data[:nCur] = in_data[key][iStart : iStop]

            in_net.forward(start = startLayer)
            for key in in_outKeyArray:
                if (isScalar(in_net, key)):
                    result[key][iBatch] = in_net.blobs[key].data
                else:
                    result[key][iStart : iStop] = in_net.blobs[key].data[:nCur]
    else:
        for iBatch in range(nBatches):
            iStart = iBatch * batchSize
            iStop  = min(iStart + batchSize, nSamples)

            nCur = iStop - iStart
            for key in inKeyArray:
                in_net.blobs[key].data[:nCur] = in_data[key][iStart : iStop]

            in_net.forward(start = startLayer)
            for key in in_outKeyArray:
                if (isScalar(in_net, key)):
                    result[key][iBatch] = in_net.blobs[key].data
                else:
                    result[key][iStart : iStop] = in_net.blobs[key].data[:nCur]
    return result

def split(in_datasetFileArray, in_testFraction):
    nSamples = len(in_datasetFileArray)
    yArray = np.arange(nSamples)
    xArray = np.zeros((nSamples, 1))
    _, _, iTrain, iVal = train_test_split(xArray, yArray,
                                          test_size=in_testFraction, random_state=42)
    sampleArray = np.array(in_datasetFileArray)
    return sampleArray[iTrain.tolist()], sampleArray[iVal.tolist()]

def storeDataset(in_listFileName, in_data, in_label):
    listDir, listFileName = os.path.split(in_listFileName)
    name, _ = os.path.splitext(listFileName)
    strArray = name.split("_")
    datasetFileName = strArray[0] + "_" + strArray[1] + "_" + strArray[-1] + ".h5"

    data, label = shuffle(in_data, in_label)
    with h5py.File(datasetFileName, 'w') as H:
        H.create_dataset('data', data = data)
        H.create_dataset('label', data = label)
    with open(in_listFileName, 'w') as L:
        L.write(datasetFileName)

def GetData(in_path, isLoad = True):
    fileNameArray = os.listdir(in_path)
    fileNameArray = [os.path.join(in_path, x) for x in fileNameArray]
    if (isLoad):
        nFiles = len(fileNameArray)
        sampleArray = None
        for idx, fileName in tqdm(enumerate(fileNameArray), "Dataset reading", nFiles):
            curData = readFromFile(fileName)
            if (sampleArray is None):
                sampleArray = curData
            else:
                sampleArray.extend(curData)
        return sampleArray
    else:
        return fileNameArray
