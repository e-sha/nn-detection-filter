import numpy as np
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import os

def UniqueRows(in_array):
    b = np.ascontiguousarray(in_array).view(np.dtype((
            np.void, in_array.dtype.itemsize * in_array.shape[1])))
    _, idx = np.unique(b, return_index = True)
    return in_array[idx]

def drawNorm(in_meanArray, in_stdArray, in_gt,
             in_fileTemplate = None, in_labelArray = None, in_boundArray = None):
    if (in_meanArray.ndim == 1):
        in_meanArray = in_meanArray.reshape(1, -1)
        in_stdArray  = in_stdArray.reshape(1, -1)

    nPoints = in_meanArray.shape[1]
    nDist   = in_meanArray.shape[0]
    factor  = 3.1

    fontSize = 18
    #font = {'family' : 'normal', 'weight' : 'bold', 'size'   : fontSize}
    font = {'family': 'normal', 'size': fontSize}
    matplotlib.rc('font', **font)

    lineWidth = 2

    for idx in range(nPoints):
        meanVal = in_meanArray[:, idx]
        stdVal = in_stdArray[:, idx]
        gtVal = in_gt[idx]
        if (in_boundArray is None):
            minVal = (meanVal - factor * stdVal).min()
            maxVal = (meanVal + factor * stdVal).max()
            if not gtVal is None:
                minVal = min(minVal, gtVal - stdVal.min())
                maxVal = max(maxVal, gtVal + stdVal.max())
        else:
            minVal = in_boundArray[idx, 0]
            maxVal = in_boundArray[idx, 1]

        xArray = np.linspace(minVal, maxVal, 2000)
        xArray = np.sort(np.hstack((xArray, meanVal.ravel())))

        gtValArray = np.empty(nDist)
        for iDist in range(nDist):
            curMean = meanVal[iDist]
            curStd = stdVal[iDist]
            dist = scipy.stats.norm(curMean, curStd)
            yArray = dist.pdf(xArray)
            plt.plot(xArray, yArray, linewidth = lineWidth)
            if not gtVal is None:
                gtValArray[iDist] = dist.pdf(gtVal)


        if not (gtVal is None):
            plt.plot(gtVal + np.zeros(nDist), gtValArray, "ro",
                     markerSize = 5 * lineWidth)
            plt.plot([gtVal, gtVal], [0, gtValArray.max()], "r--",
                     linewidth = lineWidth)

        if not in_labelArray is None:
            plt.legend(in_labelArray, fontsize = fontSize)

        plt.xlim([minVal, maxVal])

        if not (in_fileTemplate is None):
            fileName = in_fileTemplate.format(idx)
            plt.savefig(fileName)

        plt.show()

def conf2Var(in_conf, in_scaler = None):
    eps = 1e-6
    result = np.exp(in_conf) + eps
    if not (in_scaler is None):
        result *= in_scaler.var_
    return result

def conf2Std(in_conf, in_scaler = None):
    return np.sqrt(conf2Var(in_conf, in_scaler))

def sample2gtCameraVector(in_sample):
    return in_sample["target"].reshape((1, -1))
