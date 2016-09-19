from netUtils import TestNet, getScalerAndNet
import numpy as np

def testDetectionClassification(in_netParams, in_headArray,
        in_cameraData, in_isCount = True):
    if (in_cameraData.ndim == 1):
        in_cameraData = in_cameraData.reshape(1, -1)
    scaler, net = getScalerAndNet(in_netParams)

    headArray = in_headArray[:, :3].copy()
    nHeads = headArray.shape[0]
    if (in_cameraData.shape[0] == 1):
        data = np.hstack((headArray, in_cameraData * np.ones((nHeads, 1))))
    else:
        data = np.hstack((headArray, in_cameraData))

    data = scaler["input"].transform(data)
    data = {"data": data, "label": np.zeros((nHeads, 1))}

    res = TestNet(net, data, ["score"], in_isCount)
    score = np.exp(res["score"])
    out_result = score[:, 1] / score.sum(axis = 1)
    return out_result
