import os
import numpy as np
from PIL import Image
import os

def histc(x, bins):
    map_to_bins = np.digitize(x, bins) # Get indices of the bins to which each value in input array belongs.
    res = np.zeros(bins.shape)
    for el in map_to_bins:
        res[el-1] += 1 # Increment appropriate bin.
    return res

def evalseg(resdir, gtids, whicheval):
    gt_dir = os.path.join("../data/test")
    if not resdir:
        resdir = gt_dir
    if not gtids:
        gtids = np.unique([file_name[:4] for file_name in os.listdir(gt_dir)])
    classes, labext = (['person'], '_person') if whicheval == 1 else (
        ['skin', 'hair', 'tshirt', 'shoes', 'pants', 'dress'], '_clothes')
    num, nclasses = len(classes) + 1, len(classes)
    confcounts, count = np.zeros((num, num)), 0
    print(f"testing {len(gtids)} images")
    for i in range(len(gtids)):
        imname = gtids[i]
        gtfile = os.path.join(gt_dir, f'{imname}{labext}.png')
        if not os.path.exists(gtfile):
            print('gt file doesnt exist, skipping')
            continue
        gtim = np.array(Image.open(gtfile))
        gtim = gtim.astype(float)
        resfile = os.path.join(resdir, f'{imname}{labext}.png')
        if os.path.exists(resfile):
            resim = np.array(Image.open(resfile).convert('L'))
        else:
            resim = np.zeros(gtim.shape)
        resim = resim.astype(float)
        szgtim = gtim.shape
        szresim = resim.shape
        if(szgtim != szresim):
            raise ValueError(
                f"Results image '{imname}' is the wrong size, was {szresim[0]} x {szresim[1]}, should be {szgtim[0]} x {szgtim[1]}.")

        locs = gtim < 255
        gtim[locs] = gtim[locs].astype(float)
        sumim = gtim * num + resim
        hs = histc(sumim[locs], np.array(list(range(1,num * num+1))))
        count += np.sum(locs)
        confcounts += hs.reshape((num, num))

    conf = 100 * confcounts / (np.sum(confcounts, axis=1)[:, None] + 1E-20)
    rawcounts = confcounts
    accuracies = np.zeros(num)
    print('Accuracy for each class (intersection/union measure)')

    for j in range(num):
        gtj = np.sum(confcounts[j, :])
        resj = np.sum(confcounts[:, j])
        gtjresj = confcounts[j, j]
        accuracies[j] = 100 * gtjresj / (gtj + resj - gtjresj)
        clname = 'background'
        if j > 0:
            clname = classes[j - 1]
        print(f'  {clname:>14s}: {accuracies[j]:6.3f}%')
    accuracies = accuracies[1:]
    avacc = np.mean(accuracies)
    print('-------------------------')
    for i in range(len(classes)):
        print(f'{classes[i]} accuracy: {avacc:6.3f}%')


if __name__ == "__main__":
    evalseg("../output/test_output1", None, 1)
