import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial as spatial
import sys
import torch
import torch.nn as nn
import torch.optim as optim
# from IPython.display import clear_output as clear


class PointData:
    def __init__(self, path: str, seed=1145141919):
        self.path = path
        self.pData = np.loadtxt(self.path).astype('float32')
        sys.setrecursionlimit(int(max(1000, round(
            self.pData.shape[0] /
            10))))
        self.kdtree = spatial.cKDTree(self.pData, 10)
        self.patch = []
        self.radScale = -1
        self.rad = -1
        self.rng = np.random.RandomState(seed)
        print("Load %s success!" % (self.path))

    def getPatch(self, center, rad: float):
        return self.pData[np.array(self.kdtree.query_ball_point(center, rad))]

    def gernatePatch(self, rad: float, point_per_patch=100):
        if self.radScale < 0:
            pData_g = torch.from_numpy(self.pData).to(device)
            pmin = np.array([torch.min(pData_g[:, i]).cpu() for i in range(3)])
            pmax = np.array([torch.max(pData_g[:, i]).cpu() for i in range(3)])
            self.radScale = np.linalg.norm(pmin - pmax)
        if self.rad == rad*self.radScale:
            print("Patch already exists!")
            return
        else:
            print(self.rad, rad * self.radScale)
        self.patch = []
        self.rad = rad * self.radScale

        total_range = self.pData.shape[0]
        for i in range(total_range):
            p = self.pData[i]
            patch_ = self.getPatch(p, self.rad)
            if patch_.shape[0] > point_per_patch:
                patch_ind = self.rng.choice(patch_.shape[0],
                                            point_per_patch,
                                            replace=False)
                patch_ = patch_[patch_ind]
            if patch_.shape[0] < point_per_patch:
                patch_ = np.vstack(
                    (patch_,
                     np.array([[0] * 3] *
                              (point_per_patch - patch_.shape[0]))))
            self.patch.append(patch_)
            if i % 100 == 99:
                print("[Build patch %d%%] shape %s" %
                      (round(100*i/total_range), self.path), end='\r')
        self.patch = np.array(self.patch)
        print("\nBuild patch set success!")


device = torch.device("cuda")


def lossMin(pd, pOrg, scale=1000):
    p1 = torch.from_numpy(pd).to(device)
    po = torch.from_numpy(pOrg).to(device)
    # print(p1.shape,po.shape)

    # return np.array([(torch.min((p - po).pow(2).sum(1))*scale).cpu().numpy() for p in p1])
    res = torch.zeros(p1.shape[0], dtype=torch.float, device=device)
    for i in range(p1.shape[0]):
        res[i] = torch.min((p1[i] - po).pow(2).sum(1))
    return res.cpu().numpy() * scale


def drawPoint(fig, ax, pDataPre, pDataOrg, loss=lossMin, lossScale=1000, pointSize=5, elev=90, azim=-90, pltShow=True, hideGrid=True):
    l = loss(pDataPre, pDataOrg, lossScale)
    x = pDataPre[:, 0].reshape(-1,)
    y = pDataPre[:, 1].reshape(-1,)
    z = pDataPre[:, 2].reshape(-1,)
    print("Compute loss success!")

    # colorbar
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    my_cmap = plt.get_cmap('jet')

    pos = ax.scatter3D(x, y, z, s=pointSize, c=l, cmap=my_cmap, vmin=0, vmax=1)
    # fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=my_cmap), ax=ax)
    if hideGrid:
        plt.axis('off')
        ax.grid(False)
        # ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    ax.view_init(elev=elev, azim=azim)
    if pltShow == True:
        plt.show()


class GaussFilter(nn.Module):
    def __init__(self, kernelRad):
        super().__init__()
        self.kernelRad = kernelRad

    def forward(self, patch, pData):
        pData = pData.unsqueeze(1)
        vDis = patch-pData
        nDis = torch.norm(vDis, dim=2)/self.kernelRad
        weight = 0.39894228040143267794*torch.exp(-nDis**2/2.)
        we_sum = weight.sum(1).unsqueeze(1)
        weight = weight/we_sum
        weight = weight.unsqueeze(2)
        WdPatp = weight*patch
        res = WdPatp.sum(1)
        # for i in range(pData.shape[0]):
        #   pData[i]=(gaussKernel((patch[i]-pData[i]).))
        return res


def pkg_denoise_gauss(fig, noiseSet, orgSet, rad=0.02, lossScale=1e4, pointSize=8, subIndex=0, lineNum=1):
    noiseSet.gernatePatch(rad)
    patch_g = torch.from_numpy(noiseSet.patch).to(device)
    pData_g = torch.from_numpy(noiseSet.pData).to(device)
    print("Copy data to gpu success!")

    gf = GaussFilter(noiseSet.rad)
    with torch.no_grad():
        prev = gf(patch_g, pData_g)

    print("Finish denoise!")
    pre_pNoise = prev.cpu().numpy()

    print("Drawing Picture original point...")

    ax1 = fig.add_subplot(lineNum, 3, 1+subIndex*3, projection='3d')
    drawPoint(fig, ax1, orgSet.pData, orgSet.pData,
              lossScale=lossScale, pointSize=pointSize, pltShow=False)

    print("Drawing Picture noise point...")

    ax2 = fig.add_subplot(lineNum, 3, 2+subIndex*3, projection='3d')
    drawPoint(fig, ax2, noiseSet.pData, orgSet.pData,
              lossScale=lossScale, pointSize=pointSize, pltShow=False)

    print("Drawing Picture denoise result...")
    ax3 = fig.add_subplot(lineNum, 3, 3+subIndex*3, projection='3d')
    drawPoint(fig, ax3, pre_pNoise, orgSet.pData,
              lossScale=lossScale, pointSize=pointSize, pltShow=False)
    # clear()


if __name__ == "__main__":
    cupOrg = PointData("./pointCleanNetDataset/Cup33100k.xyz")
    cupNoise = PointData(
        "./pointCleanNetDataset/Cup33100k_noise_white_1.00e-02.xyz")
    bunnyOrg = PointData("./pointCleanNetDataset/bunny100k.xyz")
    bunnyNoise = PointData(
        "./pointCleanNetDataset/bunny100k_noise_white_1.00e-02.xyz")
    starOrg = PointData("./pointCleanNetDataset/star_halfsmooth100k.xyz")
    starNoise = PointData(
        "./pointCleanNetDataset/star_halfsmooth100k_noise_white_1.00e-02.xyz")

    for r in [0.01, 0.025, 0.05]:
        fig = plt.figure(figsize=(30, 30))
        pkg_denoise_gauss(fig, cupNoise,    cupOrg,   rad=r,
                          subIndex=0, lineNum=3, lossScale=1e3)
        pkg_denoise_gauss(fig, bunnyNoise,  bunnyOrg, rad=r,
                          subIndex=1, lineNum=3, lossScale=1e3)
        pkg_denoise_gauss(fig, starNoise,   starOrg,  rad=r,
                          subIndex=2, lineNum=3, lossScale=1e3)
        plt.tight_layout()
        plt.savefig("./denoise_a_"+str(int(r*1000))+".png")
