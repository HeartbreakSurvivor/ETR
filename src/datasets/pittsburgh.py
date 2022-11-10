import torch
import torchvision.transforms as transforms
import torch.utils.data as data

import os
from os.path import join, exists
from scipy.io import loadmat
from random import randint, random
from collections import namedtuple

import h5py
import numpy as np
from PIL import Image

from sklearn.neighbors import NearestNeighbors

def default_transform():
    return transforms.Compose([
        # 加上随机裁剪，增加数据集多样性，避免过拟合
        # 
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset', 
    'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
    'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])

def parse_dbStruct(path):
    mat = loadmat(path)
    matStruct = mat['dbStruct'].item()

    if '250k' in path.split('/')[-1]:
        dataset = 'pitts250k'
    else:
        dataset = 'pitts30k'

    whichSet = matStruct[0].item()

    dbImage = [f[0].item() for f in matStruct[1]]
    utmDb = matStruct[2].T

    qImage = [f[0].item() for f in matStruct[3]]
    utmQ = matStruct[4].T

    numDb = matStruct[5].item()
    numQ = matStruct[6].item()

    posDistThr = matStruct[7].item()
    posDistSqThr = matStruct[8].item()
    nonTrivPosDistSqThr = matStruct[9].item()

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage, 
            utmQ, numDb, numQ, posDistThr, 
            posDistSqThr, nonTrivPosDistSqThr)

class WholePittsburghDataset(data.Dataset):
    def __init__(self,
                 root_dir,
                 mode='train',
                 scale='30k',
                 onlyDB=False,
                 input_transform=default_transform(),
                 **kwargs):
        '''
            Args:
                - root_dir: the pittsburgh root directory
                - mode: options are ['train', 'val', 'test']
                - scale: '30k', '250k'
                - onlyDB: wether include query images to dataset or not
        '''
        super().__init__()

        if not exists(root_dir):
            raise FileNotFoundError('root_dir is hardcoded, please adjust to point to Pittsburth dataset')

        structFilePath = join(root_dir, 'datasets/')
        queries_dir = join(root_dir, 'queries_real/')
        self.input_transform = input_transform

        if mode not in ['train', 'val', 'test']:
            raise TypeError('mode must be on of [train, val, test]')
        if scale not in ['30k', '250k']:
            raise TypeError('scale must be 30k or 250k')

        structFile = join(structFilePath, 'pitts' + scale + '_' + mode + '.mat')
        print("structFile: ", structFile)

        self.dbStruct = parse_dbStruct(structFile)
        # print('self.dbStruct', self.dbStruct)
        self.images = [join(root_dir, dbIm) for dbIm in self.dbStruct.dbImage]

        if not onlyDB:
            self.images += [join(queries_dir, qIm) for qIm in self.dbStruct.qImage]

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.distances = None

        # with open('./pitts30k_test_rrt_db.txt', 'w') as f:
        #     dbimgs = [join(root_dir, dbIm) for dbIm in self.dbStruct.dbImage]
        #     print('len dbimgs', len(dbimgs))
        #     for i in dbimgs:
        #         img = Image.open(i)
        #         # print(img.size)
        #         width, height = img.size
        #         f.write(i+ ' ' + str(width) + ' ' + str(height) + '\n')

        # with open('./pitts30k_test_rrt_query.txt', 'w') as f:
        #     qimgs = [join(queries_dir, dbIm) for dbIm in self.dbStruct.qImage]
        #     print('len qimgs', len(qimgs))
        #     for i in qimgs:
        #         img = Image.open(i)
        #         width, height = img.size
        #         f.write(i+ ' ' + str(width) + ' ' + str(height) + '\n')


    def __getitem__(self, index):
        img = Image.open(self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if self.positives is None:
            knn = NearestNeighbors()
            knn.fit(self.dbStruct.utmDb)

            print('self.dbStruct.posDistThr', self.dbStruct.posDistThr)
            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ, radius=self.dbStruct.posDistThr)
        return self.positives

def pitts_collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (query, positive, negatives).
    
    Args:
        data: list of tuple (query, positive, negatives). 
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """

    batch = list(filter (lambda x:x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)
    import itertools
    indices = list(itertools.chain(*indices))

    return query, positive, negatives, negCounts, indices

class PittsburghTripletDataset(data.Dataset):
    def __init__(self,
                 root_dir,
                 mode='train',
                 scale='30k',
                 nNeg=10, 
                 margin=0.1,
                 nNegSample=1000,
                 cache_file=None,
                 input_transform=default_transform(),
                 **kwargs):
        '''
            Args:
                - root_dir: the pittsburgh root directory
                - mode: options are ['train', 'val']
                - scale: '30k', '250k'
                - nNeg: nubmer negative for training
                - nNegSample: number of negatives to randomly sample
                - cache_file: the cached feature file used to select positive and negative samples
                - input_transform: image preprocess
        '''
        super().__init__()

        self.root_dir = root_dir
        self.queries_dir = join(root_dir, 'queries_real/')
        self.structFilePath = join(root_dir, 'datasets/')

        self.margin = margin
        self.cache = cache_file # filepath of HDF5 containing feature vectors for images

        self.nNeg = nNeg # number of negatives used for training
        self.nNegSample = nNegSample # number of negatives to randomly sample
        self.input_transform = input_transform

        if not exists(root_dir):
            raise FileNotFoundError('root_dir is hardcoded, please adjust to point to Pittsburth dataset')

        # if not self.cache:
        #     raise ValueError('must specific cache file')

        if mode not in ['train', 'val']:
            raise TypeError('mode must be on of [train, val]')
        if scale not in ['30k', '250k']:
            raise TypeError('scale must be 30k or 250k')

        structFile = join(self.structFilePath, 'pitts' + scale + '_' + mode + '.mat')
        print("structFile: ", structFile)

        self.dbStruct = parse_dbStruct(structFile)
        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset
        print('dataset type, db_size, query_size ', self.dataset, self.dbStruct.numDb, self.dbStruct.numQ)

        # potential positives are those within nontrivial threshold range
        # fit NN to find them, search by radius
        knn = NearestNeighbors()
        knn.fit(self.dbStruct.utmDb)

        # TODO use sqeuclidean as metric?
        self.nontrivial_positives = list(knn.radius_neighbors(self.dbStruct.utmQ,
                radius=self.dbStruct.nonTrivPosDistSqThr**0.5, 
                return_distance=False))
        print('nontrivial_positives size: ', len(self.nontrivial_positives))
        
        # radius returns unsorted, sort once now so we dont have to later
        for i,posi in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(posi)

        # its possible some queries don't have any non trivial potential positives, filter those out
        self.queries = np.where(np.array([len(x) for x in self.nontrivial_positives])>0)[0]
        print('self.queries size: ', len(self.queries))
        
        # potential negatives are those outside of posDistThr range
        potential_positives = knn.radius_neighbors(self.dbStruct.utmQ,
                radius=self.dbStruct.posDistThr, 
                return_distance=False)
        print('potential_positives size: ', len(potential_positives))
        # print('potential_positives[0]', sorted(potential_positives[0]))
        # print('potential_positives[1432]', potential_positives[1432])

        self.potential_negatives = []
        for pos in potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(self.dbStruct.numDb),
                pos, assume_unique=True))
        print('potential_negatives size: ', len(self.potential_negatives))
        # print('potential_negatives[0]', self.potential_negatives[0])
        # print('potential_negatives[1432]', self.potential_negatives[1432])

        self.negCache = [np.empty((0,)) for _ in range(self.dbStruct.numQ)]

    def __getitem__(self, index):
        index = self.queries[index] # re-map index to match dataset

        if not exists(self.cache):
            raise ValueError('the cache feature file do not exist')

        with h5py.File(self.cache, mode='r') as h5:
            # h5feat中包含提取好的db和query的图片的feature
            h5feat = h5.get("features")

            qOffset = self.dbStruct.numDb
            qFeat = h5feat[index+qOffset]

            posFeat = h5feat[self.nontrivial_positives[index].tolist()]
            knn = NearestNeighbors() # TODO replace with faiss?
            knn.fit(posFeat)

            # find the nearest image of query in nontrivial_positives as the positive sample
            # 从query图片潜在的正样本集合中，挑选出特征向量最近的图片的下标
            dPos, posNN = knn.kneighbors(qFeat.reshape(1,-1), 1)
            dPos = dPos.item()
            posIndex = self.nontrivial_positives[index][posNN[0]].item()

            # 从当前query图片的潜在的负样本集合中，随机选出nNegSample个
            negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
            negSample = np.unique(np.concatenate([self.negCache[index], negSample]))

            # negFeat = h5feat[negSample.tolist()]
            # 把这些选出来的负样本的特征拿出来，然后放进knn中去建立索引
            negFeat = h5feat[negSample.astype(int).tolist()]
            knn.fit(negFeat)
            
            # to quote netvlad paper code: 10x is hacky but fine
            # 选出与query最相近的10*nNeg个负样本，相当于做困难负样本挖掘
            dNeg, negNN = knn.kneighbors(qFeat.reshape(1,-1), self.nNeg*10)
            dNeg = dNeg.reshape(-1)
            negNN = negNN.reshape(-1)

            # try to find negatives that are within margin, if there aren't any return none
            # 检查是否有违反条件的，即负样本的距离比正样本+一个margn的距离还小的情况
            violatingNeg = dNeg < dPos + self.margin**0.5
            # print('violatingNeg', violatingNeg)
            if np.sum(violatingNeg) < 1:
                #if none are violating then skip this query
                return None

            negNN = negNN[violatingNeg][:self.nNeg]
            negIndices = negSample[negNN].astype(np.int32)
            self.negCache[index] = negIndices

        # query 和 它的正样本，分别是一张图
        query = Image.open(join(self.queries_dir, self.dbStruct.qImage[index]))
        positive = Image.open(join(self.root_dir, self.dbStruct.dbImage[posIndex]))

        if self.input_transform:
            query = self.input_transform(query)
            positive = self.input_transform(positive)

        negatives = []
        for negIndex in negIndices:
            negative = Image.open(join(self.root_dir, self.dbStruct.dbImage[negIndex]))
            if self.input_transform:
                negative = self.input_transform(negative)
            negatives.append(negative)

        negatives = torch.stack(negatives, 0)

        # 返回 query、positive、 negatives，以及他们对应的下标
        return query, positive, negatives, [index, posIndex] + negIndices.tolist()

    def __len__(self):
        # 这里的queries是经过筛选后的，train 30k中有7416张query图片
        # 但是有些图片，找不到positive样本，所以就要筛掉，最后剩7320张query图片，来组成三元组训练样本
        return len(self.queries)

if __name__ == '__main__':
    # dataset = PittsburghTripletDataset('/root/halo/datasets/Pittsburgh')
    # print('len dataset: ', len(dataset))
    # q,p,n,index = dataset[0]
    # print(q.shape, p.shape, n.shape, index)

    # dataset = WholePittsburghDataset(root_dir='/root/halo/datasets/Pittsburgh',
    #                                 mode='train',
    #                                 scale='30k',
    #                                 onlyDB=True
    #                                 )
    # print('len dataset: ', len(dataset))
    # getPositives = dataset.getPositives()
    # print('getPositives shape ', getPositives.shape)
    # print('getPositives[0] ', getPositives[0])

    # np.save('/root/halo/delg/data/pitts30k/train/new_pitts30k_train_gt', getPositives)


    dataset = WholePittsburghDataset(root_dir='/root/halo/datasets/Pittsburgh',
                                    mode='train',
                                    scale='30k',
                                    onlyDB=True
                                    )
    print('len dataset: ', len(dataset))
    getPositives = dataset.getPositives()
    print('getPositives shape ', getPositives.shape)
    print('getPositives[0] ', getPositives[0])

    np.save('./pitts30k_train_gt', getPositives)