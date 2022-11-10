import random
from selectors import EpollSelector
import numpy as np

import faiss
import json
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from os.path import join, exists
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torchvision.transforms.functional import normalize, to_tensor
from torchvision.transforms.transforms import RandomCrop
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import autocast, GradScaler

IMG_H, IMG_W = 480, 640
default_train_transform = transforms.Compose(
    [
        # image augmentation
        # transforms.RandomApply([transforms.GaussianBlur(5)], p=0.2),
        # transforms.ColorJitter(),
        # transforms.RandomGrayscale(),
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),
    ]
)

default_scale_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop((IMG_H, IMG_W), scale=(0.08, 0.36)),
        # transforms.ColorJitter(),
        # transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def build_gldv2_struct(root_dir, filepath='gldv2_cluster.json'):
    filepath = root_dir + '/../' + filepath
    print('gldv2 struct file: ', filepath)
    if not exists(filepath):
        logger.info('start to build image with label file')
        logger.info('recursive traversal {} to find all images', root_dir)

        cluster = {str(folder): list(map(str, list(folder.rglob("*.jpg")))) for folder in Path(root_dir).iterdir()}
        js = json.dumps(cluster)
        with open(filepath, 'w') as f:
            f.write(js)
    else:
        with open(filepath, 'r') as f:
            cluster = json.load(f)
    return cluster

def whole_gldv2_train_test_split(root_dir:str):
    cluster = build_gldv2_struct(root_dir)

    total_imgs = sum([len(v) for k, v in cluster.items()])
    logger.info('total different places size: {}', len(list(cluster.keys())))
    logger.info('total images: {}', total_imgs)
    # remove empty folder
    empty_folder = [k for k, v in cluster.items() if len(v) < 2]
    for e in empty_folder:
        del cluster[e]
    logger.info('total not empty different places size: {}', len(list(cluster.keys())))

    print('cluster keys: ', len(cluster.keys()))
    places =  list(cluster.keys())
    random.shuffle(places)
    places = set(random.sample(places, 1000)) # random selcet 15000 different places
    print('places: ', len(places))

    val = []
    train = []
    imgWithLabel = []
    sIdx = 0 # add label for dataset
    for idx, (c, imgs) in tqdm(enumerate(cluster.items())):
        pick = None
        if c in places: # random choose a picture from this places
            pick = random.choice(imgs)
            
            # main difference
            for img in imgs:
                data = img + ' ' + str(sIdx)
                if img != pick: # the choosed picture should be put in val set
                    train.append(data)
                else:
                    val.append(data)
                imgWithLabel.append(data)
            sIdx += 1 # the same place has the same label
    
    logger.info('train size: {}, val size: {}, imgWithLabel size: {}', len(train), len(val), len(imgWithLabel))
    
    with open(root_dir + '/../' + 'gldv2_train_clean_labels_s.txt', 'w') as f:
        f.writelines('\n'.join(imgWithLabel))
    with open(root_dir + '/../' + 'gldv2_train_clean_train_s.txt', 'w') as f:
        f.writelines('\n'.join(train))
    with open(root_dir + '/../' + 'gldv2_train_clean_val_s.txt', 'w') as f:
        f.writelines('\n'.join(val))

    print('is the same? ', set(imgWithLabel) == (set(train) | set(val)))

    # X, Y = [], []
    # for iwl in imgWithLabel:
    #     img, label = iwl.split()
    #     X.append(img)
    #     Y.append(int(label)) # convert str to int
    
    # use random_state=0 for reproducibility, do not change!
    # test size set to 15000
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=15000, shuffle=True)

    # for sanity check
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=1000, test_size=100, random_state=0)

    # save train and test set
    # train = [x + ' ' + str(y) for x, y in zip(X_train, Y_train)]
    # val = [x + ' ' + str(y) for x, y in zip(X_test, Y_test)]
    # with open('./gldv2_train_clean_train.txt', 'w') as f:
    #     f.writelines('\n'.join(train))
    # with open('./gldv2_train_clean_val.txt', 'w') as f:
    #     f.writelines('\n'.join(val))

    # logger.info('X_train: {}, X_test: {}, Y_train: {}, Y_test: {}', len(X_train), len(X_test), len(Y_train), len(Y_test))

def load_train_val_dataset(root_dir:str, mode):
    x, y = [], []
    if mode == 'train':
        # fp = root_dir + '/../' + 'gldv2_train_clean_train_s.txt'
        fp = root_dir + '/../' + 'gldv2_train_clean_train.txt'
    else:
        # fp = root_dir + '/../' + 'gldv2_train_clean_val_s.txt'
        fp = root_dir + '/../' + 'gldv2_train_clean_val.txt'

    with open(fp, 'r') as f:
        for line in f:
            line = line.strip()
            img, label = line.split()
            x.append(img)
            y.append(int(label))
    logger.info('mode: {}, X size: {}, y size: {}', mode, len(x), len(y))
    return x, y

class GLDv2ClassificationDataset(Dataset):
    '''
        construct whole dataset for classification task use loss like ArcFace.
    '''
    def __init__(self, 
                root_dir,
                mode='train', # options in ['train', 'test', 'val']
                transform=default_train_transform):

        self.root_dir = root_dir
        self.transform = transform
        self.imgs, self.labels = load_train_val_dataset(self.root_dir, mode)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img, label = self.imgs[index], self.labels[index]
        img = self.transform(Image.open(self.imgs[index]))
        return img, label

def whole_gldv2_cluster_train_test_split(root_dir:str, mode:str):
    cluster = build_gldv2_struct(root_dir)

    places = set(cluster.keys())
    total_imgs = sum([len(v) for k, v in cluster.items()])
    logger.info('total {} different places', len(places))
    logger.info('total {} images', total_imgs)
    
    # remove empty folder
    empty_folder = [k for k, v in cluster.items() if len(v) < 2]
    for e in empty_folder:
        del cluster[e]
    logger.info('total not empty different places size: {}', len(list(cluster.keys())))

    logger.info('start to random split train and test places')
    test_place_size = 5000
    logger.info('random select {} different places ', test_place_size)
    # use Random(0) to get a private random number generator for reproducibility, do not change!
    test_place = random.Random(0).sample(places, test_place_size)
    train_places = places - set(test_place)
    logger.info('different train places: {}, different test places: {} ', len(train_places), len(test_place))

    if mode == 'train':
        train_cluster = {k: cluster[k] for k in train_places}
        train_cluster_id = {cls: i for i, cls in enumerate(train_cluster)}
        return train_cluster, train_cluster_id
    else:
        test_query = []
        test_gt = []
        test_db = []

        # randomly select an image as query image from curremt place, and the rest are considered as db images
        cur_idx = 0
        for p in test_place:
            cur_place = set(cluster[p])
            query = random.Random(0).sample(cur_place, 1)
            db = cur_place - set(query)

            test_query.extend(query)
            test_db.extend(db)
            # store the query image's gt index in db
            test_gt.append(list(np.arange(cur_idx, cur_idx+len(db))))
            cur_idx += len(db)

        return test_query, test_gt, test_db

class ImageFromList(Dataset):
    def __init__(self, img_list, transform=default_train_transform):
        super(ImageFromList, self).__init__()
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        sample = Image.open(self.img_list[index])
        return self.transform(sample)

def gld_collate_fn(batch):
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

    query, positive, negatives = zip(*batch)

    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)

    return query, positive, negatives, negCounts

class GLDv2TripletDataset(Dataset):
    def __init__(self, 
                root_dir,
                num_query=1,
                num_positive=1,
                num_negative=10,
                transform=default_train_transform):
        
        super(GLDv2TripletDataset, self).__init__()

        self.cluster = whole_gldv2_cluster_train_test_split(root_dir, 'train')
        self.cluster_id = {cls: i for i, cls in enumerate(self.cluster)}

        self.num_query = num_query
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.transform = transform
        self.num_negative_pool = 1

    def _find_positives(self, ndcg=False):
        logger.info("collecting positive samples start...")
        self.query, self.positives, self.negative_pool = [], [], []
        self.query_label, self.negative_label = [], []

        for cls in tqdm(self.cluster):
            if len(self.cluster[cls]) < self.num_negative_pool:
                continue
            self.negative_pool.extend(random.sample(self.cluster[cls], self.num_negative_pool))
            self.negative_label.extend([self.cluster_id[cls]] * self.num_negative_pool)

            # if the number of images of current place less than num_query + num_positive, means that we can't even construct a basic triplet
            if len(self.cluster[cls]) < self.num_query + self.num_positive or random.randint(0, 9) != 0:
                continue

            # selecet query+positive image from current place
            selected = random.sample(self.cluster[cls], self.num_query + self.num_positive)
            self.query.extend(selected[:self.num_query])
            self.positives.extend(selected[self.num_query:])
            self.query_label.extend([self.cluster_id[cls]] * self.num_query)
        logger.info('tuple config, query: {}, positive: {}, negatives: {}', self.num_query, self.num_positive, self.num_negative)
        logger.info("selected {} query images, {} positive images, {} negative images", len(self.query), len(self.positives), len(self.negative_pool))
        
        query_label, negative_label = torch.tensor(self.query_label), torch.tensor(self.negative_label)
        q_size, n_size = query_label.size(0), negative_label.size(0)
        query_label = query_label.unsqueeze(1).expand(q_size, n_size)
        negative_label = negative_label.unsqueeze(0).expand(q_size, n_size)
        self.mask = query_label == negative_label
        logger.info("collecting positives complete")
    
    def create_epoch_ds(self, net):
        # 1. for each query image, random sampling to find positive and negative images
        self._find_positives()

        self.negatives = []
        query_dataloader = DataLoader(ImageFromList(self.query), batch_size=48, num_workers=24)
        negative_dataloader = DataLoader(ImageFromList(self.negative_pool), batch_size=48, num_workers=24)

        # query_features, negative_features = torch.zeros(len(self.negative_pool), 2048).float().cuda(), torch.zeros(len(self.negative_pool), 2048).float().cuda()
        query_features, negative_features = None, None
        with torch.no_grad():
            with autocast():
                logger.info("calculating query features...")

                for idx, inputs in enumerate(tqdm(query_dataloader)):
                    inputs = inputs.to('cuda')
                    # inputs = inputs.type_as()
                    out, _, _ = net(inputs)
                    if query_features is None:
                        query_features = out
                    else:
                        query_features = torch.cat((query_features, out), dim=0)
                logger.info("query image features calculating complete")
                logger.info("calculating negative features...")

                for idx, inputs in enumerate(tqdm(negative_dataloader)):
                    inputs = inputs.to('cuda')
                    out, _, _ = net(inputs)
                    if negative_features is None:
                        negative_features = out
                    else:
                        negative_features = torch.cat((negative_features, out), dim=0)
                logger.info("negative image features calculating finish")

            logger.info("start calculating similarity...")

            similarity = torch.mm(query_features, negative_features.T)
            similarity = similarity * torch.logical_not(self.mask)

            logger.info("sorting")
            rank = torch.argsort(similarity, descending=True)[:, :self.num_negative]
            negative_pool = np.array(self.negative_pool)
            for r in rank:
                self.negatives.extend(np.take_along_axis(negative_pool, r.numpy(), axis=0).tolist())
        logger.info('dataset created finished.')
        logger.info("selected {} query images, {} positive images, {} negative images", len(self.query), len(self.positives), len(self.negatives))

    def __getitem__(self, index):
        # query_idx, positive_idx, negative_idx = index, index*self.num_positive, index*self.num_negative
        query = [self.query[index]]
        positive = self.positives[index * self.num_positive: (index + 1) * self.num_positive]
        negatives = self.negatives[index * self.num_negative: (index + 1) * self.num_negative]

        logger.info('query: ', query)
        logger.info('positive: ', positive)
        logger.info('negative: ', negatives)

        img_list = query + positive + negatives
        sample = torch.zeros(len(img_list), 3, IMG_H, IMG_W).float()
        for idx, img in enumerate(img_list):
            sample[idx, :, :, :] = self.transform(Image.open(img))
        
        logger.info('samples shape: {}', sample.shape)

        query = torch.stack(sample[:self.num_query],0)
        positive = torch.stack(sample[self.num_query:self.num_query+self.num_positive], 0)
        negatives = torch.stack(sample[-self.num_negative:], 0)

        return query, positive, negatives

    def __len__(self) -> int:
        return len(self.query)

class GLDv2ValidationDataset(Dataset):
    def __init__(self, 
                root_dir,
                transform=default_train_transform):
        super().__init__()
        self.transform = transform
        self.query, self.query_gt, self.db = \
            whole_gldv2_cluster_train_test_split(root_dir, 'test')
        self.imgs = self.query + self.db

    def get_query(self):
        return self.query

    def get_db(self):
        return self.db

    def get_query_gt(self):
        return self.query_gt

    def __getitem__(self, index):
        img = self.transform(Image.open(self.imgs[index]))
        return img

    def __len__(self) -> int:
        return len(self.imgs)


def sample_superpoint_train_set(root_dir:str):

    cluster = build_gldv2_struct(root_dir)

    total_imgs = sum([len(v) for k, v in cluster.items()])
    logger.info('total different places size: {}', len(list(cluster.keys())))
    logger.info('total images: {}', total_imgs)
    # remove empty folder
    empty_folder = [k for k, v in cluster.items() if len(v) < 2]
    for e in empty_folder:
        del cluster[e]
    print('total not empty different places size: ', len(list(cluster.keys())))
    print('cluster keys: ', len(cluster.keys()))

    # 筛选出大于10张图的地点
    larger_places = [k for k, v in cluster.items() if len(v) >= 10]
    print('places largere than 10 images: ', len(larger_places))

    print('before shuffle', larger_places[:3])
    random.shuffle(larger_places)
    print('after shuffle', larger_places[:3])

    places = larger_places[:20000] # random selcet 15000 different places
    print('selected places: ', len(places))

    imgWithLabel = []
    for idx, p in tqdm(enumerate(places)):
        imgs = cluster[p]
        # for each place, we sample 100 image at most.
        if len(imgs) > 100:
            # shuffle一下，取100张
            random.shuffle(imgs)
            imgs = imgs[:100]

        for img in imgs:
            # pic = Image.open(img)
            # width, height = pic.size
            # data = img + ' ' + str(idx) + ' ' + str(width) + ' ' + str(height)

            data = img + ' ' + str(idx)
            imgWithLabel.append(data)

    logger.info('imgWithLabel size: {}', len(imgWithLabel))
    
    store_path = '/root/halo/code/MSFFT/datasets/GLD_v2/sp_train/'

    with open(store_path + '/sp_gldv2_train.txt', 'w') as f:
        f.writelines('\n'.join(imgWithLabel))
    # with open(root_dir + '/../' + 'gldv2_train_clean_train_s.txt', 'w') as f:
    #     f.writelines('\n'.join(train))
    # with open(root_dir + '/../' + 'gldv2_train_clean_val_s.txt', 'w') as f:
    #     f.writelines('\n'.join(val))

    # print('is the same? ', set(imgWithLabel) == (set(train) | set(val)))

    # X, Y = [], []
    # for iwl in imgWithLabel:
    #     img, label = iwl.split()
    #     X.append(img)
    #     Y.append(int(label)) # convert str to int
    
    # use random_state=0 for reproducibility, do not change!
    # test size set to 15000
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=15000, shuffle=True)

    # for sanity check
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=1000, test_size=100, random_state=0)

    # save train and test set
    # train = [x + ' ' + str(y) for x, y in zip(X_train, Y_train)]
    # val = [x + ' ' + str(y) for x, y in zip(X_test, Y_test)]
    # with open('./gldv2_train_clean_train.txt', 'w') as f:
    #     f.writelines('\n'.join(train))
    # with open('./gldv2_train_clean_val.txt', 'w') as f:
    #     f.writelines('\n'.join(val))

    # logger.info('X_train: {}, X_test: {}, Y_train: {}, Y_test: {}', len(X_train), len(X_test), len(Y_train), len(Y_test))


if __name__ == "__main__":
    root_dir = "/root/halo/datasets/GoogleLandmarkv2/train_clean"

    # construct training set for SuperPoint
    sample_superpoint_train_set(root_dir)

    # construct train and val dataset
    # whole_gldv2_train_test_split(root_dir)

    # # for pretrain
    # dataset = GLDv2ClassificationDataset(root_dir, 'train')
    # print('train len: ', len(dataset))
    # i1, l1 = dataset[0]
    # print('image', i1.shape)

    # dataset = GLDv2ClassificationDataset(root_dir, 'test')
    # print('test len: ', len(dataset))
    # i1, l1 = dataset[0]
    # print('image', i1.shape)

    # dataset = GLDv2ValidationDataset(root_dir)
    # print('val len: ', len(dataset))
    # i1 = dataset[0]
    # print('image', i1.shape)
    # q, gt, db = dataset.get_query(), dataset.get_query_gt(),dataset.get_db()
    # print('query size, gt, db ', len(q), len(gt), len(db))
    # print('query0 path: ', q[0])
    # print('query0 gt : ', gt[0])
    # for i in gt[0]:
    #     print('db path: ', db[i])

    # print('query989 path: ', q[989])
    # print('query989 gt : ', gt[989], len(gt[989]))
    # for i in gt[989]:
    #     print('db path: ', db[i])