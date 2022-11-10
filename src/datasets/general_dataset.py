import os
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.neighbors import NearestNeighbors

import torch.utils.data as data
import torchvision.transforms as transforms

# from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR

def input_transform(resize=(480, 640)):
    if resize[0] > 0 and resize[1] > 0:
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

class PlaceDataset(data.Dataset):
    def __init__(self, dataset_name, query_file_path, index_file_path, dataset_root_dir=None, ground_truth_path=None, config=None):
        super().__init__()
        
        self.db_pose = []
        self.dataset_name = dataset_name
        self.queries, self.database, self.numQ, self.numDb, self.utmQ, self.utmDb, self.posDistThr = None, None, None, None, None, None, None

        if query_file_path is not None:
            self.queries, self.numQ = self.parse_text_file(query_file_path)
        if index_file_path is not None:
            self.database, self.numDb = self.parse_text_file(index_file_path)
        if ground_truth_path is not None:
            if dataset_name in ['pitts_30k', 'tokyo247','msls_val', 'nordland']:
                self.utmQ, self.utmDb, self.posDistThr = self.parse_gt_file(ground_truth_path)
            else:
                with open(ground_truth_path, 'r') as f:
                    self.db_pose = [l.strip() for l in f.readlines()]

        if self.queries is not None:
            # images 排列 Db+Query
            self.images = self.database + self.queries
        else:
            self.images = self.database

        # self.images = [os.path.join(dataset_root_dir, image) for image in self.images]
        
        # # check if images are relative to root dir
        # if not os.path.isfile(self.images[0]):
        #     if os.path.isfile(os.path.join(PATCHNETVLAD_ROOT_DIR, self.images[0])):
        #         self.images = [os.path.join(PATCHNETVLAD_ROOT_DIR, image) for image in self.images]

        self.positives = None
        self.distances = None

        # self.resize = (int(config['imageresizeH']), int(config['imageresizeW']))
        # self.mytransform = input_transform(self.resize)
        self.mytransform = input_transform()

    def get_db_size(self):
        return len(self.database)

    def get_query_size(self):
        return len(self.queries)

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        img = self.mytransform(img)
        return img, index

    def __len__(self):
        return len(self.images)

    def get_db_pose(self):
        return self.db_pose

    def get_positives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.utmDb)

            print('posDistThr', self.posDistThr)
            self.distances, self.positives = knn.radius_neighbors(self.utmQ, radius=self.posDistThr)

        return self.positives

    @staticmethod
    def parse_text_file(textfile):
        print('Parsing dataset...')

        with open(textfile, 'r') as f:
            image_list = f.read().splitlines()

        if 'robotcar' in image_list[0].lower():
            image_list = [os.path.splitext('/'.join(q_im.split('/')[-3:]))[0] for q_im in image_list]

        num_images = len(image_list)

        print('Done! Found %d images' % num_images)

        return image_list, num_images

    @staticmethod
    def parse_gt_file(gtfile):
        print('Parsing ground truth data file...')
        gtdata = np.load(gtfile)
        return gtdata['utmQ'], gtdata['utmDb'], gtdata['posDistThr']


if __name__ == '__main__':

    # 'query_path': ROOT_DIR + "MSFFT/datasets/Tokyo247/tokyo247_query.txt",
    #     'db_path': ROOT_DIR + "MSFFT/datasets/Tokyo247/tokyo247_db.txt",
    #     'gt_path': ROOT_DIR + "MSFFT/datasets/Tokyo247/tokyo247.npz",
    #     'out_dir': ROOT_DIR + "MSFFT/output_results/tokey247"


    # dataset = PlaceDataset(dataset_name="tokyo247", 
    #     query_file_path='/root/halo/code/MSFFT/datasets/Tokyo247/tokyo247_query.txt', 
    #     index_file_path='/root/halo/code/MSFFT/datasets/Tokyo247/tokyo247_db.txt', 
    #     dataset_root_dir = None, 
    #     ground_truth_path='/root/halo/code/MSFFT/datasets/Tokyo247/tokyo247.npz',
    #     config=None)
    # gt = dataset.get_positives()
    # print('gt shaep', gt.shape)

    # np.save('/root/halo/code/MSFFT/datasets/Tokyo247/tokyo247_gt', gt)
    
    dataset = PlaceDataset(dataset_name="pitts_30k", 
        query_file_path='/root/halo/code/MSFFT/datasets/Pitts30k/pitts30k_query.txt', 
        index_file_path='/root/halo/code/MSFFT/datasets/Pitts30k/pitts30k_db.txt', 
        dataset_root_dir = None, 
        ground_truth_path='/root/halo/code/MSFFT/datasets/Pitts30k/pitts30k_test.npz',
        config=None)
    gt = dataset.get_positives()
    print('gt shaep', gt.shape)

    np.save('/root/halo/code/MSFFT/datasets/Pitts30k/pitts30k_gt', gt)
