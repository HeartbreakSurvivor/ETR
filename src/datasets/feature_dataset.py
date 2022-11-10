import numpy as np
import bisect, torch
import os.path as osp
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

def read_file(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines

class FeatureDataset(Dataset):
    def __init__(self,
            data_dir: str,
            sample_file: list,
            max_sequence_len: int,
            gnd_data=None):

        print('self.data_dir', data_dir)
        self.data_dir = data_dir

        samples = []
        lines = read_file(sample_file)
        for line in lines:
            name, label, width, height = line.split(',')
            samples.append((name, int(label), int(width), int(height)))

        self.categories = sorted(list(set([int(entry[1]) for entry in samples])))
        self.cat_to_label = dict(zip(self.categories, range(len(self.categories))))

        self.samples = [(entry[0], self.cat_to_label[entry[1]], entry[2], entry[3]) for entry in samples]
        self.targets = [entry[1] for entry in self.samples]
        self.gnd_data = gnd_data
        self.max_sequence_len = max_sequence_len
        self.scales = [0.5, 0.70710677, 1., 1.41421354, 2., 2.82842708, 4.]
  
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        '''
        Output
            global_desc: (2048, )
            local_desc: (max_sequence_len, 128)
            positions: (max_sequence_len, 2)
            local_mask: (max_sequence_len, )
            scale_inds: (max_sequence_len, )
            label: int
            name: str
        '''

        image_path, label, width, height = self.samples[index]
        image_name  = osp.splitext(osp.basename(image_path))[0]

        ####################################################################
        feat_path = osp.join(self.data_dir, image_name + '.npz')
        if not osp.exists(feat_path):
            print('feat_path', feat_path)
        # print('feat_path', feat_path)
        assert osp.exists(feat_path)

        feat = np.load(feat_path)
        global_desc = feat['global_desc']
        locations = feat['locations']
        desc = feat['local_desc']
        scales = feat['scales']
        ####################################################################
        # mask 默认都是True，代表都需要屏蔽
        local_mask = torch.ones(self.max_sequence_len, dtype=torch.bool)
        local_desc = np.zeros((self.max_sequence_len, 128), dtype=np.float32)
        scale_inds = torch.zeros(self.max_sequence_len).long()
        seq_len = min(desc.shape[0], self.max_sequence_len)
        local_desc[:seq_len] = desc[:seq_len]
        local_mask[:seq_len] = False # False means not mask
        scale_inds[:seq_len] = torch.as_tensor([bisect.bisect_right(self.scales, s) for s in scales[:seq_len]]).long() - 1

        ###############################################
        # Sine embedding
        positions = torch.zeros(self.max_sequence_len, 2).float()
        normx = locations[:, 1]/float(width)
        normy = locations[:, 0]/float(height)
        positions[:seq_len] = torch.from_numpy(np.stack([normx, normy], -1)).float()[:seq_len]
        ##############################################
        
        global_desc = torch.from_numpy(global_desc).float()
        local_desc = torch.from_numpy(local_desc).float()
        
        return global_desc, local_desc, local_mask, scale_inds, positions

class FeatureDataset_sp(Dataset):
    def __init__(self,
            data_dir: str,
            sample_file: list,
            max_sequence_len: int,
            gnd_data=None,
            is_train=False):

        print('self.data_dir', data_dir)
        self.data_dir = data_dir

        self.samples = []
        lines = read_file(sample_file)
        self.is_train = is_train
        for line in lines:
            if is_train:
                name, label  = line.split(' ')
                self.samples.append((name, int(label)))
            else:
                name = line.strip()
                self.samples.append((name))

        # self.categories = sorted(list(set([int(entry[1]) for entry in samples])))
        # self.cat_to_label = dict(zip(self.categories, range(len(self.categories))))
        # self.samples = [(entry[0], self.cat_to_label[entry[1]], entry[2], entry[3]) for entry in samples]
        if is_train:
            self.targets = [entry[1] for entry in self.samples]
        self.gnd_data = gnd_data
        self.max_sequence_len = max_sequence_len
  
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        '''
        Output
            global_desc: (4096, )
            local_desc: (max_sequence_len, 256)
            positions:  (max_sequence_len, 2)
            local_mask: (max_sequence_len, )
            scale_inds: (max_sequence_len, )
            label: int
            name: str
        '''

        if self.is_train:
            # image_path, label, width, height = self.samples[index]
            image_path, label = self.samples[index]
        else:
            image_path = self.samples[index]

        image_name  = osp.splitext(osp.basename(image_path))[0]

        ####################################################################
        feat_path = osp.join(self.data_dir, image_name + '.npz')
        # print('feat_path', feat_path)
        if not osp.exists(feat_path):
            print('feat_path', feat_path)
        # print('feat_path', feat_path)
        assert osp.exists(feat_path)

        feat = np.load(feat_path)
        # global_desc = feat['global_desc']
        locations = feat['locations']
        desc = feat['local_desc']
        # scales = feat['scales']
        ####################################################################
        # mask 默认都是True，代表都需要屏蔽
        local_mask = torch.ones(self.max_sequence_len, dtype=torch.bool)
        local_desc = np.zeros((self.max_sequence_len, 256), dtype=np.float32)
        seq_len = min(desc.shape[0], self.max_sequence_len)
        local_desc[:seq_len] = desc[:seq_len]
        local_mask[:seq_len] = False # False means not mask

        ###############################################
        # Sine embedding
        positions = torch.zeros(self.max_sequence_len, 2).float()
        normx = locations[:, 1] / 1 # float(width)
        normy = locations[:, 0] / 1 # float(height)
        positions[:seq_len] = torch.from_numpy(np.stack([normx, normy], -1)).float()[:seq_len]
        ##############################################
        
        # global_desc = torch.from_numpy(global_desc).float()
        local_desc = torch.from_numpy(local_desc).float()
        
        # 0 for global desc 
        return torch.randn([1]), local_desc, local_mask, positions

class FeatureDataset_val(Dataset):
    def __init__(self, 
            data_dir: str,
            samples_file: list,
            desc_name: str, 
            max_sequence_len: int,
            gnd_data=None):

        print('self.data_dir', data_dir)

        self.data_dir = data_dir        
        self.desc_name = desc_name
        
        self.categories = sorted(list(set([int(entry[1]) for entry in samples])))
        self.cat_to_label = dict(zip(self.categories, range(len(self.categories))))
        self.samples = [(entry[0], self.cat_to_label[entry[1]], entry[2], entry[3]) for entry in samples]
        self.targets = [entry[1] for entry in self.samples]
        self.gnd_data = gnd_data
        self.max_sequence_len = max_sequence_len
        self.scales = [0.5, 0.70710677, 1., 1.41421354, 2., 2.82842708, 4.]
  
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index, load_image=False):
        '''
        Output
            global_desc: (2048, )
            local_desc: (max_sequence_len, 128)
            local_mask: (max_sequence_len, )
            scale_inds: (max_sequence_len, )
            positions: (max_sequence_len, 2)
            label: int
            name: str
        '''
        image_path, label, width, height = self.samples[index]
        image_name  = osp.splitext(osp.basename(image_path))[0]
        global_path = osp.join(self.data_dir, image_name+'.delg_global')
        local_path  = osp.join(self.data_dir, image_name+'.delg_local')
        
        assert(osp.exists(global_path) and osp.exists(local_path))

        global_desc = datum_io.ReadFromFile(global_path)
        locations, scales, desc, attention, _ = feature_io.ReadFromFile(local_path)

        local_mask = torch.ones(self.max_sequence_len, dtype=torch.bool)
        local_desc = np.zeros((self.max_sequence_len, 128), dtype=np.float32)
        scale_inds = torch.zeros(self.max_sequence_len).long()
        seq_len = min(desc.shape[0], self.max_sequence_len)
        local_desc[:seq_len] = desc[:seq_len]
        local_mask[:seq_len] = False
        scale_inds[:seq_len] = torch.as_tensor([bisect.bisect_right(self.scales, s) for s in scales[:seq_len]]).long() - 1

        ###############################################
        # Sine embedding
        positions = torch.zeros(self.max_sequence_len, 2).float()
        normx = locations[:, 1]/float(width)
        normy = locations[:, 0]/float(height)
        positions[:seq_len] = torch.from_numpy(np.stack([normx, normy], -1)).float()[:seq_len]
        ##############################################
        
        global_desc = torch.from_numpy(global_desc).float()
        local_desc = torch.from_numpy(local_desc).float()
        
        if load_image:
            image = Image.open(osp.join(self.data_dir, image_path)).convert('RGB')
            image = image.resize((512, 512))
            return F.to_tensor(image), global_desc, local_desc, local_mask, scale_inds, positions, label, image_name
        else:
            return global_desc, local_desc, local_mask, scale_inds, positions, label, image_name

