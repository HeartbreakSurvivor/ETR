import os.path as osp
import numpy as np
from tqdm import tqdm
from numpy import linalg as LA


if __name__ == '__main__':

    with open('/root/halo/code/MSFFT/datasets/NordLand/nordland_query.txt') as fid:
        query_lines = fid.read().splitlines()
    with open('/root/halo/code/MSFFT/datasets/NordLand/nordland_db.txt') as fid:
        gallery_lines = fid.read().splitlines()

    query_feats = []
    query_path = '/root/halo/delg/data/nordland/delg_r101_gldv2/query'
    for i in tqdm(range(len(query_lines))):
        imgname = osp.basename(query_lines[i]).split('.')[0]
        path = osp.join(query_path, imgname+'.npz')

        assert osp.exists(path)
        feat = np.load(path)
        query_feats.append(feat['global_desc'])

    query_feats = np.stack(query_feats, axis=0)
    print('query_feats shape', query_feats.shape)
    np.save('/root/halo/code/MSFFT/datasets/NordLand/delg_r101_feats/delg_qfeat', query_feats)

    query_feats = query_feats / LA.norm(query_feats, axis=-1)[:,None]

    index_feats = []
    db_path = '/root/halo/delg/data/nordland/delg_r101_gldv2/index'
    for i in tqdm(range(len(gallery_lines))):
        imgname = osp.basename(gallery_lines[i]).split('.')[0]
        path = osp.join(db_path, imgname+'.npz')

        assert osp.exists(path)
        feat = np.load(path)
        index_feats.append(feat['global_desc'])

        # path = osp.join(db_path, imgname+'.delg_global')
        # index_feats.append(datum_io.ReadFromFile(path))
        
    index_feats = np.stack(index_feats, axis=0)
    np.save('/root/halo/code/MSFFT/datasets/NordLand/delg_r101_feats/delg_dbfeat', index_feats)

    index_feats = index_feats / LA.norm(index_feats, axis=-1)[:,None]

    sims = np.matmul(query_feats, index_feats.T)
    print('sim shape', sims.shape)

    nn_inds  = np.argsort(-sims, -1)
    print('nn_inds.shape', nn_inds.shape)

    np.save('/root/halo/code/MSFFT/datasets/NordLand/delg_r101_feats/nordland_nn_inds', nn_inds)
    # nn_dists = deepcopy(sims)


    # for i in range(query_feats.shape[0]):
    #     for j in range(index_feats.shape[0]):
    #         nn_dists[i, j] = sims[i, nn_inds[i, j]]

    # np.save('/root/halo/delg/data/pitts30k/test/pitts_ranks', nn_dists)

    # if save_nn_inds:
    #     output_path = osp.join(data_dir, 'nn_inds_%s.pkl'%feature_name)
    #     pickle_save(output_path, nn_inds)


    # gnd_data = pickle_load(osp.join(data_dir, gnd_name))
    # compute_metrics('revisited', nn_inds.T, gnd_data['gnd'], kappas=[1,5,10])