import os.path as osp
from pathlib import Path
import numpy as np
from tqdm import tqdm
from numpy import linalg as LA


if __name__ == '__main__':
    query_broken_files = []
    index_broken_files = []

    with open('/root/halo/code/MSFFT/datasets/TokyoTM/tokyoTM_query_c.txt') as fid:
        query_lines = fid.read().splitlines()
        query_lines = [q.split(',')[0] for q in query_lines]

    with open('/root/halo/code/MSFFT/datasets/TokyoTM/tokyoTM_db_c.txt') as fid:
        gallery_lines = fid.read().splitlines()
        gallery_lines = [q.split(',')[0] for q in gallery_lines]

    query_feats = []
    query_path = '/root/halo/code/MSFFT/datasets/TokyoTM/delg_feats/query'
    for i in tqdm(range(len(query_lines))):
        imgname = osp.basename(query_lines[i])
        imgname = Path(imgname).stem
        path = osp.join(query_path, imgname+'.npz')

        try:
            assert osp.exists(path)
            feat = np.load(path)
            query_feats.append(feat['global_desc'])

            a = feat['local_desc']
            b = feat['locations']
            c = feat['scales']
            del a, b, c
        except Exception as e:
            print('e', e)
            query_broken_files.append(query_lines[i])

    if query_broken_files:
        print('broken query file size:', len(query_broken_files))
        with open('./query_broken_files.txt', 'w') as f:
            f.writelines(query_broken_files)

    query_feats = np.stack(query_feats, axis=0)
    print('query_feats shape', query_feats.shape)
    np.save('/root/halo/code/MSFFT/datasets/TokyoTM/delg_feats/delg_qfeat', query_feats)

    query_feats = query_feats / LA.norm(query_feats, axis=-1)[:,None]

    index_feats = []
    db_path = '/root/halo/code/MSFFT/datasets/TokyoTM/delg_feats/index'
    for i in tqdm(range(len(gallery_lines))):
        # imgname = osp.basename(gallery_lines[i]).split('.')[0]
        imgname = Path(osp.basename(gallery_lines[i])).stem
        path = osp.join(db_path, imgname+'.npz')

        try:
            assert osp.exists(path)
            feat = np.load(path)
            index_feats.append(feat['global_desc'])

            a = feat['local_desc']
            b = feat['locations']
            c = feat['scales']
            del a, b, c
        except Exception as e:
            print('e', e)
            index_broken_files.append(gallery_lines[i])

        # path = osp.join(db_path, imgname+'.delg_global')
        # index_feats.append(datum_io.ReadFromFile(path))

    if index_broken_files:
        print('broken index file size:', len(index_broken_files))
        with open('./index_broken_files.txt', 'w') as f:
            f.writelines(index_broken_files)

    if query_broken_files or index_broken_files:
        print('total broken file', len(query_broken_files) + len(index_broken_files))
        exit(-1)

    index_feats = np.stack(index_feats, axis=0)
    np.save('/root/halo/code/MSFFT/datasets/TokyoTM/delg_feats/delg_dbfeat', index_feats)

    index_feats = index_feats / LA.norm(index_feats, axis=-1)[:,None]

    sims = np.matmul(query_feats, index_feats.T)
    print('sim shape', sims.shape)

    nn_inds  = np.argsort(-sims, -1)
    print('nn_inds.shape', nn_inds.shape)

    np.save('/root/halo/code/MSFFT/datasets/TokyoTM/delg_feats/delg_tokyoTM_rank_index', nn_inds)
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