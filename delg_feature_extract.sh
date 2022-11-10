
###################################################################
###################### delg feature extraction ####################
###################################################################

# for tokyo247 query
python3 delg_feature_extract.py \
  --delf_config_path r50delg_gldv2clean_config.pbtxt \
  --dataset_file_path dataset/Tokyo247/tokyo247_query_c.txt  \
  --output_features_dir dataset/Tokyo247/delg_feats/query

# for tokyo247 index
python3 delg_feature_extract.py \
  --delf_config_path r50delg_gldv2clean_config.pbtxt \
  --dataset_file_path dataset/Tokyo247/tokyo247_db_c.txt  \
  --output_features_dir dataset/Tokyo247/delg_feats/index

