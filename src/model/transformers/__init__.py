from .transformers import TransformerEncoder

def build_transformer(config):
    return TransformerEncoder(
        image_size = config['image_size'],
        channels = config['channels'],
        patch_size = config['patch_size'],
        hidden_dim = config['hidden_dim'],
        depth = config['depth'],
        heads = config['heads'],
        ffn_dim = config['ffn_dim'],
        dropout = config['dropout'],
        emb_dropout = config['emb_dropout']
    )