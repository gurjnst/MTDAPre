# SISO
batch_size = 16
val_batch_size = 16
epoch = 50
opt = 'adamw'
weight_decay = 0.01
sched = 'cosine'  # onecycle cosine
lr = 1e-3  #5e-4  1e-3

dataname = 'weather'
method = 'MTDAPre'
model = 'Early'  # Early
model_config = {
    # image h w c
    'height': 32,
    'width': 64,
    # SISO 1->1  MISO 5->1  MIMO 5->5
    'num_channels': 1,
    'out_channels': 1,
    # video length in and out
    'pre_seq': 12,
    'after_seq': 12,
    # patch size
    'patch_size': 4,
    'dim': 128,
    'heads': 8,
    'dim_head': 16,
    # dropout
    'dropout': 0.1,
    'attn_dropout': 0.1,
    'drop_path': 0.2,
    'scale_dim': 4,
    # depth
    'depth': 1,
    'Ndepth': 6,
}