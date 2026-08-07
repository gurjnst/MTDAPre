"""
Quick‑test for MTDAPre model on WeatherBench dataset.
"""
import torch
from thop import profile, clever_format
from openstl.models.mtdapre_t import MTDAPre_Model

if __name__ == "__main__":
    model_config = {
        # image h w c
        'height': 32,
        'width': 64,
        # SISO 1->1  MISO 5->1  MIMO 5->5
        'num_channels': 5,
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
        'auxdepth': 4,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Test device: {device}\n")
    model = MTDAPre_Model(model_config).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:    {total_params:,}")
    print(f"Trainable params:{trainable_params:,}\n")

    dummy_input = torch.randn(
        1,
        model_config['pre_seq'],
        model_config['num_channels'],
        model_config['height'],
        model_config['width']
    ).to(device)

    with torch.no_grad():
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    flops_str, params_str = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops_str}")
    print(f"Params(thop): {params_str}")

    out = model(dummy_input)
    expected_shape = (1, model_config['pre_seq'], model_config['out_channels'], 32, 64)
    print(f"\nInput shape:{tuple(dummy_input.shape)}")
    print(f"Output shape:{tuple(out.shape)}")
    assert out.shape == expected_shape