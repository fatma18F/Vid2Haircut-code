from pyhocon import ConfigFactory
import torch


def get_config(conf_path):
    f = open(conf_path)

    conf_text = f.read()
    f.close()

    config = ConfigFactory.parse_string(conf_text)
    return config



def create_coarse_model(projector_type_elow, config, ckpt_path_elow, device):

    elow = create_projector_backbone(projector_type_elow, config)
    checkpoint = torch.load(ckpt_path_elow, map_location=device)
    state_dict = checkpoint['lp_enc']

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')  # Remove `module.` prefix
        new_state_dict[new_key] = v


    elow.load_state_dict(new_state_dict)
    elow.to(device)
    elow.eval()

    params_number =  sum(param.numel() for param in elow.parameters())
    print(f'load ckpt {ckpt_path_elow} in coarse model with {params_number}')

    return elow



def create_projector_backbone(projector_type, config):

    if projector_type == 'fine':
        from modules.fine import Lp3DEncoder
        lp_enc = Lp3DEncoder(**config['lp_encoder_fine'])
        print('Projector type is fine')

    elif projector_type == 'coarse':
        from modules.coarse import Lp3DEncoder
        lp_enc = Lp3DEncoder(**config['lp_encoder'])
        print('Projector type is coarse')
    else:
        print('proj not found')
        
    return lp_enc
