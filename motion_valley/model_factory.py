"""Explicit architecture dispatch; existing experiment configurations stay legacy."""
from .model266 import Motion266Flow
from .model_no_text import NoTextMotion266Flow
from .model_wan266 import WanMotion266


def build_control_model(config):
    architecture=config.get('architecture','legacy_axial')
    if architecture=='kimodo_frame_twostage_266_v1':
        from .model_kimodo266 import Kimodo266
        if config['text_enabled'] or config['mode']!='uniform' or config.get('input_mode')!='concat_mask':
            raise ValueError('Kimodo266 requires no-text, uniform field and concat_mask')
        return Kimodo266(**{k:config[k] for k in ['width','depth','heads','ffn_dim','sampling_steps']},
                         register_tokens=config.get('register_tokens',50))
    common={key:config[key] for key in ['width','depth','heads','text_dim']}
    if architecture=='legacy_axial':
        cls=Motion266Flow if config['text_enabled'] else NoTextMotion266Flow
        return cls(**common)
    if architecture=='wan_joint_time_v1':
        extra={key:config[key] for key in ['ffn_dim','freq_dim','time_embedding_scale']}
        if 'input_mode' in config: extra['input_mode']=config['input_mode']
        return WanMotion266(**common,**extra,text_enabled=config['text_enabled'])
    raise ValueError(f'Unknown architecture: {architecture}')
