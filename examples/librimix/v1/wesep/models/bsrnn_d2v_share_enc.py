import torch.nn as nn
import torch
import torch.nn.functional as F

from wesep.models.bsrnn import BSRNN
from transformers import AutoProcessor, Data2VecAudioForCTC

class IntermediateLayer(nn.Module):
    def __init__(self,hidden_size,featdim):
        super(IntermediateLayer,self).__init__()
        self.featdim=featdim
        self.Conv = nn.Conv1d(in_channels=featdim,out_channels=hidden_size,kernel_size=1)  

    def forward(self,input):
        if input.shape[1] !=self.featdim:
            input = input.transpose(1,2)
        hidden_state = F.relu(self.Conv(input).transpose(1,2))
        return hidden_state


class DecoderLayer(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            stride,
            kernel_size,
    ):
        super(DecoderLayer,self).__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.layer_norm = nn.LayerNorm(
            out_channel,
            eps = 1.e-5,
            elementwise_affine=True
        )
        self.activation = nn.ReLU()
    def forward(
            self,
            input,
    ):
        hidden_states = self.deconv(input).transpose(-1,-2)
        hidden_states = self.layer_norm(hidden_states).transpose(-1,-2)
        hidden_states = self.activation(hidden_states)
        return hidden_states
class BSRNN_D2V_share_enc(nn.Module):
    def __init__(
        self,
        spk_emb_dim=256,
        sr=16000,
        win=512,
        stride=128,
        feature_dim=128,
        num_repeat=6,
        use_spk_transform=False,
        use_bidirectional=True,
        spk_fuse_type='concat',
        btf_layer_num =2,
        pretrained_d2v4ctc_dir=None,
        return_mask =True,
        return_real_mask=False,
        pretrained_extraction_path=None,
    ):
        super(BSRNN_D2V_share_enc,self).__init__()
        self.separator = BSRNN(
            spk_emb_dim=spk_emb_dim,
            sr=sr,
            win=win,
            stride=stride,
            feature_dim=feature_dim,
            num_repeat=num_repeat,
            use_spk_transform=use_spk_transform,
            use_bidirectional=use_bidirectional,
            spk_fuse_type=spk_fuse_type,
            return_mask = return_mask,
            return_real_mask=return_real_mask,
            share_encoder = True      
        )
        if pretrained_d2v4ctc_dir is None:
            raise ValueError
        if pretrained_extraction_path is not None:
            self.separator.load_state_dict(torch.load(pretrained_extraction_path))
        self.asr_model = Data2VecAudioForCTC.from_pretrained(pretrained_d2v4ctc_dir)
        self.interlayer_T_d2v = IntermediateLayer(
            featdim = self.separator.enc_dim ,
            hidden_size = self.asr_model.config.hidden_size,
        )
        self.interlayer_B = IntermediateLayer(
            featdim = self.asr_model.config.hidden_size,
            hidden_size = self.separator.enc_dim ,
        )
        self.D2V_Extractor = self.asr_model.data2vec_audio.feature_extractor
        self.interlayer_T_bsrnn = nn.Conv1d(
            self.separator.enc_dim,
            self.D2V_Extractor.conv_layers[-1].conv.out_channels,
            1
        )
        self.D2V_feature_projection = self.asr_model.data2vec_audio.feature_projection
        self.D2V_tf_encoder = self.asr_model.data2vec_audio.encoder
        self.D2V_Decoder = self.asr_model.lm_head
        self.btf_layer_num = btf_layer_num
        self.BSRNN_Decoder = []
        for layer in self.D2V_Extractor.conv_layers:
            self.BSRNN_Decoder.append(
                DecoderLayer(
                    in_channel = layer.conv.out_channels,
                    out_channel = layer.conv.in_channels,
                    kernel_size = layer.conv.kernel_size[0],
                    stride  = layer.conv.stride[0]
                )
            )
        self.BSRNN_Decoder.reverse()
        self.BSRNN_Decoder = nn.Sequential(*self.BSRNN_Decoder)
        for layer in self.BSRNN_Decoder:
            nn.init.xavier_uniform_(layer.deconv.weight.data)
        for param in self.asr_model.parameters():
            param.requires_grad = False

        self.bn = nn.BatchNorm1d(self.D2V_Extractor.conv_layers[-1].conv.out_channels)

    def forward(
        self,
        wav_inputs,
        processed_wav_mix,
        spk_embeddings,
    ):
        extract_features = self.D2V_Extractor(wav_inputs).transpose(1,2)
        hidden_states, _ = self.D2V_feature_projection(extract_features)
        position_embeddings = self.D2V_tf_encoder.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.D2V_tf_encoder.layer_norm(hidden_states)
        hidden_states = self.D2V_tf_encoder.dropout(hidden_states)

        for layer in self.D2V_tf_encoder.layers[:self.btf_layer_num]: 
            hidden_states = layer(hidden_states)[0]
        #mask B,F,T    hidden_state  B,T,D

        real_spec = self.interlayer_B(hidden_states)
        # sepc = torch.cat([real_spec.unsqueeze(1),imag_spec.unsqueeze(1)],dim=1)
        est_spec, _= self.separator(
            input=real_spec,
            embeddings=spk_embeddings,
            n_sample = processed_wav_mix.shape[-1]
        )
        est_spec = self.interlayer_T_bsrnn(est_spec)
        est_spec = self.bn(est_spec)
        est_wav = self.BSRNN_Decoder(est_spec).squeeze(1)
        print('est_wav',est_wav)
        return est_wav
        
        
    def freeze_extraction_branch(
        self
    ):
        for param in self.separator.parameters():
            param.requires_grad = False

    def unfreeze_extraction_branch(
        self
    ):
        for param in self.separator.parameters():
            param.requires_grad = True

    def unfreeze_extraction_mask(
            self,
    ):
        for param in self.separator.mask.parameters():
            param.requires_grad = True

    def reset_param_recur(self):    
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()