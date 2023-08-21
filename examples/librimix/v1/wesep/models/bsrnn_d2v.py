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

class BSRNN_D2V(nn.Module):
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
        return_real_mask=True,
        pretrained_extraction_path=None,
    ):
        super(BSRNN_D2V,self).__init__()
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
            return_real_mask=return_real_mask
        )
        if pretrained_d2v4ctc_dir is None:
            raise ValueError
        if pretrained_extraction_path is not None:
            self.separator.load_state_dict(torch.load(pretrained_extraction_path))
        self.asr_model = Data2VecAudioForCTC.from_pretrained(pretrained_d2v4ctc_dir)
        self.interlayer_T = IntermediateLayer(
            featdim = self.separator.enc_dim,
            hidden_size = self.asr_model.config.hidden_size,
        )

        # self.interlayer_B = IntermediateLayer(
        #     featdim = self.asr_model.config.hidden_size,
        #     hidden_size = self.separator.feature_dim ,
        # )
        self.D2V_Extractor = self.asr_model.data2vec_audio.feature_extractor
        self.D2V_feature_projection = self.asr_model.data2vec_audio.feature_projection
        self.D2V_tf_encoder = self.asr_model.data2vec_audio.encoder
        self.D2V_Decoder = self.asr_model.lm_head
        self.btf_layer_num = btf_layer_num
        for param in self.asr_model.parameters():
            param.requires_grad = False

    def forward(
        self,
        wav_inputs,
        spk_embeddings,
    ):
        output,mask = self.separator(
            input=wav_inputs,
            embeddings=spk_embeddings
        )
        extract_features = self.D2V_Extractor(wav_inputs).transpose(1,2)
        hidden_states, _ = self.D2V_feature_projection(extract_features)
        position_embeddings = self.D2V_tf_encoder.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.D2V_tf_encoder.layer_norm(hidden_states)
        hidden_states = self.D2V_tf_encoder.dropout(hidden_states)
        # deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        for layer in self.D2V_tf_encoder.layers[:self.btf_layer_num]: 
            hidden_states = layer(hidden_states)[0]
        #mask B,F,T    hidden_state  B,T,D
        mask = self.interlayer_T(mask)
        assert mask.shape[-1] == hidden_states.shape[-1]
        T_hidden_state,T_mask = hidden_states.shape[1],mask.shape[1]
        if T_hidden_state > T_mask:
            mask = F.pad(mask, (0,0,0,T_hidden_state-T_mask))
        else:
            mask = mask[:,:T_hidden_state,:]
        assert hidden_states.shape == mask.shape,'hidden_states shape: {} , mask shape: {}'\
            .format(hidden_states.shape,mask.shape)
        hidden_states = hidden_states * mask

        for layer in self.D2V_tf_encoder.layers[self.btf_layer_num:]:
            hidden_states = layer(hidden_states)[0]
        hidden_states = self.asr_model.dropout(hidden_states)
        logits = self.D2V_Decoder(hidden_states) 
        log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        return  log_probs,output
        