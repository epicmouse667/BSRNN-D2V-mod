import torch.nn as nn
import torch
import torch.nn.functional as F

from wesep.models.bsrnn import BSRNN
from transformers import AutoProcessor, Data2VecAudioForCTC

class BSRNN_D2V_pipeline(nn.Module):
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
        super(BSRNN_D2V_pipeline,self).__init__()
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
        )
        if pretrained_d2v4ctc_dir is None:
            raise ValueError
        if pretrained_extraction_path is not None:
            self.separator.load_state_dict(torch.load(pretrained_extraction_path))
        self.asr_model = Data2VecAudioForCTC.from_pretrained(pretrained_d2v4ctc_dir)
        self.D2V_Extractor = self.asr_model.data2vec_audio.feature_extractor
        self.D2V_feature_projection = self.asr_model.data2vec_audio.feature_projection
        self.D2V_tf_encoder = self.asr_model.data2vec_audio.encoder
        self.D2V_Decoder = self.asr_model.lm_head
        self.btf_layer_num = btf_layer_num
        self.autoprocessor = AutoProcessor.from_pretrained(pretrained_d2v4ctc_dir)
        for param in self.asr_model.parameters():
            param.requires_grad = False

    def forward(
        self,
        wav_inputs,
        processed_wav_mix,
        spk_embeddings,
    ):
        output,mask = self.separator(
            input=wav_inputs,
            embeddings=spk_embeddings
        )
        # output = torch.cat([self.autoprocessor(x,return_tensors='pt',sampling_rate=self.separator.sr)['input_values'] for x in output],dim=0).cuda()
        # print('output shape: ',output.shape)
        output_new = output.detach()
        extract_features = self.D2V_Extractor(output_new).transpose(1,2)
        hidden_states, _ = self.D2V_feature_projection(extract_features)
        position_embeddings = self.D2V_tf_encoder.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.D2V_tf_encoder.layer_norm(hidden_states)
        hidden_states = self.D2V_tf_encoder.dropout(hidden_states)
        # deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        for layer in self.D2V_tf_encoder.layers: 
            hidden_states = layer(hidden_states)[0]
        hidden_states = self.asr_model.dropout(hidden_states)
        logits = self.D2V_Decoder(hidden_states) 
        return  logits,output
        
        
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
        