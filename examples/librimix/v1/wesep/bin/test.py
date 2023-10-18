import sys
sys.path.append('/workspace2/zixin/wesep/examples/librimix/v1')
from wesep.models.bsrnn_d2v import BSRNN_D2V
from wesep.models.bsrnn_d2v_pipeline import BSRNN_D2V_pipeline
from wesep.models.bsrnn import BSRNN
from transformers import AutoProcessor
from tools.extract_embed_premix import compute_fbank
from wesep.utils.metrics import SISNR_CTC_Loss,SISNRLoss
import torchaudio
import onnxruntime as ort
import torch
import soundfile as sf
import torch.distributed as dist

model = BSRNN_D2V_pipeline(
    btf_layer_num= 2,
    feature_dim= 128,
    num_repeat= 6,
    pretrained_d2v4ctc_dir= '/workspace2/zixin/Download/wav2vec2-base-960h',
    # pretrained_extraction_path= 'exp/BSRNN/train100/FiLM_no_spk_transform/models/model_150.pt',
    return_mask= True,
    return_real_mask= True,
    spk_emb_dim= 256,
    spk_fuse_type= 'FiLM',
    sr= 16000,
    stride= 320,
    use_spk_transform= False,
    win= 1024,
)
# model = BSRNN(
#     win=1024,
#     stride =320,
#     spk_fuse_type='FiLM'
# )
epoch = 73
uttid = '1272-128104-0000_2035-147961-0014'
# dist
model.load_state_dict(torch.load('/workspace2/zixin/wesep/examples/librimix/v1/exp/BSRNN_D2V/train100/pipeline/models/model_{}.pt'.format(epoch)))
autoprocessor = AutoProcessor.from_pretrained('/workspace2/zixin/Download/wav2vec2-base-960h')
wav_input = torchaudio.load('/workspace2/zixin/Datasets/Libri2Mix/wav16k/max/dev/mix_clean/{}.wav'.format(uttid))[0]
processed_wav_mix = autoprocessor(wav_input,return_tensors ="pt",sampling_rate=16000)['input_values']
target_wav = torchaudio.load('/workspace2/zixin/Datasets/Libri2Mix/wav16k/max/dev/s1/{}.wav'.format(uttid))[0]
aux_speech_path ='/workspace2/zixin/Datasets/Libri2Mix/wav16k/max/dev/s1/1272-135031-0015_2277-149896-0006.wav'
feats = compute_fbank(aux_speech_path)
feats = feats.unsqueeze(0).numpy()
onnx_path = "/workspace2/zixin/wesep/examples/librimix/v1/data/voxceleb_resnet34_LM.onnx"
so = ort.SessionOptions()
session = ort.InferenceSession(onnx_path, sess_options=so)
embed = session.run(
                    output_names=['embs'],
                    input_feed={
                        'feats': feats
                    }
                )
with torch.no_grad():
    logits,output = model(
        wav_inputs = wav_input,
        processed_wav_mix = processed_wav_mix.view(1,-1),
        spk_embeddings = torch.tensor(embed[0])
    )
    # output = model(
    #     wav_input,
    #     torch.tensor(embed[0])
    # )
    sf.write('{}_7976_{}epoch.wav'.format(uttid,epoch),output.squeeze(0),16000)
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = autoprocessor.batch_decode(predicted_ids)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32)
    print(transcription)
    label = autoprocessor(text='''MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL''',
                              return_tensors = "pt").input_ids
    print(output.shape,log_probs.shape,label.shape)
    criterion = SISNR_CTC_Loss(0.05)
    loss,si_snr,_ = criterion(
        est_wav = output.cuda(),
        target_wav = target_wav.cuda(),
        wav_len = torch.tensor([output.shape[-1]]).unsqueeze(0).cuda(),
        log_prob = log_probs.cuda(),
        prob_len = torch.tensor([log_probs.shape[1]]).unsqueeze(0).cuda(),
        label = label.cuda(),
        label_len = torch.tensor([label.shape[-1]]).unsqueeze(0).cuda()
    )
    print('loss: ',loss,'si_snr: ',si_snr,'ctc loss: ',(loss-si_snr)/criterion.alpha)
   # MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL
   # MYBBILTER IS THE IMPOSTOR OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL  19 epoch
   # MOCTER WILTER IS THE IMPOSTIRE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL 32 epoch

