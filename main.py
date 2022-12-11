from model.models import speech_model
import torchaudio as ta
import numpy as np
import torch

def get_fu(path_ = 'temp.wav'):
    _wavform, _ = ta.load_wav( path_ )
    _feature = ta.compliance.kaldi.fbank(_wavform, num_mel_bins=40) 
    _mean = torch.mean(_feature)
    _std = torch.std(_feature)
    _T_feature =  (_feature - _mean) / _std
    inst_T = _T_feature.unsqueeze(0)
    return inst_T
def main(path_ = 'temp.wav'):
    inst_T = get_fu( path_ )
    log_  = model_lo( inst_T )
    _pre_ = log_.transpose(0,1).detach().numpy()[0]
    liuiu = [dd for dd in _pre_.argmax(-1) if dd != 0]
    str_end = ''.join([ num_wor[dd] for dd in liuiu ])
    return str_end

model_lo = speech_model()
device_ = torch.device('cpu')
model_lo.load_state_dict(torch.load('models/sp_model.pt' , map_location=device_))
model_lo.eval()
print ('00')
# num_wor = np.load('models/dic.dic.npy').item()
num_wor = np.load('models/dic.dic.npy',allow_pickle=True).item()

if __name__=='__main__':
    path_ = 'temp.wav'
    result_ = main(path_)
    print('\n')
    print ( '识别结果是： ' ,  result_ )
