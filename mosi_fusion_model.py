import torch
from torch.autograd import Variable 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence
import time
import numpy as np
from mosi_helper import *



class ModelIO():
    '''
    The ModelIO class implements a load() and a save() method that
    makes model loading and saving easier. Using these functions not
    only saves the state_dict but other important parameters as well from
    __dict__. If you instantiate from this class, please make sure all the
    required arguments of the __init__ method are actually saved in the class
    (i.e. self.<param> = param). 
    That way, it is possible to load a model with the default parameters and
    then change the parameters to correct values from stored in the disk.
    '''
    ignore_keys = ['_backward_hooks','_forward_pre_hooks','_backend',\
        '_forward_hooks']#,'_modules','_parameters','_buffers']
    def save(self, fout):
        '''
        Save the model parameters (both from __dict__ and state_dict())
        @param fout: It is a file like object for writing the model contents.
        '''
        model_content={}
        # Save internal parameters
        for akey in self.__dict__:
            if not akey in self.ignore_keys:
                model_content.update(self.__dict__)
        # Save state dictionary
        model_content['state_dict']=self.state_dict()
        try:
            torch.save(model_content,fout)
        except:
            time.sleep(5)
            torch.save(model_content,fout)

    def load(self,fin,map_location=None):
        '''
        Loads the parameters saved using the save method
        @param fin: It is a file-like obkect for reading the model contents.
        @param map_location: map_location parameter from
        https://pytorch.org/docs/stable/torch.html#torch.load
        Note: although map_location can move a model to cpu or gpu,
        it doesn't change the internal model flag refering gpu or cpu.
        '''
        data=torch.load(fin,map_location)
        self.__dict__.update({key:val for key,val in data.items() \
            if not key=='state_dict'})
        self.load_state_dict(data['state_dict'])



class LSTM_custom(nn.Module):
    '''
    A custom implementation of LSTM in pytorch. Donot use. VERY slow
    '''
    def __init__(self,input_dim,hidden_dim):
        super(LSTM_custom,self).__init__()
        self.W_xi = nn.Linear(input_dim,hidden_dim)
        self.W_hi = nn.Linear(hidden_dim,hidden_dim)
        self.W_xf = nn.Linear(input_dim,hidden_dim)
        self.W_hf = nn.Linear(hidden_dim,hidden_dim)
        self.W_xg = nn.Linear(input_dim,hidden_dim)
        self.W_hg = nn.Linear(hidden_dim,hidden_dim)
        self.W_xo = nn.Linear(input_dim,hidden_dim)
        self.W_ho = nn.Linear(hidden_dim,hidden_dim)
        self.drop = nn.Dropout(0.2)


    def dropout(self):
        s_dict=self.W_hi.state_dict()
        s_dict['weight']=self.drop(s_dict['weight'])
        self.W_hi.load_state_dict(s_dict)

        s_dict=self.W_hf.state_dict()
        s_dict['weight']=self.drop(s_dict['weight'])
        self.W_hf.load_state_dict(s_dict)

        s_dict=self.W_hg.state_dict()
        s_dict['weight']=self.drop(s_dict['weight'])
        self.W_hg.load_state_dict(s_dict)

        s_dict=self.W_ho.state_dict()
        s_dict['weight']=self.drop(s_dict['weight'])
        self.W_ho.load_state_dict(s_dict)

    def forward(self,x,hidden):
        h,c = hidden[0],hidden[1]
        i = torch.sigmoid(self.W_xi(x) + self.W_hi(h))
        f = torch.sigmoid(self.W_xf(x) + self.W_hf(h))
        g = torch.tanh(self.W_xg(x) + self.W_hg(h))
        o = torch.sigmoid(self.W_xo(x)+self.W_ho(h))
        c_ = f*c + i*g
        h_ = o * torch.tanh(c_)
        return h_,c_


class MOSI_fusion_classifier(nn.Module,ModelIO):

    def init_hidden(self,hidden_dim):
        return (variablize(torch.zeros(1, hidden_dim)),variablize(torch.zeros(1, hidden_dim)))

    def __init__(self,lan_param,audio_param,face_param,out_dim):
        super(MOSI_fusion_classifier,self).__init__()
        self.lan_lstm=LSTM_custom(lan_param['input_dim'],lan_param['hidden_dim'])
        #self.face_lstm=LSTM_custom(face_param['input_dim'],face_param['hidden_dim'])
        #self.audio_lstm=LSTM_custom(audio_param['input_dim'],audio_param['hidden_dim'])
        self.z=lan_param['hidden_dim']
        #self.z=face_param['hidden_dim']
        #self.z=audio_param['hidden_dim']
        self.W_fhout = nn.Linear(self.z,out_dim)
        self.h = self.init_hidden(lan_param['hidden_dim'])
        #self.h = self.init_hidden(face_param['hidden_dim'])
        #self.h = self.init_hidden(audio_param['hidden_dim'])


    def forward(self,opinion):
        for i,x in enumerate(opinion):
            x_lan,x_audio,x_face=filter_train_features(x)
            x_lan=variablize(torch.FloatTensor(x_lan))
            x_audio=variablize(torch.FloatTensor(x_audio))
            x_face=variablize(torch.FloatTensor(x_face))

            if i==0:
                if self.training:
                    self.lan_lstm.dropout()
                    #self.face_lstm.dropout()
                    #self.audio_lstm.dropout()
                h_x,c_x=self.lan_lstm.forward(x_lan,self.h)
                #h_x,c_x=self.face_lstm.forward(x_face,self.h)
                #h_x,c_x=self.audio_lstm.forward(x_audio,self.h)
            else:
                h_x,c_x=self.lan_lstm.forward(x_lan,(h_x,c_x))
                #h_x,c_x=self.face_lstm.forward(x_face,(h_x,c_x))
                #h_x,c_x=self.audio_lstm.forward(x_audio,(h_x,c_x))

        out=self.W_fhout(h_x)
        # return torch.sigmoid(out)
        return out


def test_model():
    opinion=np.array([[1,2,3,4,5,6,7,8],[3,4,5,6,7,8,9,10]])
    input_dim=3
    lstm_hidden_dim=2
    out_dim=1

    mosi_model=MOSI_fusion_classifier(input_dim,lstm_hidden_dim,out_dim)
    # print mosi_model
    print mosi_model.forward(opinion)


# test_model()