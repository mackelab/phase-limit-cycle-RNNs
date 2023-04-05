import torch
import numpy as np
from torch.utils.data import Dataset

class seqDS(Dataset):
    """
    A Dataset class for phase coding working memory task
    """
   
    def __init__(self,task_params,dataset_len=256):
        """
        Initialise dataset object
        Args:
            task_params: dictionary of task parameters
        """
        self.task_params = task_params
        self.T = task_params["stim_dur"][-1]+ \
                          task_params["stim_on"][-1]+ \
                        task_params["probe_dur"]
        
        self.time = torch.arange(0,self.T, self.task_params["dt"]/1000, dtype =torch.float32)
        self.trial_len = len(self.time)
        
        # this is not really important as trials are randomly generated
        # needs to be bigger than training batch size, that's all
        self.len = dataset_len

    def __getitem__(self, idx):
        return self.create_stim()
    
    def __len__(self):
        return self.len
    
    def create_stim(self):
        """
        Create a single trial, stimulus, mask and target
        
        Returns:
            inp: tensor containing trial input
            target: tensor containing target output
            mask: tensor containing mask
        """

        #convert dt from ms to seconds
        dt_sec = self.task_params["dt"]/1000

        #stim onset duration
        if len(self.task_params["stim_on"])>1:
            stim_start = torch.randint(low=int(self.task_params["stim_on"][0]/dt_sec),
                                  high=int(self.task_params["stim_on"][1]/dt_sec), size=(1, ))
        else:
            stim_start = int(self.task_params["stim_on"][0]/dt_sec)
            
        # stim durataion
        if len(self.task_params["stim_dur"])>1:
            stim_dur = torch.randint(low=int(self.task_params["stim_dur"][0]/dt_sec),
                                    high=int(self.task_params["stim_dur"][1]/dt_sec), size=(1, ))
        else:
            stim_dur = int(self.task_params["stim_dur"][0]/dt_sec)

        # target phase offset
        offsets = [of*np.pi*2 for of in self.task_params["offsets"]]
        n_stim = len(offsets)
        n_inp = self.task_params["n_osc"] + n_stim
        self.phase = torch.zeros_like(self.time)

        #potentially sample random frequency and amplitude
        eps= 1e-6
        if self.task_params["freq_var"]>eps and self.task_params["amp_var"]>eps:
            samp = np.random.multivariate_normal([self.task_params["freq"],1], np.array([[self.task_params["freq_var"],self.task_params["freq_amp_covar"]],
                                                                    [self.task_params["freq_amp_covar"],self.task_params["amp_var"]]]))#, size=None, check_valid='warn', tol=1e-8)
            freq = samp[0]
            amp = samp[1]
        elif self.task_params["freq_var"]>eps:
            samp = np.random.normal(self.task_params["freq"], np.sqrt(self.task_params["freq_var"]))#, size=None, check_valid='warn', tol=1e-8)
            freq = samp
            amp = 1
        elif self.task_params["amp_var"]>eps:
            samp = np.random.multivariate_normal(1, np.sqrt(self.task_params["amp_var"]))#, size=None, check_valid='warn', tol=1e-8)
            freq = self.task_params["freq"]
            amp = samp
        else:
            freq = self.task_params["freq"]
            amp = 1

        # initialise phase array
        dw = np.pi*freq*2
        w0 = torch.rand(1)*np.pi*2
        self.phase=torch.arange(len(self.phase))*dw*dt_sec+w0

        # create input array
        inp = torch.zeros((self.trial_len,n_inp), dtype =torch.float32)
        inp[:,:self.task_params["n_osc"]] = torch.tile(self.phase, (self.task_params["n_osc"],1)).T        
        
        #draw label
        label = torch.randint(0,n_stim, size = (1, ))

        #create target 
        target = torch.tile(self.phase, (1,1)).T        
        target[stim_start:]+=offsets[label]
        
        # create mask
        mask = torch.unsqueeze(torch.ones_like(target[:,0]),-1)
        mask[0:stim_start+stim_dur]=0

        #stimulus input
        inp[stim_start:stim_start+stim_dur,-(label+1)]=1
              
        #add cosine?
        if self.task_params["n_osc"]==2:
            inp[:,1] = torch.cos(inp[:,0])*amp

        #convert phase to sine wave
        inp[:,0] = self.task_params["signal"](inp[:,0])*amp
        inp[:,0]+=torch.randn(inp[:,0].size())*np.sqrt(dt_sec)*self.task_params["noise_sin"]
        target[:,0] = torch.sin(target[:,0])    
        target*=self.task_params["out_scale"]

        return inp, target, mask
