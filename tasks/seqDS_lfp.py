import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.signal import firwin
from neo.io.neuroscopeio import NeuroScopeIO
from scipy.signal import resample


class seqDS(Dataset):
    """
    A Dataset class for phase coding working memory task,
    using real LFP as reference oscillation
    """
   
    def __init__(self,task_params, path):
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

        # load and process LFP data, this takes a while!
        main_freq, self.data = extract_lfp_phase(resample_rate = task_params['resample_rate'], 
                                     filter_freqs = task_params['filter_freqs'], 
                                     phase_freqs = task_params['phase_freqs'], 
                                     rat = task_params['rat'], 
                                     max_dur = task_params["max_dur"],
                                     buffer = task_params["buffer"],
                                     path = path,
                                     tolerance = task_params["artifact_tol"])
        self.len = len(self.data)
        self.trial_len = len(self.time)
        task_params["main_freq"]=main_freq
    
    def __getitem__(self, idx):
        return self.create_stim(idx)
    
    def __len__(self):
        return self.len
    
    def create_stim(self,idx):
        """
        Create a single trial, stimulus, mask and target
        
        Args:
            idx: index of trial
        
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

        #create input array
        inp = torch.zeros((self.trial_len,n_inp), dtype =torch.float32)
        
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
        
        # add LFP signal
        sig = self.data[idx][0,:self.trial_len]
        inp[:,0] = torch.from_numpy(sig)

        if self.task_params["n_out"]==2:
            target[:,1] = torch.cos(target[:,0])

        #convert phase to sine wave
        target[:,0] = torch.sin(target[:,0])
        target*=self.task_params["out_scale"]

        return inp, target, mask

def extract_phase(dat, time, dt,freqs,ref_phase):
    """
    Extract phase from LFP signal

    Args:
        dat: LFP signal
        time: time vector
        dt: sampling time step in s
        freqs: frequencies to consider
        ref_phase: whether to compute relative phase, Boolean

    Returns:
        freqs: frequency with highest power
        phase: phase of signal at frequency with highest power
    """
    phase, amp = scalogram(
        dat,
        7,
        time,
        dt,
        freqs,
        ref_phase=ref_phase
    )
    main_power = np.argmax(np.mean(amp, axis=1))
    return freqs[main_power],phase[main_power]
def extract_lfp_phase(resample_rate, filter_freqs, phase_freqs, rat,max_dur,buffer, path, tolerance = 1):
    """
    Extract LFP phases from raw data
    You first have to download this dataset:
    http://crcns.org/data-sets/hc/hc-2/about-hc-2
    Mizuseki K, Sirota A, Pastalkova E, Buzsáki G. (2009):
    Multi-unit recordings from the rat hippocampus made during open field foraging.
    http://dx.doi.org/10.6080/K0Z60KZ9

    Args:
        resample_rate: resampling rate, scaler
        filter_freqs: frequency band to filter LFP signal
        phase_freqs: list of frequencies to consider for phase extraction
        rat: rat number (1-3)
        max_dur: maximum duration of trial in s
        buffer: buffer duration in s, to avoid boundary effects
        path: path to data
        tolerance: tolerance for artifact rejection
    
    Returns:
        main_freq: frequency with highest power for each tiral
        data: LFP data for each trial
    """

    files13=['ec013.527.xml',
           'ec013.528.xml',
           'ec013.529.xml',
           'ec013.713.xml',
           'ec013.714.xml',
           'ec013.754.xml',
           'ec013.755.xml',
           'ec013.756.xml',
           'ec013.757.xml',
           'ec013.808.xml',
           'ec013.844.xml', 
          ]

    files14=['ec014.277.xml',
           'ec014.333.xml',
           'ec014.793.xml',
           'ec014.800.xml',
           'ec015.041.xml',
           'ec015.047.xml',
          ]

    files16=['ec016.397.xml',
           'ec016.430.xml',
           'ec016.448.xml',
           'ec016.582.xml',
          ]
    print("rat = " +str(rat))
    if np.isclose(rat,1):
        print("Rat 1")
        files = files13
    elif np.isclose(rat,2):
        print("Rat 2")
        files = files14
    elif np.isclose(rat,3):
        print("Rat 3")
        files = files16
    else:
        print("RAT NOT RECOGNISED")
        files=None

    fs=1250
    fs/=resample_rate
    neo_data = []
    n_artifacts = 0

    #loop through files
    for file in files[:1]:
        filen= path+"/"+file
        reader = NeuroScopeIO(filename=filen)
        seg = reader.read_segment()

        # get lfp
        lfp = np.array(seg.analogsignals[0])

        # resample
        n_samples = int(len(lfp)/resample_rate)
        lfp = resample(lfp,n_samples,axis=0)

        # filter
        if len(filter_freqs)<1:
            lfpf = lfp
            lfpf -= np.mean(lfpf, axis = 0,keepdims=True)
        elif len(filter_freqs)>1:
            lfpf = firwin_filter_band(lfp,filter_freqs,fs)
        else:
            lfpf = firwin_filter_high(lfp,filter_freqs,fs)

        # normalize
        amp_mod = 1/(np.sqrt(2)*(np.std(lfpf,axis=0,keepdims=True)))
        lfpf*=amp_mod

        neo_data.append(lfpf)
        print(str(len(lfpf)/fs)+"s of data")
        print("n_channels = " + str(np.shape(seg.analogsignals[0])[1]))

    max_trial_length = int(max_dur*fs)
    start_trial = int(buffer*fs)

    data = [] 
    main_freq=[]
    t=np.arange(0,max_dur,1/fs)
    freqs = phase_freqs

    #extract phases and frequencies
    for session in neo_data:
        channel_inds = np.random.choice(np.arange(session.shape[1]),int(len(session)/max_trial_length)+1)
        for channel_ind, trial_ind in enumerate(np.arange(0,len(session)-max_trial_length,max_trial_length)):
            trial = session[trial_ind:trial_ind+max_trial_length,channel_inds[channel_ind]]
            # check artifact
            if artifact(trial, tolerance):
                n_artifacts +=1
            else:
                freq,phase = extract_phase(trial,time=t[:len(trial)],dt=1/fs,freqs=freqs,ref_phase=False)
                data.append(np.array([trial[start_trial:],phase[start_trial:]+0.5*np.pi]))
            main_freq.append(freq)
    print("Detected " + str(n_artifacts) + " artifacts")
    return main_freq, data



def wrap(x):
    """returns angle with range [-pi, pi]"""
    return np.arctan2(np.sin(x), np.cos(x))


def complex_wavelet(timestep, freq, cycles, kernel_length=5):
    """
    Create wavelet of a certain frequency

    Args:
        timestep: simulation timestep in seconds
        freq: frequency of the wavelet
        cycles: number of oscillations of wavelet
        kernel_length: adapted per frequency
    Note:
        normalisation as in: https://www.frontiersin.org/articles/10.3389/fnhum.2010.00198/full#B22
        retains signal energy, irrespective of freq, sum of the length of the wavelet is 1
    """

    gauss_sd = cycles / (2 * np.pi * freq)
    t = np.arange(0, kernel_length * gauss_sd, timestep)
    t = np.r_[-t[::-1], t[1:]]
    gauss = (1 / (np.sqrt(2 * np.pi) * gauss_sd)) * np.exp(
        -(t**2) / (2 * gauss_sd**2)
    )
    sine = np.exp(2j * np.pi * freq * t)
    wavelet = gauss * sine * timestep

    return wavelet


def inst_phase(sign, kernel, t, f, ref_phase=True, mode="same"):
    """
    Calculate instaneous angle and magnitude at certain frequency

    Args:
        sign: signal to analyse
        kernel: convolve with this, e.g. complex wavelet
        t: array of timesteps
        f: frequency to extract
        ref_phase: calculate phase with respect to sinusoid
        mode: convolution mode (see numpy.convolve)
    Returns:
        phase: phase at each timestep
        amp: amplitude at each timestep

    Note:
        normalisation as in: https://www.frontiersin.org/articles/10.3389/fnhum.2010.00198/full#B22
        retains signal energy, irrespective of freq, sum of the length of the wavelet is 1
    """

    conv = np.convolve(sign, kernel, mode=mode)

    # cut off more in case kernel is too long
    if len(conv) > len(sign):
        st = (len(conv) - len(sign)) // 2
        conv = conv[st : st + len(sign)]
    amp = np.abs(conv)
    arg = np.angle(conv)
    if ref_phase:
        ref = wrap(2 * np.pi * f * t)
        phase = wrap(arg - ref)
    else:
        phase = arg
    return phase, amp


def scalogram(sign, cycles, t, timestep, freqs, kernel_length=5, ref_phase=True):
    """
    Create a scalogram of a signal, using complex wavelets

    Args:
        sign: signal to analyse
        t: array of timesteps
        timestep: timestep in seconds
        freqs: list of frequencies to extract
        timestep: simulation timestep in seconds
        cycles: number of oscillations of wavelet
        kernel_length: adapted per frequency

    Returns:
        phasegram: phase at each timestep, for each frequency
        ampgram: amplitude at each timestep, for each frequency

    Note:
        normalisation as in: https://www.frontiersin.org/articles/10.3389/fnhum.2010.00198/full#B22
        retains signal energy, irrespective of freq, sum of the length of the wavelet is 1
    """

    phasegram = np.zeros((len(freqs), len(t)))
    ampgram = np.zeros((len(freqs), len(t)))
    for i, f in enumerate(freqs):
        kernel = complex_wavelet(timestep, f, cycles, kernel_length)
        phase, amp = inst_phase(sign, kernel, t, f, ref_phase=ref_phase)
        phasegram[i] = phase
        ampgram[i] = amp * np.sqrt(2)
    return phasegram, ampgram

def firwin_filter_high(data,cutoff,fs):
    filt = firwin(511, cutoff, pass_zero='highpass', fs=fs)
    y=np.apply_along_axis(lambda m: np.convolve(m, filt, mode='same'), axis=0, arr=data)

    return y


def firwin_filter_band(data,cutoff,fs):
    filt = firwin(512, cutoff, pass_zero='bandpass', fs=fs)
    y=np.apply_along_axis(lambda m: np.convolve(m, filt, mode='same'), axis=0, arr=data)
    return y

def artifact(data, tolerance):
    if np.max(abs(data))>(1+tolerance):
        return True
    return False