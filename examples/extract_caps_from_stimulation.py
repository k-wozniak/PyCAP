import scipy
import scipy.io
import scipy.signal
import numpy as np

class StimulationData:
    """ Keeps track of the stimulation data. """
    raw_data: np.ndarray
    sample_rate: float
    stimulus_pulses: np.ndarray
    stimulation_locations = np.ndarray

    def __init__(self, path: str, raw_data = "rawdata", sample_rate = "sampleRate", stimulus_pulse = "stimulusPulse", remove_from_start = 2):
        loaded_data = scipy.io.loadmat(path)

        self.raw_data = loaded_data[raw_data]
        # For some reason ofthen array in an array, so any operation such as sum, 
        # mean will take the single value and return as a number only
        self.sample_rate = np.sum(loaded_data[sample_rate])
        self.stimulus_pulses = loaded_data[stimulus_pulse].flatten()

        self.stimulation_locations = self.find_stim_locations(self.stimulus_pulses, remove_from_start=remove_from_start)
        self.raw_data = self.normalise_signals(self.raw_data)

    def find_stim_locations(self, stimulus_pulses: np.ndarray, boundry = 2, remove_from_start = 0):
        pulses = np.copy(stimulus_pulses) # make a copy so the original data is not edited

        pulses[pulses < boundry] = 0 # remove noise to have a perfect square wave
        pulses[pulses >= boundry] = 1

        diff = np.diff(pulses) # using the diff find the begining edge of the wave
        locations = np.where(diff >= 1)[0] # find locations of the wave
        locations = locations[remove_from_start:] # remove n number of begining pulses

        return locations

    def normalise_signals(self, signals):
        signals = np.copy(signals)
        
        signals = [scipy.signal.medfilt(signals[:, i], 7) for i in range(signals.shape[1])]
        signals = [np.convolve(signals[:, i], np.ones(5), 'valid') / 5 for i in range(signals.shape[1])]
        modes = max(set(round(signals, 3)), key=signals.count)
        signals = signals - max(set(round(signals, 3)), key=signals.count)

        return signals

    def get_signals(self, stim_level: int, levels: int, window_length = 1500):
        min_l = np.min(abs(np.diff(self.stimulation_locations)))

        if min_l > window_length:
            raise Exception("Window length too large. Impulses will overlap.")

        if len(self.stimulation_locations) % levels != 0:
            raise Exception("Incorrect number of stimulation levels.")

        # A for loop so it is easier to read
        signals = []
        for i in range(0, len(self.stimulation_locations), stim_level):
            start = self.stimulation_locations[i]
            end = start + window_length
            signals.append(self.raw_data[ start:end, :])

        return signals

stimulaiton_data = StimulationData("electricalStimulation.mat")

t1 = np.array([1, 2, 3, 4, 5])

t2 = t1.T

print(scipy.signal.medfilt(t1, 3))
print(scipy.signal.medfilt(t2, 3))