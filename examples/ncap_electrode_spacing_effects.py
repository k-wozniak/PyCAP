from PyCAP.recordingProbes.simple_recording_probe import SimpleRecordingProbe
from PyCAP.compoundElectrodes.overlapping_multipolar_electrodes import OverlappingMultipolarElectrodes
from PyCAP.excitationSources.simple_excitation_source import SimpleExcitationSource
from PyCAP.model.model_params import ModelParams
from PyCAP.model.model import Model

from PyCAP.solvers import bipolar_electrodes as be
from PyCAP.solvers.utils.qs_generation import generate_q

import numpy as np
import matplotlib.pyplot as plt

import pickle
import os

def simulate():
    # Generate cv distribution with unity -----------------------------------------
    cv_min, cv_max, cv_step = (30, 60, 1)
    cv_dis = np.arange(cv_min, cv_max + cv_step, cv_step)
    cv_dis = np.c_[cv_dis, np.ones(cv_dis.shape[0])] # A uniform distribution

    # Set model params
    params = ModelParams(cv_dis, 0.02)
    params.fs = 100e3 # Hz

    model = Model(params)

    excitation_source = SimpleExcitationSource(params.time_series)
    model.add_excitation_source(excitation_source)

    # Add nine recording probes similar to the experimental data ------------------
    probes_start = 80e-3
    probes_center_to_center_distance = 3.5e-3
    number_of_probes = 10

    probes_distances = []
    probes = []
    for i in range(number_of_probes):
        # Find the distance
        current_distance = probes_start + (i*probes_center_to_center_distance)
        
        # Generate the probe
        recording_probe = SimpleRecordingProbe(current_distance)
        
        # Add probe to the model
        model.add_recording_probe(recording_probe)
        
        # Save info
        probes.append(recording_probe)
        probes_distances.append(current_distance)

    # Simulate --------------------------------------------------------------------
    model.simulate()

    # Generate Bipolar signals
    bipolar = OverlappingMultipolarElectrodes([-1, 1])
    bipolar.add_recording_probes(probes)
    bipolar_signals = bipolar.get_all_recordings()

    # Solve for different electrode placement errors ------------------------------
    search_range = np.arange(10, 80, 1)

    probes_start = 80e-3
    probes_center_to_center_distance = 3.5e-3
    number_of_probes = 10
    fs = 100e3 # Hz
    signal_length = len(bipolar_signals[0])

    patterns = [[1,1], [-1,-1], [1,-1], [-1, 1]]

    w_expected = np.zeros_like(search_range)
    w_expected[30:60] = 1
    w_expected = w_expected / np.sum(w_expected)

    outputs = []

    for percentage_offset in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]: #, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1]:
        for number_of_electrodes_effected in range(number_of_probes):
            for pattern in patterns:

                # Find distances
                distances = ((np.array(range(number_of_probes)) * probes_center_to_center_distance) 
                                + probes_start)

                # Shift distances
                for electrode in range(number_of_electrodes_effected):
                    sign = pattern[electrode % 2]
                    distances[electrode] = (distances[electrode] + (sign * percentage_offset * distances[electrode]))

                # Generate Qs
                qs = []
                for pos in distances:
                    qs.append(generate_q(signal_length, pos, search_range, fs))

                w = be.NCap(bipolar_signals, qs)

                diff = w_expected - w
                ssd = np.sum(diff**2)

                outputs.append({
                        'ssd': ssd,
                        'percentage_offset': percentage_offset,
                        'number_of_electrodes_effected': number_of_electrodes_effected,
                        'pattern': pattern,
                        'w': w,
                    })
    
    return outputs

# SCRIPT ----------------------------------------------------------------------

file_name = "effects_of_electrode_spacings"

# Check if file exist if not generate data and save
if os.path.exists(file_name):
    with open(file_name, "rb") as fp:
        outputs = pickle.load(fp)
else:
    outputs = simulate()
    with open(file_name, "wb") as fp:
        pickle.dump(outputs, fp)

patterns = [d['pattern'] for d in outputs]
patterns = np.unique(np.array(patterns), axis=0)
patterns = patterns.tolist()

# Display data
for pattern in patterns:
    output_pattern = [d for d in outputs if d['pattern'] == pattern]

    ssd_array = np.array([d['ssd'] for d in output_pattern])
    percentage_offset = np.array([d['percentage_offset'] for d in output_pattern])
    number_of_electrodes_effected = np.array([d['number_of_electrodes_effected'] for d in output_pattern])
    
    fig = plt.figure()
    
    ax = plt.axes(projection='3d')

    #ax.plot(percentage_offset, number_of_electrodes_effected, ssd_array, linewidth=1, antialiased=False)
    ax.plot_trisurf(percentage_offset, number_of_electrodes_effected, ssd_array, cmap=plt.cm.jet, linewidth=0.1)

    ax.set_title(str(pattern))
    ax.set_xlabel("percentage_offset")
    ax.set_ylabel("number_of_electrodes_effected")
    ax.set_zlabel("ssd_array")

    plt.show()

# display ssd vs percentage offset vs number of electrodes effected
# display separate graph for each pattern
