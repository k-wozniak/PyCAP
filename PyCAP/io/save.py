"""
from scipy.io import savemat
import hashlib
import bencode

def save_model_data(model, file_name = None):
    
    if file_name is None:
        file_name = model_hash(model_params, recording_probes)


    # Difficult to extend when the model will extend -.-
    
    file_data = {
        "cv_diss": cv_dis,
        "fs": params.fs,

        "stimulation sources"
        "noise sources"
        "probes" # __class__.__name__

        "probes distances"
        "probes_start": probes_start,
        "probes_center_to_center_distance": probes_center_to_center_distance,
        "number_of_probes": number_of_probes,
    }

    savemat(file_name, file_data)

def model_hash(model_params, recording_probes):
    data = [
        model_params.cv_distribution,
        model_params.fs,
        model_params.simulation_length,
        
    ]
    

    hash_obj = hashlib.md5(repr(data).encode('utf-8'))

    return hash_obj.hexdigest()
"""