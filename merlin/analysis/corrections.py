# image_correction 
# by Pu Zheng 
# 2024.08.21
import numpy as np
import cv2
from merlin.core import analysistask, dataset
from merlin.util import illumination

class GenerateCorrection(analysistask.AnalysisTask):
    """
    An abstract class for illumination correction

    Args:
        analysistask (_type_): _description_
    """
    
    def __init__(self, 
                 dataSet:dataset.MERFISHDataSet, 
                 parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if "write_correction_profile" not in self.parameters:
            self.parameters["write_correction_profile"] = True
        
        self.writeCorrectionProfile = self.parameters["write_correction_profile"]
    
    def get_individual_correction_profile(
        self, fov: int, dataChannel: int, zIndex: int,
        ):
        """
        Args:
            fov: index of the field of view
            dataChannel: index of the data channel
            zIndex: index of the z position
            chromaticCorrector: the ChromaticCorrector to use to chromatically
                correct the images. If not supplied, no correction is
                performed.
        Returns:
            a 2-dimensional numpy array as an illumination profile, channel of this image
        """
        inputImage = self.dataSet.get_raw_image(
            dataChannel, fov, self.dataSet.z_index_to_position(zIndex))
        correctionProfile = illumination.image_2_illumination_profile(inputImage, channel=str(dataChannel))
        # Save
        if self.writeCorrectionProfile:
            self.save_correction_profile(correctionProfile, fov, dataChannel)
        return correctionProfile, dataChannel
        

    def save_correction_profile(self, correction_profile:np.ndarray, fov: int, dataChannel: int):
        """
        Save the correction profile to the dataset

        Args:
            correction_pfs: a dictionary of correction profiles
        """
        self.dataSet.save_numpy_analysis_result(
            correction_profile, 
            f'offsets_{dataChannel}',
            self.get_analysis_name(), resultIndex=fov,
            subdirectory='raw_profiles') # here only save the array information
    