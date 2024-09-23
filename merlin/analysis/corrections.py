# image_correction 
# by Pu Zheng 
# 2024.08.21
import numpy as np
import cv2
from merlin.core import analysistask, dataset
from merlin.util import illumination

class GenerateIndivualCorrection(analysistask.AnalysisTask):
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
    
    def _process_individual_correction_profile(
        self, fov: int, dataChannel: int, zIndex: int,
        correction_type:str='illumination',
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
        if correction_type == 'illumination':
            correctionProfile = illumination.image_2_illumination_profile(inputImage, channel=str(dataChannel))
        else:
            raise NotImplementedError(f"correction type {correction_type} not implemented in this class: GenerateIndivualCorrection")
        # Save
        if self.writeCorrectionProfile:
            self.save_correction_profile(correctionProfile, fov, dataChannel)
        return correctionProfile, dataChannel

    def _save_individual_correction_profile(self, correction_profile:np.ndarray, 
                                           fov:int, dataChannel: int, 
                                           correction_type:str='illumination',
        ):
        """
        Save the correction profile to the dataset

        Args:
            correction_profile: the correction profile to save
            fov: the field of view index
            dataChannel: the data channel index
            correction_type: the type of correction
        """
        self.dataSet.save_numpy_analysis_result(
            correction_profile, 
            f'{correction_type}_{dataChannel}',
            self.get_analysis_name(), resultIndex=fov,
            subdirectory='raw_profiles') # here only save the array information
    
    def _get_individual_corretion_profile(self, fov:int, dataChannel:int, correction_type:str='illumination'):
        """
        Get the correction profile for a specific field of view and data channel

        Args:
            fov: the field of view index
            dataChannel: the data channel index
            correction_type: the type of correction
        Returns:
            the correction profile
        """
        return self.dataSet.load_numpy_analysis_result(
            f'{correction_type}_{dataChannel}',
            self.get_analysis_name(), resultIndex=fov,
            subdirectory='raw_profiles')
        
class GenerateIndividualIlluminationCorrection(GenerateIndivualCorrection):
    """
    A class for generating illumination correction profiles

    Args:
        GenerateCorrection (_type_): _description_
    """
    
    def __init__(self, 
                 dataSet:dataset.MERFISHDataSet, 
                 parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        self.correction_type = 'illumination'

    def process_individual_correction_profile(self, fov: int, dataChannel: int, zIndex: int):
        return super()._process_individual_correction_profile(fov, dataChannel, zIndex, correction_type=self.correction_type)
    
    def load_correction_profile(self, fov:int, dataChannel:int):
        return super()._get_individual_corretion_profile(fov, dataChannel, correction_type=self.correction_type)
    
    def save_correction_profile(self, correction_profile:np.ndarray, fov:int, dataChannel:int):
        return super()._save_individual_correction_profile(correction_profile, fov, dataChannel, correction_type=self.correction_type)
    
    def _run_analysis(self):
        """
        Generate the illumination correction profiles
        """
        for fov in self.dataSet.get_fovs():
            for dataChannel in self.dataSet.get_data_channels():
                for zIndex in self.dataSet.zIndexes:
                    self.process_individual_correction_profile(fov, dataChannel, zIndex)
        
    
class GenerateMergedCorrection(analysistask.AnalysisTask):
    """
    A class for generating illumination correction profiles

    Args:
        GenerateCorrection (_type_): _description_
    """
    
    def __init__(self, 
                 dataSet:dataset.MERFISHDataSet, 
                 parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)


    def process(self):
        """
        Generate the illumination correction profiles
        """
        for fov in self.dataSet.get_fovs():
            for dataChannel in self.dataSet.get_data_channels():
                for zIndex in self.dataSet.zIndexes:
                    self.process_individual_correction_profile(fov, dataChannel, zIndex)
    