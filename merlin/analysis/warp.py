from typing import List
from typing import Union
import numpy as np
from skimage import transform
from skimage import feature
from skimage import registration
import cv2

from merlin.core import analysistask
from merlin.util import aberration


class Warp(analysistask.ParallelAnalysisTask):

    """
    An abstract class for warping a set of images so that the corresponding
    pixels align between images taken in different imaging rounds.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'write_fiducial_images' not in self.parameters:
            self.parameters['write_fiducial_images'] = False
        if 'write_aligned_images' not in self.parameters:
            self.parameters['write_aligned_images'] = False

        self.writeAlignedFiducialImages = self.parameters[
                'write_fiducial_images']

    def get_aligned_image_set(
            self, fov: int,
            chromaticCorrector: aberration.ChromaticCorrector=None
    ) -> np.ndarray:
        """Get the set of transformed images for the specified fov.

        Args:
            fov: index of the field of view
            chromaticCorrector: the ChromaticCorrector to use to chromatically
                correct the images. If not supplied, no correction is
                performed.
        Returns:
            a 4-dimensional numpy array containing the aligned images. The
                images are arranged as [channel, zIndex, x, y]
        """
        dataChannels = self.dataSet.get_data_organization().get_data_channels()
        zIndexes = range(len(self.dataSet.get_z_positions()))
        return np.array([[self.get_aligned_image(fov, d, z, chromaticCorrector)
                          for z in zIndexes] for d in dataChannels])

    def get_aligned_image(
            self, fov: int, dataChannel: int, zIndex: int,
            chromaticCorrector: aberration.ChromaticCorrector=None
    ) -> np.ndarray:
        """Get the specified transformed image

        Args:
            fov: index of the field of view
            dataChannel: index of the data channel
            zIndex: index of the z position
            chromaticCorrector: the ChromaticCorrector to use to chromatically
                correct the images. If not supplied, no correction is
                performed.
        Returns:
            a 2-dimensional numpy array containing the specified image
        """
        inputImage = self.dataSet.get_raw_image(
            dataChannel, fov, self.dataSet.z_index_to_position(zIndex))
        transformation = self.get_transformation(fov, dataChannel)
        if chromaticCorrector is not None:
            imageColor = self.dataSet.get_data_organization()\
                            .get_data_channel_color(dataChannel)
            return transform.warp(chromaticCorrector.transform_image(
                inputImage, imageColor), transformation, preserve_range=True
                ).astype(inputImage.dtype)
        else:
            return transform.warp(inputImage, transformation,
                                  preserve_range=True).astype(inputImage.dtype)

    def _process_transformations(self, transformationList, fov) -> None:
        """
        Process the transformations determined for a given fov. 

        The list of transformation is used to write registered images and 
        the transformation list is archived.

        Args:
            transformationList: A list of transformations that contains a
                transformation for each data channel. 
            fov: The fov that is being transformed.
        """

        dataChannels = self.dataSet.get_data_organization().get_data_channels()

        if self.parameters['write_aligned_images']:
            zPositions = self.dataSet.get_z_positions()

            imageDescription = self.dataSet.analysis_tiff_description(
                    len(zPositions), len(dataChannels))

            with self.dataSet.writer_for_analysis_images(
                    self, 'aligned_images', fov) as outputTif:
                for t, x in zip(transformationList, dataChannels):
                    for z in zPositions:
                        inputImage = self.dataSet.get_raw_image(x, fov, z)
                        transformedImage = transform.warp(
                                inputImage, t, preserve_range=True) \
                            .astype(inputImage.dtype)
                        outputTif.save(
                                transformedImage,
                                photometric='MINISBLACK',
                                contiguous=True,
                                metadata=imageDescription)

        if self.writeAlignedFiducialImages:

            fiducialImageDescription = self.dataSet.analysis_tiff_description(
                    1, len(dataChannels))

            with self.dataSet.writer_for_analysis_images(
                    self, 'aligned_fiducial_images', fov) as outputTif:
                for t, x in zip(transformationList, dataChannels):
                    inputImage = self.dataSet.get_fiducial_image(x, fov)
                    transformedImage = transform.warp(
                            inputImage, t, preserve_range=True) \
                        .astype(inputImage.dtype)
                    # append layer into file
                    outputTif.save(
                            transformedImage, 
                            photometric='MINISBLACK',
                            contiguous=True,
                            metadata=fiducialImageDescription)

        self._save_transformations(transformationList, fov)

    def _save_transformations(self, transformationList: List, fov: int) -> None:
        self.dataSet.save_numpy_analysis_result(
            np.array([np.array(_m) for _m in transformationList], dtype=np.float64), 
            'offsets',
            self.get_analysis_name(), resultIndex=fov,
            subdirectory='transformations') # here only save the array information

    def get_transformation(self, fov: int, dataChannel: int=None
                            ) -> Union[transform.EuclideanTransform,
                                 List[transform.EuclideanTransform]]:
        """Get the transformations for aligning images for the specified field
        of view.

        Args:
            fov: the fov to get the transformations for.
            dataChannel: the index of the data channel to get the transformation
                for. If None, then all data channels are returned.
        Returns:
            a EuclideanTransform if dataChannel is specified or a list of
                EuclideanTransforms for all dataChannels if dataChannel is
                not specified.
        """
        transformationList = self.dataSet.load_numpy_analysis_result(
            'offsets', self, resultIndex=fov, subdirectory='transformations')
        # convert to Transform format
        transformationList = [transform.EuclideanTransform(_m) for _m in transformationList]
        if dataChannel is not None:
            return transformationList[dataChannel]
        else:
            return transformationList


class FiducialCorrelationWarp(Warp):

    """
    An analysis task that warps a set of images taken in different imaging
    rounds based on the crosscorrelation between fiducial images.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'highpass_sigma' not in self.parameters:
            self.parameters['highpass_sigma'] = 3

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return []

    def _filter(self, inputImage: np.ndarray) -> np.ndarray:
        highPassSigma = self.parameters['highpass_sigma']
        highPassFilterSize = int(2 * np.ceil(2 * highPassSigma) + 1)

        highPassImage =  inputImage.astype(float) - cv2.GaussianBlur(
            inputImage, (highPassFilterSize, highPassFilterSize),
            highPassSigma, borderType=cv2.BORDER_REPLICATE)
        highPassImage[highPassImage < 0] = 0
        return highPassImage

    def _run_analysis(self, fragmentIndex: int):
        # TODO - this can be more efficient since some images should
        # use the same alignment if they are from the same imaging round       
        ref_bit = 0 
        fixedRawImage = self.dataSet.get_fiducial_image(ref_bit, fragmentIndex)
        fixedImage = self._filter(fixedRawImage) # get the first round as ref
        # calculate offsets
        offsets = []
        for bit in self.dataSet.get_data_organization().get_data_channels():
            movingRawImage = self.dataSet.get_fiducial_image(bit, fragmentIndex)
            movingImage = self._filter(movingRawImage)
            _offset = registration.phase_cross_correlation(
                fixedImage,movingImage,upsample_factor=100,normalization=None)[0]
            # if all zero, calculate again
            #if not _offset.any() and bit != ref_bit:
            #    _offset = registration.phase_cross_correlation(
            #        fixedRawImage,movingRawImage,upsample_factor=100,normalization=None)[0]
            # append
            offsets.append(_offset)
        print(offsets)
        transformations = [transform.SimilarityTransform(translation=[-_offset[1], -_offset[0]]) 
                           for _offset in offsets]
        self._process_transformations(transformations, fragmentIndex)
