import cv2
import numpy as np
from scipy import ndimage
from skimage import measure
from skimage import segmentation
from skimage import exposure
import rtree
import time
import networkx as nx
from cellpose import models

from typing import List, Dict, Tuple
from scipy.spatial import cKDTree

from merlin.core import dataset
from merlin.core import analysistask
from merlin.util import spatialfeature
from merlin.util import watershed


class FeatureSavingAnalysisTask(analysistask.ParallelAnalysisTask):

    """
    An abstract analysis class that saves features into a spatial feature
    database.
    """

    def __init__(self, dataSet: dataset.DataSet, parameters=None,
                 analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def _reset_analysis(self, fragmentIndex: int = None) -> None:
        super()._reset_analysis(fragmentIndex)
        self.get_feature_database().empty_database(fragmentIndex)

    def get_feature_database(self) -> spatialfeature.SpatialFeatureDB:
        """ Get the spatial feature database this analysis task saves
        features into.

        Returns: The spatial feature database reference.
        """
        return spatialfeature.HDF5SpatialFeatureDB(self.dataSet, self)


class WatershedSegment(FeatureSavingAnalysisTask):

    """
    An analysis task that determines the boundaries of features in the
    image data in each field of view using a watershed algorithm.
    
    Since each field of view is analyzed individually, the segmentation results
    should be cleaned in order to merge cells that cross the field of
    view boundary.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'seed_channel_name' not in self.parameters:
            self.parameters['seed_channel_name'] = 'DAPI'
        if 'watershed_channel_name' not in self.parameters:
            self.parameters['watershed_channel_name'] = 'polyT'

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters['warp_task'],
                self.parameters['global_align_task']]

    def get_cell_boundaries(self) -> List[spatialfeature.SpatialFeature]:
        featureDB = self.get_feature_database()
        return featureDB.read_features()

    def _run_analysis(self, fragmentIndex):
        globalTask = self.dataSet.load_analysis_task(
                self.parameters['global_align_task'])

        seedIndex = self.dataSet.get_data_organization().get_data_channel_index(
            self.parameters['seed_channel_name'])
        seedImages = self._read_and_filter_image_stack(fragmentIndex,
                                                       seedIndex, 5)

        watershedIndex = self.dataSet.get_data_organization() \
            .get_data_channel_index(self.parameters['watershed_channel_name'])
        watershedImages = self._read_and_filter_image_stack(fragmentIndex,
                                                            watershedIndex, 5)
        seeds = watershed.separate_merged_seeds(
            watershed.extract_seeds(seedImages))
        normalizedWatershed, watershedMask = watershed.prepare_watershed_images(
            watershedImages)

        seeds[np.invert(watershedMask)] = 0
        watershedOutput = segmentation.watershed(
            normalizedWatershed, measure.label(seeds), mask=watershedMask,
            connectivity=np.ones((3, 3, 3)), watershed_line=True)

        zPos = np.array(self.dataSet.get_data_organization().get_z_positions())
        featureList = [spatialfeature.SpatialFeature.feature_from_label_matrix(
            (watershedOutput == i), fragmentIndex,
            globalTask.fov_to_global_transform(fragmentIndex), zPos)
            for i in np.unique(watershedOutput) if i != 0]

        featureDB = self.get_feature_database()
        featureDB.write_features(featureList, fragmentIndex)

    def _read_and_filter_image_stack(self, fov: int, channelIndex: int,
                                     filterSigma: float) -> np.ndarray:
        filterSize = int(2*np.ceil(2*filterSigma)+1)
        warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])
        return np.array([cv2.GaussianBlur(
            warpTask.get_aligned_image(fov, channelIndex, z),
            (filterSize, filterSize), filterSigma)
            for z in range(len(self.dataSet.get_z_positions()))])


class CellPoseSegment(FeatureSavingAnalysisTask):
    '''A task that determines the boundaries of features in the
    image data in each field of view using the Cellpose method
    '''
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        if 'nuclear_channel' not in self.parameters:
            self.parameters['nuclear_channel'] = 'DAPI'
        if 'membrane_channel' not in self.parameters:
            self.parameters['membrane_channel'] = 'conA'
        if 'cytoplasm_channel' not in self.parameters:
            self.parameters['cytoplasm_channel'] = 'polyT'
        if 'use_gpu' not in self.parameters:
            self.parameters['use_gpu'] = False
        if 'diameter' not in self.parameters:
            self.parameters['diameter'] = 50
        if 'min_size' not in self.parameters:
            self.parameters['min_size'] = 200
        if 'flow_threshold' not in self.parameters:
            self.parameters['flow_threshold'] = 0.4
        if 'stitch_threshold' not in self.parameters:
            self.parameters['stitch_threshold'] = 0.1
        if 'mask_threshold' not in self.parameters:
            self.parameters['mask_threshold'] = 1.4
        if 'verbose' not in self.parameters:
            self.parameters['verbose'] = True

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters['warp_task'],
                self.parameters['global_align_task']]

    def get_cell_boundaries(self) -> List[spatialfeature.SpatialFeature]:
        featureDB = self.get_feature_database()
        return featureDB.read_features()

    def scale_image(self, img, saturation_percentile=99.9):
        return np.minimum(img, np.percentile(img, saturation_percentile))

    def high_pass_filter_individual_z(self, image, sigma, truncate):
        lowpass = np.array([ndimage.gaussian_filter(image[z], sigma, mode='nearest', truncate=truncate)
                           for z in range(image.shape[0])])
        gauss_highpass = image - lowpass
        gauss_highpass[lowpass > image] = 0
        return gauss_highpass

    def adaptive_equalize_hist_individual_z(self, image, clip_limit=0.03):
        image_normalized = [image[z] / np.max(image[z]) for z in range(image.shape[0])]
        return np.array([exposure.equalize_adapthist(image_normalized[z], clip_limit=clip_limit) 
                        for z in range(image.shape[0])])

    def preprocess_image_channels(self, nuclear_image, membrane_marker_image):
        # Remove the hot-pixels
        nuclear_image = self.scale_image(nuclear_image, 99.9)
        membrane_marker_image = self.scale_image(membrane_marker_image, 99.9)

        # Run the high-pass filter for the membrance channel
        sigma = 5
        truncate = 2
        membrane_marker_image = self.high_pass_filter_individual_z(membrane_marker_image, sigma, truncate)

        # Enhance the contrast by adaptive histogram equalization
        nuclear_image = self.adaptive_equalize_hist_individual_z(nuclear_image, clip_limit=0.05)
        membrane_marker_image = self.adaptive_equalize_hist_individual_z(membrane_marker_image, clip_limit=0.05)

        return nuclear_image, membrane_marker_image
    @staticmethod
    def get_overlapping_objects(segmentationZ0: np.ndarray,
                                segmentationZ1: np.ndarray,
                                n0: int,
                                fraction_threshold0: float=0.4,
                                fraction_threshold1: float=0.4) -> Tuple[np.float64, 
                                                  np.float64, np.float64]:
        """compare cell labels in adjacent image masks
        Args:
            segmentationZ0: a 2 dimensional numpy array containing a
                segmentation mask in position Z
            segmentationZ1: a 2 dimensional numpy array containing a
                segmentation mask adjacent to segmentationZ0
            n0: an integer with the index of the object (cell/nuclei)
                to be compared between the provided segmentation masks
        Returns:
            a tuple (n1, f0, f1) containing the label of the cell in Z1
            overlapping n0 (n1), the fraction of n0 overlaping n1 (f0) and
            the fraction of n1 overlapping n0 (f1)
        """

        z1Indexes = np.unique(segmentationZ1[segmentationZ0 == n0])

        z1Indexes = z1Indexes[z1Indexes > 0]

        if z1Indexes.shape[0] > 0:

            # calculate overlap fraction
            n0Area = np.count_nonzero(segmentationZ0 == n0)
            n1Area = np.zeros(len(z1Indexes))
            overlapArea = np.zeros(len(z1Indexes))

            for ii in range(len(z1Indexes)):
                n1 = z1Indexes[ii]
                n1Area[ii] = np.count_nonzero(segmentationZ1 == n1)
                overlapArea[ii] = np.count_nonzero((segmentationZ0 == n0) *
                                                   (segmentationZ1 == n1))

            n0OverlapFraction = np.asarray(overlapArea / n0Area)
            n1OverlapFraction = np.asarray(overlapArea / n1Area)
            index = list(range(len(n0OverlapFraction)))

            # select the nuclei that has the highest fraction in n0 and n1
            r1, r2, indexSorted = zip(*sorted(zip(n0OverlapFraction,
                                                  n1OverlapFraction,
                                                  index),
                                      key=lambda x:x[0]+x[1],
                                      reverse=True))

            if (n0OverlapFraction[indexSorted[0]] > fraction_threshold0 and
                    n1OverlapFraction[indexSorted[0]] > fraction_threshold1):
                return (z1Indexes[indexSorted[0]],
                        n0OverlapFraction[indexSorted[0]],
                        n1OverlapFraction[indexSorted[0]])
            else:
                return (False, False, False)
        else:
            return (False, False, False)
    @staticmethod
    def combine_2d_segmentation_masks_into_3d(segmentationOutput: np.ndarray,
                                              minKept_zLen:int=2) -> np.ndarray:
        """Take a 3 dimensional segmentation masks and relabel them so that
        nuclei in adjacent sections have the same label if the area their
        overlap surpases certain threshold
        Args:
            segmentationOutput: a 3 dimensional numpy array containing the
                segmentation masks arranged as (z, x, y).
        Returns:
            ndarray containing a 3 dimensional mask arranged as (z, x, y) of
                relabeled segmented cells
        """

        # Initialize empty array with size as segmentationOutput array
        segmentationCombinedZ = np.zeros(segmentationOutput.shape)

        # copy the mask of the section farthest to the coverslip to start
        segmentationCombinedZ[-1, :, :] = segmentationOutput[-1, :, :]

        # starting far from coverslip
        for z in range(segmentationOutput.shape[0]-1, 0, -1):

            # get non-background cell indexes for plane Z
            zIndex = np.unique(segmentationCombinedZ[z, :, :])[
                                    np.unique(segmentationCombinedZ[z, :, :]) > 0]

            # get non-background cell indexes for plane Z-1
            zm1Index = np.unique(segmentationOutput[z-1, :, :])[
                                    np.unique(segmentationOutput[z-1, :, :]) > 0]
            assigned_zm1Index = []

            # compare each cell in z0
            for n0 in zIndex:
                n1, f0, f1 = CellPoseSegment.get_overlapping_objects(segmentationCombinedZ[z, :, :],
                                                     segmentationOutput[z-1, :, :],
                                                     n0)
                if n1:
                    segmentationCombinedZ[z-1, :, :][
                        (segmentationOutput[z-1, :, :] == n1)] = n0
                    assigned_zm1Index.append(n1)

            # keep the un-assigned indices in the Z-1 plane
            unassigned_zm1Index = [i for i in zm1Index if i not in assigned_zm1Index]
            max_current_id = np.max(segmentationCombinedZ[z-1:, :, :])
            for i in range(len(unassigned_zm1Index)):
                unassigned_id = unassigned_zm1Index[i]
                segmentationCombinedZ[z-1, :, :][
                        (segmentationOutput[z-1, :, :] == unassigned_id)] = max_current_id + 1 +i
        # remove label with only 1 z-layer
        segmentationCleanedZ = np.zeros(segmentationOutput.shape, dtype=np.int16)
        for _lb in np.arange(1, np.max(segmentationCombinedZ)+1):
            _cellMask = (segmentationCombinedZ==_lb)
            _cell_zIndex = np.where(_cellMask.any((1,2)))[0]
            #print(_cell_zIndex)
            if len(_cell_zIndex) <= minKept_zLen:
                continue
            else:
                segmentationCleanedZ[segmentationCombinedZ==_lb] = np.max(segmentationCleanedZ) + 1
                        
        return segmentationCleanedZ

    def _read_image_stack(self, fov: int, channelIndex: int) -> np.ndarray:
        warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])
        return np.array([warpTask.get_aligned_image(fov, channelIndex, z)
                         for z in range(len(self.dataSet.get_z_positions()))])

    def _save_mask(self, fov: int, 
        SegmentationMask3D: np.ndarray, ):
        return 

    def _run_analysis(self, fragmentIndex):
        globalTask = self.dataSet.load_analysis_task(
                self.parameters['global_align_task'])

        # read membrane and nuclear indices
        nuclear_ids = self.dataSet.get_data_organization().get_data_channel_index(
                self.parameters['nuclear_channel'])
        try:
            cytoplasm_ids = self.dataSet.get_data_organization().get_data_channel_index(
                    self.parameters['cytoplasm_channel'])
        except:
            cytoplasm_ids = None
        # read images and perform segmentation
        nuclear_images = self._read_image_stack(fragmentIndex, nuclear_ids)
        if cytoplasm_ids is None:
            run_cytoplasm = False
        else:
            run_cytoplasm = True
            cytoplasm_images = self._read_image_stack(fragmentIndex, cytoplasm_ids)
        # Load the cellpose model. 'nuclei' works the best so far for dapi only
        model = models.Cellpose(gpu=self.parameters['use_gpu'], model_type='nuclei')
        # Run the cellpose prediction
        masks, flows, styles, diams = model.eval(np.stack([nuclear_images, nuclear_images], axis=3), 
                                        diameter=self.parameters['diameter'], 
                                        do_3D=False, channels=[1,0], 
                                        resample=True, min_size=self.parameters['min_size'],
                                        flow_threshold=self.parameters['flow_threshold'],
                                        stitch_threshold=self.parameters['stitch_threshold'],
                                        verbose=self.parameters['verbose'],
                                        )
        # Combine 2D segmentation to 3D segmentation
        masks3d = self.combine_2d_segmentation_masks_into_3d(masks, minKept_zLen=np.round(masks.shape[0]/10))

        # Watershed with polyt if applicable
        if run_cytoplasm:
            # prepare watershed
            normalizedWatershed, watershedMask = watershed.prepare_watershed_images(cytoplasm_images, self.parameters['mask_threshold'])
            watershedMask[masks3d > 0] = True
            # run watershed                                                
            _final_mask = segmentation.watershed(
                normalizedWatershed, masks3d, mask=watershedMask,
                connectivity=np.ones((3, 3, 3)), watershed_line=True)
        else:
            _final_mask = masks3d

        # save mask3d
        self._save_mask(fragmentIndex, _final_mask)

        # Get the boundary features
        zPos = np.array(self.dataSet.get_data_organization().get_z_positions())
        featureList = [spatialfeature.SpatialFeature.feature_from_label_matrix(
            (_final_mask == i), fragmentIndex,
            globalTask.fov_to_global_transform(fragmentIndex), 
            zPos, 
            i) # add label
            for i in np.unique(_final_mask) if i != 0]

        featureDB = self.get_feature_database()
        featureDB.write_features(featureList, fragmentIndex, _final_mask)


    def _run_analysis_membrane(self, fragmentIndex):
        globalTask = self.dataSet.load_analysis_task(
                self.parameters['global_align_task'])

        # read membrane and nuclear indices
        nuclear_ids = self.dataSet.get_data_organization().get_data_channel_index(
                self.parameters['nuclear_channel'])
        membrane_ids = self.dataSet.get_data_organization().get_data_channel_index(
                self.parameters['membrane_channel'])

        # read images and perform segmentation
        nuclear_images = self._read_image_stack(fragmentIndex, nuclear_ids)
        membrane_images = self._read_image_stack(fragmentIndex, membrane_ids)

        # preprocess the images 
        nuclear_images_pp, membrane_images_pp = preprocess_image_channels(nuclear_images, membrane_images)

        # Combine the images into a stack
        zero_images = np.zeros(nuclear_images.shape)
        stacked_images = np.stack((zero_images, membrane_images_pp, nuclear_images_pp), axis=3)

        # Load the cellpose model. 'cyto2' performs better than 'cyto'.
        model = models.Cellpose(gpu=self.parameters['use_gpu'], model_type='cyto2')

        # Run the cellpose prediction
        chan = [2,3]
        masks, flows, styles, diams = model.eval(stacked_images, diameter=self.parameters['diameter'], 
                                        do_3D=False, channels=chan, 
                                        resample=True, min_size=self.parameters['min_size'])

        # Combine 2D segmentation to 3D segmentation
        masks3d = self.combine_2d_segmentation_masks_into_3d(masks)
        # save mask3d
        self._save_mask(fragmentIndex, masks3d)

        # Get the boundary features
        zPos = np.array(self.dataSet.get_data_organization().get_z_positions())
        featureList = [spatialfeature.SpatialFeature.feature_from_label_matrix(
            (masks3d == i), fragmentIndex,
            globalTask.fov_to_global_transform(fragmentIndex), zPos)
            for i in np.unique(masks3d) if i != 0]

        featureDB = self.get_feature_database()
        featureDB.write_features(featureList, fragmentIndex)




class CleanCellBoundaries(analysistask.ParallelAnalysisTask):
    '''
    A task to construct a network graph where each cell is a node, and overlaps
    are represented by edges. This graph is then refined to assign cells to the
    fov they are closest to (in terms of centroid). This graph is then refined
    to eliminate overlapping cells to leave a single cell occupying a given
    position.
    '''
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.segmentTask = self.dataSet.load_analysis_task(
            self.parameters['segment_task'])
        self.alignTask = self.dataSet.load_analysis_task(
            self.parameters['global_align_task'])

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['segment_task'],
                self.parameters['global_align_task']]

    def return_exported_data(self, fragmentIndex) -> nx.Graph:
        return self.dataSet.load_graph_from_gpickle(
            'cleaned_cells', self, fragmentIndex)

    def _run_analysis(self, fragmentIndex) -> None:
        allFOVs = np.array(self.dataSet.get_fovs())
        fovBoxes = self.alignTask.get_fov_boxes()
        fovIntersections = sorted([i for i, x in enumerate(fovBoxes) if
                                   fovBoxes[fragmentIndex].intersects(x)])
        intersectingFOVs = list(allFOVs[np.array(fovIntersections)])

        spatialTree = rtree.index.Index()
        count = 0
        idToNum = dict()
        for currentFOV in intersectingFOVs:
            cells = self.segmentTask.get_feature_database()\
                .read_features(currentFOV)
            cells = spatialfeature.simple_clean_cells(cells)

            spatialTree, count, idToNum = spatialfeature.construct_tree(
                cells, spatialTree, count, idToNum)

        graph = nx.Graph()
        cells = self.segmentTask.get_feature_database()\
            .read_features(fragmentIndex)
        cells = spatialfeature.simple_clean_cells(cells)
        graph = spatialfeature.construct_graph(graph, cells,
                                               spatialTree, fragmentIndex,
                                               allFOVs, fovBoxes)

        self.dataSet.save_graph_as_gpickle(
            graph, 'cleaned_cells', self, fragmentIndex)


class CombineCleanedBoundaries(analysistask.AnalysisTask):
    """
    A task to construct a network graph where each cell is a node, and overlaps
    are represented by edges. This graph is then refined to assign cells to the
    fov they are closest to (in terms of centroid). This graph is then refined
    to eliminate overlapping cells to leave a single cell occupying a given
    position.

    """
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.cleaningTask = self.dataSet.load_analysis_task(
            self.parameters['cleaning_task'])

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters['cleaning_task']]

    def return_exported_data(self):
        kwargs = {'index_col': 0}
        return self.dataSet.load_dataframe_from_csv(
            'all_cleaned_cells', analysisTask=self.analysisName, **kwargs)

    def _run_analysis(self):
        allFOVs = self.dataSet.get_fovs()
        graph = nx.Graph()
        for currentFOV in allFOVs:
            subGraph = self.cleaningTask.return_exported_data(currentFOV)
            graph = nx.compose(graph, subGraph)

        cleanedCells = spatialfeature.remove_overlapping_cells(graph)

        self.dataSet.save_dataframe_to_csv(cleanedCells, 'all_cleaned_cells',
                                           analysisTask=self)


class RefineCellDatabases(FeatureSavingAnalysisTask):
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.segmentTask = self.dataSet.load_analysis_task(
            self.parameters['segment_task'])
        self.cleaningTask = self.dataSet.load_analysis_task(
            self.parameters['combine_cleaning_task'])

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters['segment_task'],
                self.parameters['combine_cleaning_task']]

    def _run_analysis(self, fragmentIndex):

        cleanedCells = self.cleaningTask.return_exported_data()
        originalCells = self.segmentTask.get_feature_database()\
            .read_features(fragmentIndex)
        featureDB = self.get_feature_database()
        cleanedC = cleanedCells[cleanedCells['originalFOV'] == fragmentIndex]
        cleanedGroups = cleanedC.groupby('assignedFOV')
        for k, g in cleanedGroups:
            cellsToConsider = g['cell_id'].values.tolist()
            featureList = [x for x in originalCells if
                           str(x.get_feature_id()) in cellsToConsider]
            featureDB.write_features(featureList, fragmentIndex)


class ExportCellMetadata(analysistask.AnalysisTask):
    """
    An analysis task exports cell metadata.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.segmentTask = self.dataSet.load_analysis_task(
            self.parameters['segment_task'])

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['segment_task']]

    def _run_analysis(self):
        df = self.segmentTask.get_feature_database().read_feature_metadata()

        self.dataSet.save_dataframe_to_csv(df, 'feature_metadata',
                                           self.analysisName)
