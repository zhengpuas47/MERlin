import os
import numpy as np
import pandas
from scipy import optimize
from concurrent.futures import ProcessPoolExecutor

from merlin.core import analysistask
from merlin.analysis import decode


def _compute_fov_histogram(args):
    """Read one FOV's barcodes from HDF5 and return partial (blank, coding) histograms.

    Must be a module-level function so ProcessPoolExecutor can pickle it.
    """
    h5_path, blank_set, coding_set, intensityBins, distanceBins, areaBins = args
    hist_shape = (len(intensityBins) - 1, len(distanceBins) - 1, len(areaBins) - 1)
    zero = np.zeros(hist_shape)

    try:
        with pandas.HDFStore(h5_path, mode='r') as store:
            if 'barcodes' not in store:
                return zero, zero
            barcodes = pandas.read_hdf(
                store, key='barcodes',
                columns=['barcode_id', 'mean_intensity', 'min_distance', 'area'])
    except OSError:
        return zero, zero

    if len(barcodes) == 0:
        return zero, zero

    barcodeData = barcodes[['mean_intensity', 'min_distance', 'area']].values.astype(float)
    barcodeData[:, 0] = np.log10(np.maximum(barcodeData[:, 0], 1e-10))
    barcode_ids = barcodes['barcode_id'].values

    blank_mask = np.isin(barcode_ids, list(blank_set))
    coding_mask = np.isin(barcode_ids, list(coding_set))

    blank_counts = np.histogramdd(
        barcodeData[blank_mask], bins=(intensityBins, distanceBins, areaBins))[0]
    coding_counts = np.histogramdd(
        barcodeData[coding_mask], bins=(intensityBins, distanceBins, areaBins))[0]

    return blank_counts, coding_counts


class AbstractFilterBarcodes(decode.BarcodeSavingParallelAnalysisTask):
    """
    An abstract class for filtering barcodes identified by pixel-based decoding.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_codebook(self):
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        return decodeTask.get_codebook()


class FilterBarcodes(AbstractFilterBarcodes):

    """
    An analysis task that filters barcodes based on area and mean
    intensity.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'area_threshold' not in self.parameters:
            self.parameters['area_threshold'] = 3
        if 'intensity_threshold' not in self.parameters:
            self.parameters['intensity_threshold'] = 200
        if 'distance_threshold' not in self.parameters:
            self.parameters['distance_threshold'] = 1e6

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 1000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['decode_task']]

    def _run_analysis(self, fragmentIndex):
        decodeTask = self.dataSet.load_analysis_task(
                self.parameters['decode_task'])
        areaThreshold = self.parameters['area_threshold']
        intensityThreshold = self.parameters['intensity_threshold']
        distanceThreshold = self.parameters['distance_threshold']
        barcodeDB = self.get_barcode_database()
        barcodeDB.write_barcodes(
            decodeTask.get_barcode_database().get_filtered_barcodes(
                areaThreshold, intensityThreshold,
                distanceThreshold=distanceThreshold, fov=fragmentIndex),
            fov=fragmentIndex)


class GenerateAdaptiveThreshold(analysistask.AnalysisTask):

    """
    An analysis task that generates a three-dimension mean intenisty,
    area, minimum distance histogram for barcodes as they are decoded.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'tolerance' not in self.parameters:
            self.parameters['tolerance'] = 0.001
        # ensure decode_task is specified
        decodeTask = self.parameters['decode_task']

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 5000

    def get_estimated_time(self):
        return 1800

    def _get_n_processors(self):
        return self.parameters.get('n_processors', 1)

    def get_dependencies(self):
        return [self.parameters['run_after_task']]

    def get_blank_count_histogram(self) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result('blank_counts', self)

    def get_coding_count_histogram(self) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result('coding_counts', self)

    def get_total_count_histogram(self) -> np.ndarray:
        return self.get_blank_count_histogram() \
               + self.get_coding_count_histogram()

    def get_area_bins(self) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result('area_bins', self)

    def get_distance_bins(self) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result(
            'distance_bins', self)

    def get_intensity_bins(self) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result(
            'intensity_bins', self, None)

    def get_blank_fraction_histogram(self) -> np.ndarray:
        """ Get the normalized blank fraction histogram indicating the
        normalized blank fraction for each intensity, distance, and area
        bin.

        Returns: The normalized blank fraction histogram. The histogram
            has three dimensions: mean intensity, minimum distance, and area.
            The bins in each dimension are defined by the bins returned by
            get_area_bins, get_distance_bins, and get_area_bins, respectively.
            Each entry indicates the number of blank barcodes divided by the
            number of coding barcodes within the corresponding bin
            normalized by the fraction of blank barcodes in the codebook.
            With this normalization, when all (both blank and coding) barcodes
            are selected with equal probability, the blank fraction is
            expected to be 1.
        """
        blankHistogram = self.get_blank_count_histogram()
        totalHistogram = self.get_coding_count_histogram()
        blankFraction = blankHistogram / totalHistogram
        blankFraction[totalHistogram == 0] = np.finfo(blankFraction.dtype).max
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        codebook = decodeTask.get_codebook()
        blankBarcodeCount = len(codebook.get_blank_indexes())
        codingBarcodeCount = len(codebook.get_coding_indexes())
        blankFraction /= blankBarcodeCount/(
                blankBarcodeCount + codingBarcodeCount)
        return blankFraction

    def calculate_misidentification_rate_for_threshold(
            self, threshold: float) -> float:
        """ Calculate the misidentification rate for a specified blank
        fraction threshold.

        Args:
            threshold: the normalized blank fraction threshold
        Returns: The estimated misidentification rate, estimated as the
            number of blank barcodes per blank barcode divided
            by the number of coding barcodes per coding barcode.
        """
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        codebook = decodeTask.get_codebook()
        blankBarcodeCount = len(codebook.get_blank_indexes())
        codingBarcodeCount = len(codebook.get_coding_indexes())
        blankHistogram = self.get_blank_count_histogram()
        codingHistogram = self.get_coding_count_histogram()
        blankFraction = self.get_blank_fraction_histogram()

        selectBins = blankFraction < threshold
        codingCounts = np.sum(codingHistogram[selectBins])
        blankCounts = np.sum(blankHistogram[selectBins])

        return ((blankCounts/blankBarcodeCount) /
                (codingCounts/codingBarcodeCount))

    def calculate_threshold_for_misidentification_rate(
            self, targetMisidentificationRate: float) -> float:
        """ Calculate the normalized blank fraction threshold that achieves
        a specified misidentification rate.

        Args:
            targetMisidentificationRate: the target misidentification rate
        Returns: the normalized blank fraction threshold that achieves
            targetMisidentificationRate
        """
        tolerance = self.parameters['tolerance']
        def misidentification_rate_error_for_threshold(x, targetError):
            return self.calculate_misidentification_rate_for_threshold(x) \
                - targetError
        return optimize.newton(
            misidentification_rate_error_for_threshold, 0.2,
            args=[targetMisidentificationRate], tol=tolerance, x1=0.3,
            disp=False)

    def calculate_barcode_count_for_threshold(self, threshold: float) -> float:
        """ Calculate the number of barcodes remaining after applying
        the specified normalized blank fraction threshold.

        Args:
            threshold: the normalized blank fraction threshold
        Returns: The number of barcodes passing the threshold.
        """
        blankHistogram = self.get_blank_count_histogram()
        codingHistogram = self.get_coding_count_histogram()
        blankFraction = self.get_blank_fraction_histogram()
        return np.sum(blankHistogram[blankFraction < threshold]) \
            + np.sum(codingHistogram[blankFraction < threshold])

    def extract_barcodes_with_threshold(self, blankThreshold: float,
                                        barcodeSet: pandas.DataFrame
                                        ) -> pandas.DataFrame:
        selectData = barcodeSet[
            ['mean_intensity', 'min_distance', 'area']].values
        selectData[:, 0] = np.log10(selectData[:, 0])
        blankFractionHistogram = self.get_blank_fraction_histogram()

        barcodeBins = np.array(
            (np.digitize(selectData[:, 0], self.get_intensity_bins(),
                         right=True),
             np.digitize(selectData[:, 1], self.get_distance_bins(),
                         right=True),
             np.digitize(selectData[:, 2], self.get_area_bins()))) - 1
        barcodeBins[0, :] = np.clip(
            barcodeBins[0, :], 0, blankFractionHistogram.shape[0]-1)
        barcodeBins[1, :] = np.clip(
            barcodeBins[1, :], 0, blankFractionHistogram.shape[1]-1)
        barcodeBins[2, :] = np.clip(
            barcodeBins[2, :], 0, blankFractionHistogram.shape[2]-1)
        raveledIndexes = np.ravel_multi_index(
            barcodeBins[:, :], blankFractionHistogram.shape)

        thresholdedBlankFraction = blankFractionHistogram < blankThreshold
        return barcodeSet[np.take(thresholdedBlankFraction, raveledIndexes)]

    @staticmethod
    def _extract_counts(barcodes, intensityBins, distanceBins, areaBins):
        barcodeData = barcodes[
            ['mean_intensity', 'min_distance', 'area']].values
        barcodeData[:, 0] = np.log10(barcodeData[:, 0])
        return np.histogramdd(
            barcodeData, bins=(intensityBins, distanceBins, areaBins))[0]

    def _run_analysis(self):
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        codebook = decodeTask.get_codebook()
        barcodeDB = decodeTask.get_barcode_database()
        n_processors = self._get_n_processors()

        completeFragments = \
            self.dataSet.load_numpy_analysis_result_if_available(
                'complete_fragments', self, [False]*self.fragment_count())

        areaBins = self.dataSet.load_numpy_analysis_result_if_available(
            'area_bins', self, np.arange(1, 35))
        distanceBins = self.dataSet.load_numpy_analysis_result_if_available(
            'distance_bins', self,
            np.arange(
                0, decodeTask.parameters['distance_threshold']+0.02, 0.01))
        intensityBins = self.dataSet.load_numpy_analysis_result_if_available(
            'intensity_bins', self, None)

        blankCounts = self.dataSet.load_numpy_analysis_result_if_available(
            'blank_counts', self, None)
        codingCounts = self.dataSet.load_numpy_analysis_result_if_available(
            'coding_counts', self, None)

        self.dataSet.save_numpy_analysis_result(areaBins, 'area_bins', self)
        self.dataSet.save_numpy_analysis_result(distanceBins, 'distance_bins', self)

        # --- Phase 1: initialize intensity bins from a sample of FOVs ---
        if intensityBins is None or blankCounts is None or codingCounts is None:
            allFovs = [i for i in range(self.fragment_count())
                       if decodeTask.is_complete(i)]
            sampleSize = min(20, len(allFovs))
            sampledFragments = np.random.choice(allFovs, size=sampleSize, replace=False)

            def extreme_values(inputData: pandas.Series):
                return inputData.min(), inputData.max()

            intensityExtremes = [
                extreme_values(barcodeDB.get_barcodes(
                    i, columnList=['mean_intensity'])['mean_intensity'])
                for i in sampledFragments]
            maxIntensity = np.log10(np.max([x[1] for x in intensityExtremes]))
            intensityBins = np.arange(0, 2 * maxIntensity, maxIntensity / 100)
            self.dataSet.save_numpy_analysis_result(intensityBins, 'intensity_bins', self)

            blankCounts = np.zeros((len(intensityBins) - 1,
                                    len(distanceBins) - 1,
                                    len(areaBins) - 1))
            codingCounts = np.zeros((len(intensityBins) - 1,
                                     len(distanceBins) - 1,
                                     len(areaBins) - 1))

        # --- Phase 2: accumulate histograms across all remaining FOVs ---
        pendingFovs = [i for i in range(self.fragment_count())
                       if not completeFragments[i] and decodeTask.is_complete(i)]

        if pendingFovs:
            # Build file paths for each FOV's barcode HDF5
            barcodeDir = self.dataSet.get_analysis_subdirectory(
                barcodeDB._analysisTask, 'barcodes')
            h5_paths = [os.path.join(barcodeDir, f'barcode_data_{fov}.h5')
                        for fov in pendingFovs]

            blank_set = set(codebook.get_blank_indexes())
            coding_set = set(codebook.get_coding_indexes())
            args_list = [
                (path, blank_set, coding_set, intensityBins, distanceBins, areaBins)
                for path in h5_paths]

            if n_processors > 1:
                with ProcessPoolExecutor(max_workers=n_processors) as executor:
                    for fov, (bc, cc) in zip(
                            pendingFovs, executor.map(_compute_fov_histogram, args_list)):
                        blankCounts += bc
                        codingCounts += cc
                        completeFragments[fov] = True
            else:
                for fov, args in zip(pendingFovs, args_list):
                    bc, cc = _compute_fov_histogram(args)
                    blankCounts += bc
                    codingCounts += cc
                    completeFragments[fov] = True

            self.dataSet.save_numpy_analysis_result(
                completeFragments, 'complete_fragments', self)
            self.dataSet.save_numpy_analysis_result(blankCounts, 'blank_counts', self)
            self.dataSet.save_numpy_analysis_result(codingCounts, 'coding_counts', self)


class AdaptiveFilterBarcodes(AbstractFilterBarcodes):

    """
    An analysis task that filters barcodes based on a mean intensity threshold
    for each area based on the abundance of blank barcodes. The threshold
    is selected to achieve a specified misidentification rate.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'misidentification_rate' not in self.parameters:
            self.parameters['misidentification_rate'] = 0.05

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 1000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['adaptive_task'],
                self.parameters['decode_task']]

    def get_adaptive_thresholds(self):
        """ Get the adaptive thresholds used for filtering barcodes.

        Returns: The GenerateaAdaptiveThershold task using for this
            adaptive filter.
        """
        return self.dataSet.load_analysis_task(
            self.parameters['adaptive_task'])

    def _run_analysis(self, fragmentIndex):
        adaptiveTask = self.dataSet.load_analysis_task(
            self.parameters['adaptive_task'])
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])

        threshold = adaptiveTask.calculate_threshold_for_misidentification_rate(
            self.parameters['misidentification_rate'])

        bcDatabase = self.get_barcode_database()
        currentBarcodes = decodeTask.get_barcode_database()\
            .get_barcodes(fragmentIndex)

        bcDatabase.write_barcodes(adaptiveTask.extract_barcodes_with_threshold(
            threshold, currentBarcodes), fov=fragmentIndex)
