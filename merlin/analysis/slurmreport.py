import subprocess
import pandas
import io
import requests
import time
from matplotlib import pyplot as plt
import numpy as np

from merlin.core import analysistask


class SlurmReport(analysistask.AnalysisTask):

    """
    An analysis task that generates reports on previously completed analysis
    tasks using Slurm.

    This analysis task only works when Merlin is run through Slurm
    with every analysis task fragment run as a separate job.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return [self.parameters['run_after_task']]

    def _generate_slurm_report(self, task: analysistask.AnalysisTask):
        if isinstance(task, analysistask.ParallelAnalysisTask):
            idList = [
                self.dataSet.get_analysis_environment(task, i)['SLURM_JOB_ID']
                for i in range(task.fragment_count())]
        else:
            idList = [
                self.dataSet.get_analysis_environment(task)['SLURM_JOB_ID']]

        queryResult = subprocess.run(
            ['sacct', '--format=AssocID,Account,Cluster,User,JobID,JobName,'
             + 'NodeList,AveCPU,AveCPUFreq,MaxPages,MaxDiskRead,MaxDiskWrite,'
             + 'MaxRSS,ReqMem,CPUTime,Elapsed,Start,End,Timelimit',
             '--units=M', '-P', '-j', ','.join(idList)], stdout=subprocess.PIPE)

        slurmJobDF = pandas.read_csv(
            io.StringIO(queryResult.stdout.decode('utf-8')), sep='|')

        return self._clean_slurm_dataframe(slurmJobDF)

    @staticmethod
    def _clean_slurm_dataframe(inputDataFrame):
        outputDF = inputDataFrame[
            ~inputDataFrame['JobID'].str.contains('.extern')]
        outputDF = outputDF.assign(
            JobID=outputDF['JobID'].str.partition('.')[0])

        def get_not_nan(listIn):
            return listIn.dropna().iloc[0]

        outputDF = outputDF.groupby('JobID').aggregate(get_not_nan)

        def reformat_timedelta(elapsedIn):
            splitElapsed = elapsedIn.split('-')
            if len(splitElapsed) > 1:
                return splitElapsed[0] + ' days ' + splitElapsed[1]
            else:
                return splitElapsed[0]

        outputDF = outputDF.assign(Elapsed=pandas.to_timedelta(
            outputDF['Elapsed'].apply(reformat_timedelta), unit='s'))
        outputDF = outputDF.assign(Timelimit=pandas.to_timedelta(
            outputDF['Timelimit'].apply(reformat_timedelta), unit='s'))

        return outputDF.reindex()

    def _plot_slurm_report(self, slurmDF, analysisName):
        fig = plt.figure(figsize=(15, 4))

        plt.subplot(1, 4, 1)
        plt.boxplot([slurmDF['MaxRSS'].str[:-1].astype(float),
                     slurmDF['ReqMem'].str[:-2].astype(int)], widths=0.5)
        plt.xticks([1, 2], ['Max used', 'Requested'])
        plt.ylabel('Memory (mb)')
        plt.title('RAM')
        plt.subplot(1, 4, 2)
        plt.boxplot([slurmDF['Elapsed'] / np.timedelta64(1, 'm'),
                     slurmDF['Timelimit'] / np.timedelta64(1, 'm')],
                    widths=0.5)
        plt.xticks([1, 2], ['Elapsed', 'Requested'])
        plt.ylabel('Time (min)')
        plt.title('Run time')
        plt.subplot(1, 4, 3)
        plt.boxplot([slurmDF['MaxDiskRead'].str[:-1].astype(float)],
                    widths=0.25)
        plt.xticks([1], ['MaxDiskRead'])
        plt.ylabel('Number of mb read')
        plt.title('Disk usage')
        plt.subplot(1, 4, 4)
        plt.boxplot([slurmDF['MaxDiskWrite'].str[:-1].astype(float)],
                    widths=0.25)
        plt.xticks([1], ['MaxDiskWrite'])
        plt.ylabel('Number of mb written')
        plt.tight_layout(pad=1)
        self.dataSet.save_figure(self, fig, analysisName)

    def _run_analysis(self):
        taskList = self.dataSet.get_analysis_tasks()

        reportTime = int(time.time())
        for t in taskList:
            currentTask = self.dataSet.load_analysis_task(t)
            if currentTask.is_complete():
                slurmDF = self._generate_slurm_report(currentTask)
                self.dataSet.save_dataframe_to_csv(slurmDF, t, self,
                                                   'reports')
                dfStream = io.StringIO()
                slurmDF.to_csv(dfStream, sep='|')
                self._plot_slurm_report(slurmDF, t)

                try:
                    requests.post('http://www.georgeemanuel.com/merlin/post',
                                  data=currentTask.get_parameters(),
                                  files={'_'.join([t, self.dataSet.dataSetName,
                                                   str(reportTime)])
                                         + '.csv': dfStream},
                                  timeout=10)
                except requests.exceptions.RequestException as e:
                    pass

        datasetMeta = {
            'image_width': self.dataSet.get_image_dimensions()[0],
            'image_height': self.dataSet.get_image_dimensions()[1],
            'barcode_length': self.dataSet.get_codebook().get_bit_count(),
            'barcode_count': self.dataSet.get_codebook().get_barcode_count(),
            'fov_count': len(self.dataSet.get_fovs()),
            'z_count': len(self.dataSet.get_z_positions),
            'sequential_count': len(self.dataSet.get_sequential_rounds()),
            'dataset_name': self.dataSet.dataSetName,
            'report_time': reportTime
        }
        try:
            requests.post('http://www.georgeemanuel.com/merlin/post',
                          data=datasetMeta, timeout=10)
        except requests.exceptions.RequestException as e:
            pass