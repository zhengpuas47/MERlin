import os
import csv
import numpy as np
import pandas
from typing import List, Iterator

import merlin

def _parse_barcode_from_string(inputString):
    return np.array([int(x) for x in inputString if x is not ' '])


class Codebook(object):

    '''
    A Codebook stores the association of barcodes to genes.
    '''

    def __init__(self, dataSet, filePath=None):
        '''
        Create a new Codebook for the data in the specified data set.

        If filePath is not specified, a previously stored Codebook
        is loaded from the dataSet if it exists. If filePath is specified,
        the Codebook at the specified filePath is loaded and
        stored in the dataSet, overwriting any previously stored 
        Codebook.
        '''
        self._dataSet = dataSet

        if filePath is not None:
            if not os.path.exists(filePath):
                filePath = os.sep.join(
                        [merlin.CODEBOOK_HOME, filePath])
        
            headerLength = 3
            barcodeData = pandas.read_csv(filePath, header=headerLength, 
                    skipinitialspace=True, usecols=['name', 'id', 'barcode'],
                    converters={'barcode': _parse_barcode_from_string}) 
            with open(filePath, 'r') as inFile:
                csvReader = csv.reader(inFile, delimiter=',')
                header = [row for i,row in enumerate(csvReader) \
                        if i < headerLength]

            bitNames = [x.strip() for x in header[2][1:]]

            self._data = self._generate_codebook_dataframe(
                    barcodeData, bitNames)

            self._dataSet.save_dataframe_to_csv(
                    self._data, 'codebook', index=False)
            
        else:
            self._data = self._dataSet.load_dataframe_from_csv('codebook')

    def _generate_codebook_dataframe(self, barcodeData, bitNames):
        dfData = np.array([[currentRow['name'], currentRow['id']] \
                    + currentRow['barcode'].tolist()\
               for i, currentRow in barcodeData.iterrows()])
        df = pandas.DataFrame(dfData, columns=['name', 'id'] + bitNames)
        df[bitNames] = df[bitNames].astype('uint8')
        return df

    def get_barcode(self, index: int) -> List[int]:
        '''Get the barcode with the specified index.

        Args:
            index: the index of the barcode in the barcode list
        Returns:
            A list of 0's and 1's denoting the barcode
        '''    
        return [self._data.loc[index][n] for n in self.get_bit_names()]

    def get_barcode_count(self) -> int: 
        '''
        Get the number of barcodes in this codebook.

        Returns:
            The number of barcodes, counting barcodes for blanks and genes
        '''
        return len(self._data)

    def get_bit_count(self) -> int:
        '''
        Get the number of bits used for MERFISH barcodes in this codebook.
        '''
        return len(self.get_bit_names())

    def get_bit_names(self) -> List[str]:
        '''Get the names of the bits for this MERFISH data set.

        Returns:
            A list of the names of the bits in order from the lowest to highest
        '''
        return [s for s in self._data.columns if s not in ['name', 'id']]

    def get_barcodes(self, ignoreBlanks=False) -> np.array:
        '''Get the barcodes present in this codebook.
        
        Args:
            ignoreBlanks: flag indicating whether barcodes corresponding 
                    to blanks should be included.
        Returns:
            A list of the barcodes reperesented as lists of bits.
        '''
        bitNames = self.get_bit_names()
        if ignoreBlanks:
            return np.array([[x[n] for n in bitNames] for i,x \
                    in self._data.iterrows() if 'Blank' not in x['name']])
        else:
            return np.array([[x[n] for n in bitNames] \
                    for i,x in self._data.iterrows()])

