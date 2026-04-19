"""
Sentinel-2 Data Acquisition using Google Earth Engine

Uses Google Earth Engine (GEE) to access Sentinel-2 imagery
instead of downloading files. Provides efficient time-series analysis.
"""

import ee
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Dict


class SentinelGEEData:
    """Access and process Sentinel-2 imagery from Google Earth Engine"""
    
    def __init__(self, gee_project: str = None):
        """
        Initialize Sentinel-2 GEE accessor
        
        Args:
            gee_project: GEE project name (e.g., 'r05-general-projects')
                        If None, uses default initialization
        """
        try:
            if gee_project:
                ee.Initialize(project=gee_project)
            else:
                ee.Initialize()
            self.gee_project = gee_project
            self.metadata = {}
        except Exception as e:
            print(f"Note: GEE already initialized or requires authentication: {e}")
    
    def load_sentinel2_collection(self,
                                 aoi: ee.Geometry,
                                 start_date: str,
                                 end_date: str,
                                 cloud_cover: float = 20.0) -> ee.ImageCollection:
        """
        Load Sentinel-2 image collection from GEE
        
        Args:
            aoi: Area of interest as ee.Geometry
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            cloud_cover: Maximum cloud cover threshold (0-100)
            
        Returns:
            ee.ImageCollection of Sentinel-2 images
        """
        sentinel2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                    .filterBounds(aoi)
                    .filterDate(start_date, end_date)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover))
                    .map(self._cloud_mask)
                    .map(self._scale_sentinel2))
        
        return sentinel2
    
    @staticmethod
    def _cloud_mask(image: ee.Image) -> ee.Image:
        """
        Apply cloud masking to Sentinel-2 image
        
        Args:
            image: Sentinel-2 image
            
        Returns:
            Masked image
        """
        qa = image.select('QA60')
        scl = image.select('SCL')

        # Bits 10 and 11 are clouds and cirrus in QA60.
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        qa_mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
            qa.bitwiseAnd(cirrus_bit_mask).eq(0)
        )

        # Remove cloud shadow, medium/high cloud, cirrus, and snow/ice from SCL.
        scl_mask = (
            scl.neq(3)   # cloud shadow
            .And(scl.neq(8))   # cloud medium probability
            .And(scl.neq(9))   # cloud high probability
            .And(scl.neq(10))  # thin cirrus
            .And(scl.neq(11))  # snow/ice
        )

        # Keep only pixels valid in all masks.
        combined_mask = qa_mask.And(scl_mask)
        return image.updateMask(combined_mask)
    
    @staticmethod
    def _scale_sentinel2(image: ee.Image) -> ee.Image:
        """
        Apply scaling factors to Sentinel-2 bands
        
        Args:
            image: Sentinel-2 image
            
        Returns:
            Scaled image with reflectance values [0, 1]
        """
        optical_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
        scaled_optical = image.select(optical_bands).multiply(0.0001)
        other_bands = image.select(image.bandNames().removeAll(optical_bands))
        return scaled_optical.addBands(other_bands).copyProperties(image, image.propertyNames())
    
    def extract_ndvi_timeseries(self,
                               aoi: ee.Geometry,
                               start_date: str,
                               end_date: str,
                               cloud_cover: float = 20.0,
                               reducer: str = 'mean') -> Tuple[List[str], np.ndarray]:
        """
        Extract NDVI time series from Sentinel-2 collection
        
        Args:
            aoi: Area of interest as ee.Geometry
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            cloud_cover: Maximum cloud cover threshold
            reducer: Reducer to apply ('mean', 'median', 'min', 'max')
            
        Returns:
            Tuple of (dates, ndvi_values)
        """
        sentinel2 = self.load_sentinel2_collection(aoi, start_date, end_date, cloud_cover)
        
        # Calculate NDVI
        def add_ndvi(image):
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            return image.addBands(ndvi)
        
        ndvi_collection = sentinel2.map(add_ndvi)
        
        # Get reducer function
        if reducer == 'median':
            reducer_fn = ee.Reducer.median()
        elif reducer == 'min':
            reducer_fn = ee.Reducer.min()
        elif reducer == 'max':
            reducer_fn = ee.Reducer.max()
        else:
            reducer_fn = ee.Reducer.mean()
        
        # Calculate mean NDVI for each image
        ndvi_timeseries = ndvi_collection.select('NDVI').map(
            lambda img: img.reduceRegion(reducer_fn, aoi, 10).set({'system:time_start': img.get('system:time_start')})
        )
        
        # Convert to simple features
        info = ndvi_collection.select('NDVI').aggregate_array('system:time_start').getInfo()
        dates = [datetime.fromtimestamp(t / 1000).strftime('%Y-%m-%d') for t in info]
        
        # Extract values
        ndvi_values = ndvi_timeseries.aggregate_array('NDVI').getInfo()
        
        return dates, np.array(ndvi_values)
    
    def get_sentinel2_composite(self,
                               aoi: ee.Geometry,
                               start_date: str,
                               end_date: str,
                               cloud_cover: float = 20.0,
                               method: str = 'median') -> ee.Image:
        """
        Create composite image from Sentinel-2 collection
        
        Args:
            aoi: Area of interest as ee.Geometry
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            cloud_cover: Maximum cloud cover threshold
            method: Composite method ('median', 'mean', 'min', 'max')
            
        Returns:
            Composite ee.Image
        """
        sentinel2 = self.load_sentinel2_collection(aoi, start_date, end_date, cloud_cover)
        
        if method == 'mean':
            composite = sentinel2.mean()
        elif method == 'min':
            composite = sentinel2.min()
        elif method == 'max':
            composite = sentinel2.max()
        else:  # median
            composite = sentinel2.median()
        
        return composite
