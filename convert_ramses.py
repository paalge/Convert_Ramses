#!/usr/bin/env python3

import numpy as np
import pandas as pd
import xarray as xa
import itertools
import io
import uuid
import datetime as dt


class Ramses():

    def __init__(self, attrs, ship, azimuth, zenith):
        '''
        Class for holding building an Ramses dataset and writing it as a netCDF
        file

        Parameters
        ----------
        attrs: dict
            A dictionary with the dataset attributes (use __read_dsets_in_file)
            to make it

        ship: str
            The ship used

        azimuth: float
            The orientation of the sensor relative to the ship bow in degrees
            270 is port, 90 is starboard,
            None for vertical sensor

        zenith: float
            The zenith angle of the sensor
            0 is zenith, 180 is down
        '''
        self.ds = xa.Dataset()
        self.ds.attrs['platformname'] = ship
        self.ds.attrs['platform'] = 'ship'
        self.ds.attrs['instrument'] = attrs['IDDevice'].strip()
        self.ds.attrs['Make'] = 'TriOS Ramses'
        self.ds.attrs['Calibration'] = attrs['IDDataCal'].strip()
        self.ds.attrs['Conventions'] = 'CF-1.8, ACDD-1.3'
        self.ds.attrs['title'] = 'Irradiance and radiance measurements around Tromsø'
        self.ds.attrs['summary'] = 'This is a set of three files containing radiance and irradiance measurements from the fjords around Tromsø, intended to be used for satellite calibration.'

        self.ds.attrs['id'] = str(uuid.uuid4())  # Set as random for now
        self.ds.attrs['naming_authority'] = 'UiT The Arctic University of Norway'
        self.ds.attrs['source'] = 'TriOS Ramses radiometer'
        self.ds.attrs['processing_level'] = 'Radiometrically calibrated'
        self.ds.attrs['standard_name_vocabulary'] = 'CF Standard Name Table v73'
        self.ds.attrs['date_created'] = dt.datetime.utcnow().isoformat()
        self.ds.attrs['creator_name'] = 'Pål Gunnar Ellingsen'
        self.ds.attrs['creator_email'] = 'pal.g.ellingsen@uit.no'
        self.ds.attrs['institution'] = 'UiT The Arctic University of Norway'
        self.ds.attrs['project'] = 'The Nansen Legacy, CIRFA'

        self.time = []
        self.lats = []
        self.lons = []
        self.ints = []
        self.waves = None
        self.int_time = []  # Integration time

        # This is True for the irradiance sensor
        self.vert = 'SAMIP' in attrs['MethodName']

        self.is_up_rad = zenith < 90
        self.ds.coords['sensor_zenith_angle'] = float(zenith)
        self.ds.coords['sensor_zenith_angle'].attrs['units'] = 'degree'
        self.ds.coords['sensor_zenith_angle'].attrs['standard_name'] = 'sensor_zenith_angle'
        self.ds.coords['sensor_zenith_angle'].attrs['comment'] = 'This is the mounted angle and does not take into account the ship movements'

        if self.vert:
            # self.pres = []
            # self.presvalid = []
            self.inclv = []
            self.inclvalid = []
            self.inclx = []
            self.incly = []
        else:
            self.ds.coords['relative_sensor_azimuth_angle'] = float(azimuth)
            self.ds.coords['relative_sensor_azimuth_angle'].attrs['units'] = 'degree'
            self.ds.coords['relative_sensor_azimuth_angle'].attrs['standard_name'] = 'relative_sensor_azimuth_angle'
            self.ds.coords['relative_sensor_azimuth_angle'].attrs['comment'] = 'With respect to the bow, starboard is 90 degrees. This is the mounted angle and does not take into account the ship movements'

    def add_dataset(self, attrs, data):
        '''
        Add a dataset to the class

        Parameters
        ----------
        attrs: dict
            The dataset attributes

        data : pandas Dataframe
            The data

        '''
        inst = attrs['IDDevice'].strip()
        if self.ds.attrs['instrument'] != inst:
            raise ValueError('Trying to write data for ' + inst +
                             ' into a file for instrument ' + self.ds.attrs['instrument'])
        time = pd.Timestamp(attrs['DateTime'],
                            tz='Europe/Oslo').tz_convert('UTC')
        # Need to remove timezone after converting to UTC
        time = time.tz_localize(None)
        self.time.append(time)
        self.lats.append(float(attrs['PositionLatitude']))
        self.lons.append(float(attrs['PositionLongitude']))
        self.int_time.append(np.int32(attrs['IntegrationTime']))

        self.ints.append(data['intensity'])

        # Check that the wavelengths are the same
        if self.waves is not None:
            if not(np.array_equal(self.waves, data['radiation_wavelength'])):
                raise ValueError(
                    'Wavelengths have changed over time for the same instrument')
        else:
            self.waves = data['radiation_wavelength']

        if self.vert:
            # self.pres.append(float(attrs['Pressure']))
            self.inclv.append(float(attrs['InclV']))
            self.inclx.append(float(attrs['InclX']))
            self.incly.append(float(attrs['InclY']))
            self.inclvalid.append(bool(attrs['InclValid']))
            # self.presvalid.append(bool(attrs['PressValid'])

    def write_netcdf(self, filename):
        '''
        Write the datasets in the class to a netcdf file

        Parameters
        ----------
        filename: str
            The filename for the instrument
        '''
        # Make the coordinates
        lats = np.asarray(self.lats)
        self.ds.coords['lat'] = (('time'), lats)
        self.ds.coords['lat'].attrs['standard_name'] = 'latitude'
        self.ds.coords['lat'].attrs['long_name'] = 'latitude'
        self.ds.coords['lat'].attrs['units'] = 'degrees_north'

        self.ds.attrs['geospatial_lat_min'] = np.min(lats)
        self.ds.attrs['geospatial_lat_max'] = np.max(lats)

        lons = np.asarray(self.lons)
        self.ds.coords['lon'] = (('time'), lons)
        self.ds.coords['lon'].attrs['standard_name'] = 'longitude'
        self.ds.coords['lon'].attrs['long_name'] = 'longitude'
        self.ds.coords['lon'].attrs['units'] = 'degrees_east'

        self.ds.attrs['geospatial_lon_min'] = np.min(lons)
        self.ds.attrs['geospatial_lon_max'] = np.max(lons)

        times = np.asarray(self.time)
        print(times)
        self.ds.coords['time'] = times
        self.ds.coords['time'].attrs['standard_name'] = 'time'
        self.ds.coords['time'].attrs['long_name'] = 'UTC time'
        self.ds.coords['time'].attrs['tz'] = 'UTC'

        self.ds.attrs['time_coverage_start'] = str(times[0].isoformat())
        self.ds.attrs['time_coverage_end'] = str(times[-1].isoformat())

        self.ds.coords['wave'] = (('wave'), self.waves)
        self.ds.coords['wave'].attrs['standard_name'] = 'radiation_wavelength'
        self.ds.coords['wave'].attrs['long_name'] = 'Calibrated wavelength'
        self.ds.coords['wave'].attrs['units'] = 'nm'

        self.ds['int'] = (('time'), np.asarray(self.int_time))
        self.ds['int'].attrs['long_name'] = 'Integration time'
        self.ds['int'].attrs['units'] = 'ms'

        # Input the data
        if self.vert:
            print('shape', np.asarray(self.ints).shape)
            self.ds['irr'] = (
                ('time', 'wave'), np.asarray(self.ints))
            self.ds['irr'].attrs['units'] = 'mW m-2 nm-1'
            self.ds['irr'].attrs['long_name'] = 'surface_downwelling_spherical_irradiance_per_unit_wavelength_in_air'
            # self.ds['irr'].attrs['long_name'] = 'Downwelling irradiance per nm'

            self.ds['inclV'] = (('time'), np.asarray(self.inclv))
            # self.ds['inclV'].attrs['standard_name']='pressure'
            self.ds['inclV'].attrs['long_name'] = 'Vertical inclination'
            self.ds['inclV'].attrs['units'] = 'degree'

            self.ds['inclx'] = (('time'), np.asarray(self.inclx))
            self.ds['inclx'].attrs['long_name'] = 'X inclination'
            self.ds['inclx'].attrs['units'] = 'degree'

            self.ds['incly'] = (('time'), np.asarray(self.incly))
            self.ds['incly'].attrs['long_name'] = 'Y inclination'
            self.ds['incly'].attrs['units'] = 'degree'

            self.ds['inclV_q'] = (('time'), np.asarray(self.inclvalid))
            self.ds['inclV_q'].attrs['long_name'] = 'Validity inclination'
            self.ds['inclV_q'].attrs['comment'] = 'Boolean, where True is valid'
        elif self.is_up_rad:
            self.ds['radd'] = (
                ('time', 'wave'), np.asarray(self.ints))
            self.ds['radd'].attrs['units'] = 'mW m-2 nm-1 sr-1'
            self.ds['radd'].attrs['standard_name'] = 'surface_downwelling__radiance_per_unit_wavelength_in_air'
            self.ds['radd'].attrs['long_name'] = 'Downwelling radiance per nm'

        else:
            self.ds['radu'] = (
                ('time', 'wave'), np.asarray(self.ints))
            self.ds['radu'].attrs['units'] = 'mW m-2 nm-1 sr-1'
            self.ds['radu'].attrs['standard_name'] = 'surface_upwelling_radiance_per_unit_wavelength_in_air'
            self.ds['radu'].attrs['long_name'] = 'Upwelling radiance measured above the sea per nm'
            self.ds['radu'].attrs['comment'] = 'Measured Upwelling radiance per nm'

        # Write filename
        self.ds.to_netcdf(filename+self.ds.attrs['instrument']+'.nc')


def __read_dsets_in_file(f):
    dsets = []
    with open(f) as infile:
        n = 0
        while True:  # n < 20:
            it = itertools.dropwhile(
                lambda line: line.strip() != '[Spectrum]', infile)
            if next(it, None) is None:
                break
            dsets.append(list(itertools.takewhile(
                lambda line: line.strip() != '[END] of [Spectrum]', it)))
            n += 1
    return dsets


def splitt_dset(dset):
    dstart = None
    attrs = {}
    for idx, line in enumerate(dset):
        # print(idx, line)
        if '= RAW' in line:  # We don't want this
            return None, None
        elif line.strip() == '[Attributes]':
            continue
        elif line.strip() == '[END] of [Attributes]':
            dstart = idx+2
            break
        key, value = line.strip().split('=')
        attrs[key.strip()] = value.strip()
    data = pd.read_table(io.StringIO('\n'.join(dset[dstart:-1])), sep=' ', names=[
        'radiation_wavelength', 'intensity', 'intensity_error', 'status'], usecols=[1, 2, 3, 4])
    return attrs, data


def dsets_to_xarray(dsets, filename, ship='HYAS', up_rad='SAM_86A4', zenith=40, az=270):
    ramses = {}
    for dset in dsets:
        attrs, data = splitt_dset(dset)
        if attrs is None:
            continue

        dev = attrs['IDDevice']
        if dev in ramses.keys():
            ramses[dev].add_dataset(attrs, data)
        else:  # Not made yet
            if 'SAMIP' in attrs['MethodName']:  # Irradiance
                ramses[dev] = Ramses(attrs, ship, None, 0)
            elif dev == up_rad:  # Upwelling rad
                ramses[dev] = Ramses(attrs, ship, az, 180-zenith)
            else:  # Downwelling rad
                ramses[dev] = Ramses(attrs, ship, az, zenith)

    for r in ramses:
        ramses[r].write_netcdf(filename)


def main():
    f = "Tribox_9854_Spectra_2020-08-30_22-09-10_to_2020-09-04_10-08-00.dat"
    dsets = __read_dsets_in_file(f)
    dsets_to_xarray(dsets, '../2020-09-10_')


if __name__ == "__main__":
    main()
