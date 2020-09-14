#!/usr/bin/env python3
# encoding: utf-8
'''
 -- This program reads in TriOS Ramses data
 an entry
@author:     Pål Ellingsen
@contact:    pal.g.ellingsen@uit.no
@deffield    updated: Updated
'''

import numpy as np
import pandas as pd
import xarray as xa
import itertools
import io
import os
import uuid
import datetime as dt
import yaml
import warnings
from argparse import ArgumentParser, RawDescriptionHelpFormatter

__all__ = []
__version__ = 0.1
__date__ = '2020-09-10'
__updated__ = '2020-09-14'


class Ramses():

    def __init__(self, attrs, config, filename,  append_to=False):
        '''
        Class for holding building an Ramses dataset and writing it as a netCDF
        file

        Parameters
        ----------
        attrs: dict
            A dictionary with the dataset attributes (use __read_dsets_in_file)
            to make it


        config: dict
            A dictionary with the config for the file, if appending these values will be taken from the previous file. See file that is together with this for example of what should be there

        filename: str
            Where to store the file

        append_to: bool
            True to append to file
            Default: False
        '''

        self.ds = xa.Dataset()
        self.time = []
        self.lats = []
        self.lons = []
        self.ints = []
        self.waves = None
        self.int_time = []  # Integration time

        # This is True for the irradiance sensor
        self.vert = 'SAMIP' in attrs['MethodName']
        # This is True for the radiance sensor pointing up
        self.is_up_rad = config['up_rad'] in attrs['IDDevice']
        if self.vert:
            # self.pres = []
            # self.presvalid = []
            self.inclv = []
            self.inclvalid = []
            self.inclx = []
            self.incly = []

        self.new = not(append_to)  # Check if we are making a new file
        self.filename = filename
        self.config = config

        if self.new:  # We are making a new file
            self.ds.attrs['platformname'] = config['platformname']
            self.ds.attrs['platform'] = config['platform']
            self.ds.attrs['instrument'] = attrs['IDDevice'].strip()
            self.ds.attrs['Make'] = 'TriOS Ramses'
            self.ds.attrs['Calibration'] = attrs['IDDataCal'].strip()
            self.ds.attrs['Conventions'] = 'CF-1.8, ACDD-1.3'
            self.ds.attrs['title'] = config['title']
            self.ds.attrs['summary'] = config['summary']

            self.ds.attrs['id'] = str(uuid.uuid4())  # Set as random for now
            self.ds.attrs['naming_authority'] = config['naming_authority']
            self.ds.attrs['source'] = 'TriOS Ramses radiometer'
            self.ds.attrs['processing_level'] = 'Radiometrically calibrated'
            self.ds.attrs['standard_name_vocabulary'] = 'CF Standard Name Table v73'
            self.ds.attrs['date_created'] = dt.datetime.utcnow().isoformat()
            self.ds.attrs['creator_name'] = config['creator_name']
            self.ds.attrs['creator_email'] = config['creator_email']
            self.ds.attrs['institution'] = config['institution']
            self.ds.attrs['project'] = config['project']

            if self.vert:  # Irradiance instrument
                zenith = 0.0
            elif self.is_up_rad:  # Upwelling Radiance instrument
                zenith = 180 - float(config['zenith_angle'])
            else:  # Downwelling Radiance instrument
                zenith = float(config['zenith_angle'])

            self.ds.coords['sensor_zenith_angle'] = [zenith]
            self.ds.coords['sensor_zenith_angle'].attrs['units'] = 'degree'
            self.ds.coords['sensor_zenith_angle'].attrs['standard_name'] = 'sensor_zenith_angle'
            self.ds.coords['sensor_zenith_angle'].attrs['comment'] = 'This is the mounted angle and does not take into account the ship movements'

            if not(self.vert):
                self.ds.coords['relative_sensor_azimuth_angle'] = [float(
                    config['rel_az'])]
                self.ds.coords['relative_sensor_azimuth_angle'].attrs['units'] = 'degree'
                self.ds.coords['relative_sensor_azimuth_angle'].attrs['standard_name'] = 'relative_sensor_azimuth_angle'
                self.ds.coords['relative_sensor_azimuth_angle'].attrs['comment'] = 'With respect to the bow, starboard is 90 degrees.'

        else:  # This is for appending to a file
            self.append = xa.open_dataset(filename)
            # Check that it is the same instrument:
            if self.append.attrs['instrument'] != attrs['IDDevice']:
                raise(
                    'Trying to append to a file that has data from a different instrument')

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
        if self.new:
            if self.ds.attrs['instrument'] != inst:
                raise ValueError('Trying to write data for ' + inst +
                                 ' into a file for instrument ' + self.ds.attrs['instrument'])
        else:
            if self.append.attrs['instrument'] != inst:
                raise ValueError('Trying to write data for ' + inst +
                                 ' into a file for instrument ' + self.append.attrs['instrument'])

        time = pd.Timestamp(attrs['DateTime'],
                            tz='Europe/Oslo').tz_convert('UTC')
        # Need to remove timezone after converting to UTC
        time = time.tz_localize(None)

        if not(self.new):
            if time in self.append.time:  # Data already there
                return

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

    def write_netcdf(self):
        '''
        Writes or appends the datasets in the class to a netcdf file

        Parameters
        ----------
        filename: str
            The filename for the instrument
        '''
        # Make the coordinates

        if len(self.time) == 0:
            warnings.warn('No new data to append, not doing anything')
            return

        lats = np.asanyarray(self.lats)
        self.ds.coords['lat'] = (('time'), lats)
        self.ds.coords['lat'].attrs['standard_name'] = 'latitude'
        self.ds.coords['lat'].attrs['long_name'] = 'latitude'
        self.ds.coords['lat'].attrs['units'] = 'degrees_north'

        lons = np.asarray(self.lons)
        self.ds.coords['lon'] = (('time'), lons)
        self.ds.coords['lon'].attrs['standard_name'] = 'longitude'
        self.ds.coords['lon'].attrs['long_name'] = 'longitude'
        self.ds.coords['lon'].attrs['units'] = 'degrees_east'

        times = np.asarray(self.time)
        self.ds.coords['time'] = times
        self.ds.coords['time'].attrs['standard_name'] = 'time'
        self.ds.coords['time'].attrs['long_name'] = 'UTC time'
        self.ds.coords['time'].attrs['tz'] = 'UTC'

        self.ds.coords['wave'] = (('wave'), self.waves)
        self.ds.coords['wave'].attrs['standard_name'] = 'radiation_wavelength'
        self.ds.coords['wave'].attrs['long_name'] = 'Calibrated wavelength'
        self.ds.coords['wave'].attrs['units'] = 'nm'

        self.ds['int'] = (('time'), np.asarray(self.int_time))
        self.ds['int'].attrs['long_name'] = 'Integration time'
        self.ds['int'].attrs['units'] = 'ms'

        # Input the data
        if self.vert:
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

        def write_ranges(dset):
            dset.attrs['geospatial_lat_min'] = np.min(dset.lat[:].data)
            dset.attrs['geospatial_lat_max'] = np.max(dset.lat[:].data)

            dset.attrs['geospatial_lon_min'] = np.min(dset.lon[:].data)
            dset.attrs['geospatial_lon_max'] = np.max(dset.lon[:].data)

            dset.attrs['time_coverage_start'] = str(dset.time[0].data)
            dset.attrs['time_coverage_end'] = str(dset.time[-1].data)

            return dset
        # Write filename

        if self.new:
            ds = write_ranges(self.ds)
            # print(ds)
            ds.to_netcdf(self.filename)
        else:  # We are appending
            #print('append', self.append)
            #print('ds', self.ds)
            appended = xa.concat([self.append, self.ds], dim='time')
            appended = write_ranges(appended)
            self.append.close()
            if 'history' in appended.attrs:
                appended.attrs['history'] = appended.attrs['history']+' \n'
            else:
                appended.attrs['history'] = ''

            dt_now = dt.datetime.utcnow().isoformat()
            appended.attrs['history'] = appended.attrs['history'] +\
                dt_now + ', ' + \
                self.config['creator_name'] + ', ' + \
                __file__ + ', appending more data'

            appended['date_modified'] = dt_now
            appended.to_netcdf(self.filename)


def __read_dsets_in_file(f):
    dsets = []
    with open(f) as infile:
        n = 0
        while True:  # n < 20:
            # while n < 140:
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
            dstart = idx+3  # To comment lines and the first line of data is bad
            break
        key, value = line.strip().split('=')
        attrs[key.strip()] = value.strip()
    data = pd.read_table(io.StringIO('\n'.join(dset[dstart:-1])), sep=' ', names=[
        'radiation_wavelength', 'intensity', 'intensity_error', 'status'], usecols=[1, 2, 3, 4])
    return attrs, data


def dsets_to_xarray(dsets, folder, config):
    ramses = {}
    for dset in dsets:
        attrs, data = splitt_dset(dset)
        if attrs is None:
            continue

        dev = attrs['IDDevice']

        if dev in ramses.keys():
            ramses[dev].add_dataset(attrs, data)
        else:  # Not made yet
            filename = os.path.join(folder, dev+'.nc')
            append_to = False
            if os.path.isfile(filename):
                append_to = True
            ramses[dev] = Ramses(attrs, config, filename, append_to=append_to)

    for r in ramses:
        ramses[r].write_netcdf()


def parse_options():
    """
    Parse the command line options and return these. Also performs some basic
    sanity checks, like checking number of arguments.
    """
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)' % (
        program_version, program_build_date)
    program_shortdesc = '''
    This program reads in a *.dat* file produced by radiometric Ramses sensors 
    connected to a Tribox computer and converts it to three CF and ACDD compliant
    netCDF4 files. 
    '''
    program_license = '''%s
    Created by Pål Ellingsen on %s.
    Distributed on an "AS IS" basis without warranties,
    either expressed or implied.
    The only condition of use is that no one at TriOS or with association to 
    TriOS is allowed to use this without written permission from the author.
    USAGE
''' % (program_shortdesc, str(__date__))

    # Setup argument parser
    parser = ArgumentParser(description=program_license,
                            formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument('input', type=str, help='''The input file dat file ''')
    parser.add_argument(
        'output', type=str, help='''The output folder for storing the netcdf4 files. Any existing files will be appended to. ''')
    parser.add_argument('-V', '--version', action='version',
                        version=program_version_message)
    parser.add_argument('-c', dest='config', type=str, default='config.yaml',
                        help="The yaml config file. Se the default file for what needs to be in it [default: %(default)s]")

    # Process arguments
    args = parser.parse_args()

    # if args.verbose > 0:
    #     print("Verbose mode on")

    return args


def main(argv=None):
    '''
    This program reads in a *.dat* file produced by radiometric Ramses sensors 
    connected to a Tribox computer and converts it to three CF and ACDD compliant
    netCDF4 files. 
    '''
    args = parse_options()
    dsets = __read_dsets_in_file(args.input)

    with open(args.config, 'r') as y:
        config = yaml.load(y, Loader=yaml.FullLoader)

    dsets_to_xarray(dsets, args.output, config=config)


if __name__ == "__main__":
    main()
