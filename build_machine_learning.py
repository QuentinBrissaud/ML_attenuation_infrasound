#!/usr/bin/env python3

import utils_figures

from obspy.core.utcdatetime import UTCDateTime
import pandas as pd
from pdb import set_trace as bp
import numpy as np
import os
import pickle
import great_circle_calculator.great_circle_calculator as gcc

import matplotlib.pyplot as plt
import seaborn as sns

from obspy.geodetics.base import gps2dist_azimuth, kilometer2degrees, degrees2kilometers
import fluids
from scipy import interpolate

from multiprocessing import get_context
from functools import partial

from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skimage.transform import resize
        
from tensorflow import random as tf_random
    
def collect_one_c_profile(simulation, one_profile):

    location = (simulation.iloc[0].lat, simulation.iloc[0].lon)
    year, month, day = simulation.iloc[0].year, simulation.iloc[0].month, simulation.iloc[0].day
    date_UTC = UTCDateTime(year, month, day)
    zmax = one_profile.z.max()
    N = one_profile.shape[0]
    one_atmos_model = construct_atmospheric_model.run_MSISHWM_wrapper(location, date_UTC, zmax, N)
    
    return one_atmos_model
    
def generate_c_profiles(list_simulations_all, output_dir, type_model, list_idx_simulations):
    
    """
    Create a sound velocity dataset for a given number of simulations
    """

    ## Select all relevant simulations
    list_simulations = list_simulations_all.loc[list_simulations_all.nb_slice.isin(list_idx_simulations)]
    
    ## Loop over each slice
    grouped_simulations = list_simulations.groupby('nb_slice')
    atmos_models = pd.DataFrame()
    for ireal, (islice, simulation) in enumerate(grouped_simulations):
    
        print('Processing slice {islice}'.format(islice=islice))

        ## Retrieve simulation characteristics
        subdir   = 'slice_no_' + str(islice) + '/'
        summary_file = 'slice_no_' + str(islice) + '_summary.dat'
        c_file = 'slice_no_{islice}_velocity_{type_model}.csv'.format(islice=islice, type_model=type_model)
        
        if os.path.exists(output_dir + subdir + c_file):
            continue
        
        ## Retrieve profiles
        try:
            profiles = compute_numerical_solutions.read_RD_profiles(summary_file, output_dir + subdir)
        except:
            print('Can not read profiles at {summary_file}'.format(summary_file=summary_file))
            continue
            
        ## Load NCPA profile at x=0 km
        ranges = profiles.range.unique()
        irange, range_ = 0, 0.
        one_profile = profiles.loc[profiles.range==range_]
            
        ## Generate MSIS20 profile
        print(output_dir + subdir + c_file)
        if type_model == 'MSIS':
            one_atmos_model = collect_one_c_profile(simulation, one_profile)
            ## Store profile
            one_atmos_model[['z', 'c']].to_csv(output_dir + subdir + c_file, header=True, index=False)
    
        else:
            one_atmos_model = one_profile
            one_atmos_model['c'] = np.sqrt(401.87430086589046*one_atmos_model.t)
            one_atmos_model['slice_no'] = islice
            """
            one_atmos_model = collect_one_c_profile(simulation, one_profile)
            print(one_atmos_model)
            plt.close('all')
            plt.plot(np.sqrt(401.87430086589046*one_profile.t), one_profile.z, label='NCPA')
            plt.plot(one_atmos_model.c, one_atmos_model.z/1e3, label='MSIS')
            plt.legend()
            plt.show()
            return
            #np.sqrt(401.87430086589046*temp)
            """
            atmos_models = atmos_models.append(one_atmos_model)
        
    if type_model == 'NCPA':
        return atmos_models
    else:
        return None
        
    ##atmos_models.to_csv('/adhocdata/infrasound/2021_seed_infrAI/model_atmos_fixed/atmos1976_temp_models.csv', header=True, index=False)
        
def generate_c_profiles_CPUs(output_dir, nb_CPU=1, name_file='list_simulations.csv', type_model='MSIS'):

    """
    Generate sound velocity files using multiple CPUs
    """

    available_models = ['MSIS', 'NCPA']
    if not type_model in available_models:
        print('Model not recognized')
        return
        
    ## Read list of simulations in output_dir
    list_simulations = read_one_list_simulations(output_dir, name_file=name_file)
    
    ## Deploy on several CPUs
    available_simulations = list_simulations.nb_slice.unique()
    nb_simulations = available_simulations.size
    print('Read directory {output_dir}'.format(output_dir=output_dir))
    generate_c_profiles_partial = \
        partial(generate_c_profiles, list_simulations, output_dir, type_model)

    N = min(nb_CPU, nb_simulations)
    atmos_models = pd.DataFrame()
    ## If one CPU requested, no need for deployment
    if N == 1:
        atmos_models = generate_c_profiles_partial(available_simulations)
        
    ## Otherwise, we pool the processes
    else:

        step_idx =  nb_simulations//N
        list_of_lists = []
        for i in range(N):
            idx = np.arange(i*step_idx, (i+1)*step_idx)
            if i == N-1:
                idx = np.arange(i*step_idx, len(available_simulations))
            list_of_lists.append( available_simulations[idx] )
            
        with get_context("spawn").Pool(processes = N) as p:
            results = p.map(generate_c_profiles_partial, list_of_lists)
    
        if type_model == 'NCPA':
            for result in results:
                atmos_models = atmos_models.append(result)
            atmos_models.reset_index(drop=True, inplace=True)
            
    return atmos_models
    
def create_one_dataset(list_simulations_all, output_dir, method, r_normalization, distance_interp, max_profiles, list_idx_simulations):

    """
    Create a input (wind) and output (1d TL) dataset for a given number of simulations
    """

    options = {}
    
    ## Select all relevant simulations
    list_simulations = list_simulations_all.loc[list_simulations_all.nb_slice.isin(list_idx_simulations)]
    
    ## String formatting
    format_file_1d = '{output_dir}{subdir}{method}_1d{ext}.pe'
    
    ## Loop over each slice
    nb_slices = list_simulations.nb_slice.unique().size
    f0s = np.zeros(nb_slices)
    slice_nos = np.zeros(nb_slices)
    flag_inf = np.zeros(nb_slices, dtype=bool)
    list_orig_slice = np.zeros(nb_slices)
    grouped_simulations = list_simulations.groupby('nb_slice')
    TL_stack = np.array([])
    for ireal, (islice, simulation) in enumerate(grouped_simulations):

        ## Retrieve simulation characteristics
        #summary_file = simulation.iloc[0].summary_file
        #nb_slice = simulation.iloc[0].nb_slice
        subdir   = 'slice_no_' + str(islice) + '/'
        summary_file = 'slice_no_' + str(islice) + '_summary.dat'
        
        ## Get slice azimuth
        if not 'az' in simulation.columns:
            point_start = (simulation.iloc[0].lon, simulation.iloc[0].lat)
            point_end   = (simulation.iloc[-1].lon, simulation.iloc[-1].lat)
            az = gcc.bearing_at_p2(point_start, point_end)
        else:
            az = simulation.iloc[0].az
        az_rad = np.radians(az)
        
        ## Get original slice number
        orig_slice = simulation.iloc[0].nb_slice_orig
        list_orig_slice[ireal] = orig_slice

        ## Retrieve source characteristics
        f0 = simulation.iloc[0].f0
        f0s[ireal] = f0
        
        ## Get slice number
        slice_nos[ireal] = islice
        
        print(subdir, f0, az)
        
        ## Retrieve profiles
        try:
            profiles = compute_numerical_solutions.read_RD_profiles(summary_file, output_dir + subdir)
        except:
            print('Can not read profiles at {summary_file}'.format(summary_file=summary_file))
            continue

        ## Retrieve 1d TL if exists
        file_1d = format_file_1d.format(output_dir=output_dir, subdir=subdir, method=method, ext=compute_numerical_solutions.get_name_extension(options, islice))
        if os.path.isfile(file_1d):
            print(file_1d)
            TL_1d = pd.read_csv(file_1d, delim_whitespace=True, header=None)
            TL_1d.columns = ['r', 'baz', 'Re-p', 'Im-p']
            distances = TL_1d.r.values
            TL_1d = np.sqrt(TL_1d['Re-p']**2 + TL_1d['Im-p']**2).values
            ## Normalize TL to get log10(TL(r=r_normalization km)) = 0.
            f = interpolate.interp1d(distances, TL_1d, fill_value='extrapolate')
            TL_1d /= f(r_normalization)
            ## Convert to db
            TL_1d = 20*np.log10(TL_1d)
            
            if distances[-1] < distance_interp[-1]:
                distances = np.concatenate((distances, distance_interp[-1:]))
                TL_1d = np.concatenate((TL_1d, TL_1d[-1:]))
            if distances[0] > distance_interp[0]:
                distances = np.concatenate((distance_interp[:1], distances))
                TL_1d = np.concatenate((TL_1d[:1], TL_1d))
            f = interpolate.interp1d(distances, TL_1d, kind='cubic')
            TL_1d = f(distance_interp)
            
            
            ## Apply a smoothing over a given nb of km (given by apply_smoothing)
            """
            if apply_smoothing > 0:
                idx = np.argmin(abs(distance_interp-apply_smoothing))
                TL_1d_ = np.convolve(TL_1d, np.ones(idx)/idx, mode='valid')
                TL_1d_ = np.r_[TL_1d_, np.repeat(TL_1d_[-1], idx-1)]
            """
                     
            ## Just for debug
            if TL_1d[np.isnan(TL_1d)].size > 0:
                flag_inf[ireal] = True
            #TL_1d[np.isnan(TL_1d)] = 0.
     
            if ireal == 0:
                TL_stack = TL_1d
            else:
                TL_stack = np.c_[TL_stack, TL_1d]
        else:
            print('TL file not found at {dir}'.format(dir=file_1d))
         
        ## Store each profile
        ranges = profiles.range.unique()
        for irange, range_ in enumerate(ranges):
            one_profile = profiles.loc[profiles.range==range_]
            winds = np.sin(az_rad) * one_profile.u.values + np.cos(az_rad) * one_profile.v.values
            temp = one_profile.t.values
            temp = winds + np.sqrt(401.87430086589046*temp) # Standard atmosphere 1976
            
            ## ceff ratio
            """
            first_profile = profiles.loc[profiles.range==profiles.range.min()]
            winds_surface = np.sin(az_rad) * first_profile.u.values + np.cos(az_rad) * first_profile.v.values
            temp_surface = first_profile.t.iloc[0]
            temp_surface = winds_surface[0] + np.sqrt(401.87430086589046*temp_surface) # Standard atmosphere 1976
            
            temp /= temp_surface
            """
            
            if irange == 0:
                wind_stack_temp = winds
                temp_stack_temp = temp
            else:
                wind_stack_temp = np.c_[wind_stack_temp, winds]
                temp_stack_temp = np.c_[temp_stack_temp, temp]

        wind_stack_temp = wind_stack_temp[:,:max_profiles]
        temp_stack_temp = temp_stack_temp[:,:max_profiles]
        if ireal == 0:
            wind_stack = wind_stack_temp[None,...]
            temp_stack = temp_stack_temp[None,...]
        else:
            wind_stack = np.concatenate((wind_stack[...], wind_stack_temp[None,...]), axis=0)
            temp_stack = np.concatenate((temp_stack[...], temp_stack_temp[None,...]), axis=0)

    altitudes = one_profile.z.values
    if TL_stack.size > 0:
        TL_stack = TL_stack.T
        
    ## If missing simulations we remove all loaded simulations to avoid inconsistencies
    if not TL_stack.shape[0] == wind_stack.shape[0]:
        TL_stack = np.array([])
        
    print('Reading done')
    
    return altitudes, wind_stack, TL_stack, f0s, list_orig_slice, flag_inf, slice_nos, temp_stack



def read_one_list_simulations(output_dir, name_file='list_simulations.csv'):
    
    """
    Read list_simulations.csv file from "output_dir" and transform inputs to the right format
    """

    ## Create atmospheric profiles
    ints = ['nb_slice', 'year', 'month', 'day', 'hour']; dtype = {}; 
    for int_ in ints: dtype[int_] = float
    list_simulations = pd.read_csv(output_dir + '/' + name_file, sep=',', header=[0], keep_default_na=False, dtype=dtype)
    for int_ in ints: list_simulations[int_] = list_simulations[int_].astype(int)
    
    return list_simulations

def load_data(output_dir, nb_CPU=1, method='tloss', nb_pts_TL_interp=500, range_distance=[2., 999.], max_profiles=10, name_file='list_simulations.csv', r_normalization=5):

    """
    Load wind profiles, 1d TL curves, and dominant frequencies f0s for training
    """

    ## Read list of simulations in output_dir
    list_simulations = read_one_list_simulations(output_dir, name_file=name_file)
    
    ## Setup inputs
    options = {}
    distance_interp = np.linspace(range_distance[0], range_distance[1], nb_pts_TL_interp)
    
    ## Deploy on several CPUs
    available_simulations = list_simulations.nb_slice.unique()
    nb_simulations = available_simulations.size
    print('Read directory {output_dir}'.format(output_dir=output_dir))
    create_one_dataset_partial = \
        partial(create_one_dataset, list_simulations, output_dir, method, r_normalization, distance_interp, max_profiles)

    N = min(nb_CPU, nb_simulations)
    ## If one CPU requested, no need for deployment
    if N == 1:
        altitudes, wind_stack, TL_stack, f0s, list_orig_slice, flag_inf, slice_nos, temp_stack = create_one_dataset_partial(available_simulations)
        flag_inf = ~flag_inf
        wind_stack      = wind_stack[flag_inf,:]
        temp_stack      = temp_stack[flag_inf,:]
        if TL_stack.size > 0:
            TL_stack        = TL_stack[flag_inf,:] 
            
        f0s             = f0s[flag_inf]
        list_orig_slice = list_orig_slice[flag_inf]
        slice_nos = slice_nos[flag_inf]
        #slice_nos = available_simulations.loc[np.repeat(flag_inf, max_profiles+1)]

    ## Otherwise, we pool the processes
    else:

        step_idx =  nb_simulations//N
        list_of_lists = []
        for i in range(N):
            idx = np.arange(i*step_idx, (i+1)*step_idx)
            if i == N-1:
                idx = np.arange(i*step_idx, len(available_simulations))
            list_of_lists.append( available_simulations[idx] )
            
        with get_context("spawn").Pool(processes = N) as p:
            results = p.map(create_one_dataset_partial, list_of_lists)

        #slice_nos  = list_of_lists[0].tolist()
        altitudes  = results[0][0] 
        wind_stack = results[0][1]
        TL_stack   = np.array([])
        if results[0][2].size > 0:
            TL_stack   = results[0][2]
        f0s        = results[0][3]
        list_orig_slice = results[0][4]
        flag_inf    = ~results[0][5]
        slice_nos   = results[0][6]
        temp_stack   = results[0][7]
        for result, slice_no in zip(results[1:], list_of_lists[1:]):
            wind_stack = np.concatenate((wind_stack[...], result[1][...]), axis=0)
            if result[2].size > 0:
                TL_stack   = np.r_[TL_stack, result[2]]
            f0s        = np.concatenate((f0s, result[3]), axis=0)
            list_orig_slice = np.concatenate((list_orig_slice, result[4]), axis=0)
            flag_inf = np.concatenate((flag_inf, ~result[5]), axis=0)
            slice_nos = np.concatenate((slice_nos, result[6]), axis=0)
            temp_stack = np.concatenate((temp_stack[...], result[7][...]), axis=0)

        slice_nos   = slice_nos[flag_inf]
        wind_stack  = wind_stack[flag_inf,:]
        if TL_stack.size > 0:
            TL_stack    = TL_stack[flag_inf,:] 
        f0s         = f0s[flag_inf]
        list_orig_slice = list_orig_slice[flag_inf]
        temp_stack = temp_stack[flag_inf,:]
        
    return altitudes, wind_stack, temp_stack, distance_interp, TL_stack, f0s, list_orig_slice, slice_nos, list_simulations
    
def encode_input_image(method, encoded_input_in, coef_downsample, coef_upsample):

    """
    Pre-process input image to downscale/upscale
    """

    encoded_input = encoded_input_in[:]
    image_shape = encoded_input.shape
    encoded_input = resize(encoded_input, (image_shape[0], image_shape[1]//coef_downsample, image_shape[2]*coef_upsample), anti_aliasing=True)
    
    ## RGB channels - used for transfer learning
    if method == 'downsample_3channel':
        norm = plt.Normalize(encoded_input.min(), encoded_input.max())
        encoded_input = plt.cm.Greys(norm(encoded_input))[:,:,:,:3]
    
    ## One grey channel
    elif method == 'downsample_1channel':
        encoded_input = encoded_input[...,None]
    
    return encoded_input
    
class build_dataset:
    
    def __init__(self, output_dir, method='tloss', nb_CPU=1, nb_pts_TL_interp=500, r_normalization=5, \
                 range_distance=[2., 999.], max_profiles=10, use_ceff_inputs=True, 
                 name_file='list_simulations.csv'):

        self.output_dir = output_dir
        self.method = method
        self.max_profiles = max_profiles
        self.use_ceff_inputs = use_ceff_inputs
        self.range_distance = range_distance
        self.r_normalization = r_normalization
        self.altitudes, self.wind_stack, self.temp_stack, self.distances, self.TL_stack, self.f0s, self.list_orig_slice, self.slice_nos, self.list_simulations = \
            load_data(output_dir, nb_CPU=nb_CPU, method=method, nb_pts_TL_interp=nb_pts_TL_interp, \
                      range_distance=range_distance, max_profiles=max_profiles, name_file=name_file, r_normalization=r_normalization)

        ## Save input info
        if use_ceff_inputs:
            self.encoded_input = self.temp_stack[:]
        else:
            self.encoded_input = self.wind_stack[:]
        
        ## Save output info
        self.contains_output = False
        if self.TL_stack.size > 0:
            self.encoded_output = self.TL_stack[:]
            self.contains_output = True
        
        self.bool_encoded_input = False
        self.bool_encoded_output = False
        
    def _get_dataset_metadata(self):
    
        self.metadata = {}
        self.metadata['output_dir'] = self.output_dir
        self.metadata['max_profiles'] = self.max_profiles
        self.metadata['max_profiles'] = self.max_profiles
        self.metadata['range_distance'] = self.range_distance
        self.metadata['range_alt'] = [self.altitudes.min(), self.altitudes.max()]
        self.metadata['r_normalization'] = self.r_normalization
        self.metadata['TL_stack_shape'] = self.TL_stack.shape
        self.metadata['wind_stack_shape'] = self.wind_stack.shape
        self.metadata['encoded_wind_shape'] = self.encoded_input.shape
        self.metadata = pd.DataFrame([self.metadata])
        
    def determine_dataset_properties(self, alt_veffs=[[0., 20.], [20., 50.], [50., 100.]], ranges=np.linspace(0., 1000, 10), use_external_vel=True, external_types=['atmos_1976', 'atmos_1976', 'atmos_1976'], external_files={'MSIS_file': '/adhocdata/infrasound/2021_seed_infrAI/model_atmos_fixed/msise_temp_models.csv', 'atmos_1976':'/adhocdata/infrasound/2021_seed_infrAI/model_atmos_fixed/atmos1976_temp_models.csv'}, external_type_ground='atmos_1976'):
        """
        Compute range dependent Veff ratio for a given altiude
        """
        
        ## Determine sound speed for veff computation
        if use_external_vel:
            #msise = pd.read_csv(MSIS_file, header=[0])
            #if msise.z.max() < 1000:
            #    msise.z *= 1e3
            pd_external_files = pd.DataFrame()
            for type_ext in np.unique(external_types):
                msise = pd.read_csv(external_files[type_ext], header=[0])
                if msise.z.max() < 1000:
                    msise.z *= 1e3
                msise['type_ext'] = type_ext
                pd_external_files = pd_external_files.append(msise)
            pd_external_files.reset_index(drop=True, inplace=True)
            c_ground_orig = pd_external_files.loc[(pd_external_files['type_ext'] == external_type_ground)&(pd_external_files.z==0.), 'c'].values
            #print(pd_external_files.loc[(pd_external_files['type_ext'] == external_type_ground)&(pd_external_files.z==0.)])
            #print(c_ground_orig.shape)
             
        else:
            c_ground = np.sqrt(401.87430086589046*fluids.atmosphere.ATMOSPHERE_1976(0.).T)
                
        ## Wind lateral stds
        lateral_std_winds = self.wind_stack.std(axis=2)
        
        ## Determine veff_ratio at various altitudes
        self.properties = pd.DataFrame()
        for ialt_veff, alt_veff_bounds in enumerate(alt_veffs):
            
            str_alt = str(int(alt_veff_bounds[0])) + '-' + str(int(alt_veff_bounds[1]))
            ialt     = np.argmin(abs(self.altitudes - alt_veff_bounds[0]))
            ialt_end = np.argmin(abs(self.altitudes - alt_veff_bounds[1]))
            #print(alt_veff_bounds, ialt, ialt_end)
            data_shape = self.wind_stack.shape
            
            nb_alts = ialt_end - ialt + 1
            c_alt = np.zeros((self.slice_nos.size, nb_alts))
            """
            c_profiles = pd.DataFrame()
            for ii, islice in enumerate(self.slice_nos):
                c_file = '{output_dir}slice_no_{islice:0.0f}/slice_no_{islice:0.0f}_velocity.csv'.format(output_dir=self.output_dir, islice=islice)
                print(c_file)
                c_profile = pd.read_csv(c_file, header=[0])
                c_profiles = c_profiles.append(c_profile)
                #c_alt[ii,:] = c_profile['c'].values[ialt:ialt_end+1]
            c_profiles['slice_no'] = np.repeat(self.slice_nos, c_profile.shape[0])
            c_profiles.reset_index(drop=True, inplace=True)
            bp()
            """
            
            std = 0.
            if use_external_vel and external_types[ialt_veff] in ['MSIS', 'atmos_1976']:
                msise = pd_external_files.loc[(pd_external_files['type_ext'] == external_types[ialt_veff])]
                dz = self.altitudes[1]-self.altitudes[0]
                c_alt = msise.loc[(msise.z>=(alt_veff_bounds[0]-dz)*1e3)&(msise.z<=(alt_veff_bounds[1]+dz)*1e3)]
                #print(c_alt)
                c_alt = c_alt.groupby('slice_no').head(nb_alts).reset_index()
                c_alt = c_alt.c.values
                c_alt = c_alt.reshape((nb_alts, c_alt.size//nb_alts)).T
                c_alt = np.repeat(c_alt[...,None], ranges.size, axis=-1)
                #c_ground = np.repeat(c_ground_orig[...,None], nb_alts, axis=-1)
                c_ground = np.repeat(c_ground_orig[...,None], ranges.size, axis=-1)
            else:
                c_alt = np.zeros((nb_alts,))
                for idx, i in enumerate(range(ialt, ialt_end+1)):
                    ## Sound velocity at a given altitude
                    #c_alt[idx] = fluids.atmosphere.ATMOSPHERE_1976(self.altitudes[i]*1e3).v_sonic
                    c_alt[idx] = np.sqrt(401.87430086589046*fluids.atmosphere.ATMOSPHERE_1976(self.altitudes[i]*1e3).T)
                c_alt = np.repeat([c_alt], ranges.size, axis=0).T
            
            ## Compute veff ratio
            #print(self.wind_stack[:, ialt:ialt_end+1, :].shape, c_alt.shape, self.wind_stack[:, 0, :].shape, c_ground.shape)
            veff_ratio = np.max(self.wind_stack[:, ialt:ialt_end+1, :] + c_alt, axis=1) / (self.wind_stack[:, 0, :] + c_ground)
            
            ## Compute lateral std
            std = lateral_std_winds[:, ialt:ialt_end+1].max(axis=1)
            
            ## Convert 2d array into dataframe
            veff_ratio = pd.DataFrame(data=veff_ratio)
            veff_ratio.columns = ranges.tolist()
            veff_ratio = veff_ratio.unstack().reset_index()
            veff_ratio.columns = ['ranges', 'simu_id', 'veff-'+str_alt]
            if ialt_veff == 0:
                self.properties = veff_ratio.copy()
            else:
                self.properties['veff-'+str_alt] = veff_ratio['veff-'+str_alt]
            
            self.properties['std-'+str_alt] = np.repeat(std, ranges.size)
            
        self.properties['f0s'] = np.tile(self.f0s, ranges.size)
        self.properties['orig_slice'] = np.tile(self.list_orig_slice, ranges.size)
        
    def encode_inputs(self, n_components_input=10, coef_downsample=20, coef_upsample=4, method='PCA', 
                      input_wind=np.array([])):
    
        """
        Encode (PCA or SVD) or downsample wind models in the dataset or provided in "input_wind"
        """
    
        ## Select method
        if method == 'PCA':
            method_input = PCA
            
        elif method == 'SVD':
            method_input = TruncatedSVD
        
        elif not method in ['downsample_1channel', 'downsample_3channel']:
            sys.exit('Method for dimensionality reduction unknown')

        ## If a specific input is provided we only encode this one
        if input_wind.size > 0:
            encoded_input = input_wind
            specific_input = True
        else:
            try:
                encoded_input = self.encoded_input
            except:
                encoded_input = self.wind_stack[:]
            specific_input = False

        self.method_encoding = method
        ## PCA or SVD downsampling
        if method in ['PCA', 'SVD'] and (not specific_input or self.bool_encoded_input):
            ## Reshape 3d (n_simu, altitude, n_within_simu) wind model to 2d for SVD
            shape_init    = encoded_input.shape
            shape_end     = (shape_init[0], shape_init[2], n_components_input)
            shape_for_SVD = (shape_init[1], shape_init[0]*shape_init[2])
            reshaped_winds = encoded_input.transpose(1,0,2)
            reshaped_winds = reshaped_winds.reshape(shape_for_SVD).T
            
            ## Build scaler to remove mean and scale with std before SVD
            if not specific_input:
                self.scaler_winds = StandardScaler()
                
            reshaped_winds = self.scaler_winds.fit_transform(reshaped_winds)

            ## Optimization SVDs
            #range_n_components = np.arange(5, 20, 1)
            #scores = test_module.optimize_SVD(reshaped_winds, range_n_components)

            ## Compute SVD over whole dataset and reshape
            if not specific_input:
                self.SVD_input = method_input(n_components=n_components_input)
                self.n_components = n_components_input
            encoded_input = self.SVD_input.fit_transform(reshaped_winds)
            
            ## Error calculation
            recovered_input = self.SVD_input.inverse_transform(encoded_input)
            recovered_input = self.scaler_winds.inverse_transform(recovered_input)
            reshaped_winds_ = encoded_input.transpose(1,0,2)
            reshaped_winds_ = reshaped_winds.reshape(shape_for_SVD).T
            if not specific_input:
                self.error_input = np.sqrt(np.mean((recovered_input-reshaped_winds)**2))
                self.std_altitude_input = abs(recovered_input-reshaped_winds).std(axis=0)
            
            ## Convert output from nsample x ncomponents to nslices x nprofile_per_slice x ncomponents
            encoded_input = self.encoded_input.reshape(shape_end)
            
            ## Dummy values only used when considering input images
            altitudes_downsampled = None
            range_upsampled       = None
            if not specific_input:
                self.altitudes_downsampled = altitudes_downsampled
                self.range_upsampled       = np.range_upsampled
            
        ## RGB input channels - Used for transfer learning
        ## or one grey channel
        elif method in ['downsample_3channel', 'downsample_1channel']:
            if not specific_input:
                self.coef_downsample = coef_downsample
                self.coef_upsample = coef_upsample
            """
            image_shape = encoded_input.shape
            image_resized = resize(encoded_input, 
                                    (image_shape[0], image_shape[1]//coef_downsample, 
                                        image_shape[2]*coef_upsample), 
                                    anti_aliasing=True)
            
            altitudes_downsampled = self.altitudes[::coef_downsample]
            range_upsampled       = np.linspace(self.distances[0], self.distances[-1], image_shape[2]*coef_upsample)
            if not specific_input:
                self.altitudes_downsampled = altitudes_downsampled
                self.range_upsampled       = np.range_upsampled
            norm = plt.Normalize(image_resized.min(), image_resized.max())
            encoded_input = plt.cm.Greys(norm(image_resized))[:,:,:,:3]
            """
            encoded_input = encode_input_image(method, encoded_input, coef_downsample, coef_upsample)
            print('-->', encoded_input.shape, coef_downsample, coef_upsample)
            
            ## Build down/upscaled distances and altitude levels    
            altitudes_downsampled = self.altitudes[::coef_downsample]
            image_shape = encoded_input.shape
            range_upsampled = np.linspace(self.distances[0], self.distances[-1], image_shape[2]*coef_upsample)
            if not specific_input:
                self.altitudes_downsampled = altitudes_downsampled
                self.range_upsampled       = range_upsampled
        
        else:
            sys.exit('Input encoding impossible')
            
        """
        ## One grey channel
        elif method == 'downsample_1channel':
            image_shape = encoded_input.shape
            if not specific_input:
                self.coef_downsample = coef_downsample
            encoded_input = resize(encoded_input, 
                                    (image_shape[0], image_shape[1]//coef_downsample, 
                                        image_shape[2]*coef_upsample), 
                                    anti_aliasing=True)     
            altitudes_downsampled = self.altitudes[::coef_downsample]
            range_upsampled       = np.linspace(self.distances[0], self.distances[-1], image_shape[2]*coef_upsample)
            if not specific_input:
                self.altitudes_downsampled = altitudes_downsampled
                self.range_upsampled       = np.range_upsampled
            encoded_input = encoded_input[...,None]
        """
            
        ## If encoding an entire dataset with save the encoded inputs within the class
        if not specific_input:
            self.encoded_input = encoded_input
            
            ## Flag that we have encoded inputs
            self.bool_encoded_input = True
    
        ## If encoding specific inputs, simply return encoded inputs and corresponding x and y axes
        else:
            return range_upsampled, altitudes_downsampled, encoded_input
    
    def encode_outputs(self, n_components, method='PCA', scaler_output='minmax'):
    
        """
        Encode (PCA, SVD, or linear interpolation) 1d TL profiles
        n_components is the number of PCA/SVD truncated components 
        """
    
        ## Select method
        self.method_output_name = method
        if method == 'PCA':
            self.method_output = PCA
            
        elif method == 'SVD':
            self.method_output = TruncatedSVD
        
        elif method == 'linear':
            self.method_output = None
        
        else:
            sys.exit('Method for dimensionality reduction unknown')

        TL_stack = self.TL_stack

        ## Build scaler to remove mean and scale with std before SVD
        if scaler_output == 'standard':
            self.scaler_TL = StandardScaler()
            TL_stack = self.scaler_TL.fit_transform(TL_stack)
        elif scaler_output == 'minmax':
            self.scaler_TL = MinMaxScaler()
            TL_stack = self.scaler_TL.fit_transform(TL_stack)
        elif scaler_output == 'None':
            self.scaler_TL = lambda x : x
        else:
            sys.exit('Output scaler not recognized')
        
        ## Compute SVD over whole dataset and reshape
        if method == 'linear':
            #self.encoded_output = TL_stack
            self.encoded_output = self.TL_stack
            recovered_output    = self.TL_stack
        
        else:
            self.SVD_output = self.method_output(n_components=n_components)
            self.encoded_output = self.SVD_output.fit_transform(TL_stack)
            recovered_output = self.SVD_output.inverse_transform(self.encoded_output)
            recovered_output = self.scaler_TL.inverse_transform(recovered_output)
            
        self.error_output = np.sqrt(np.mean((recovered_output-self.TL_stack)**2))
        self.std_distance_output = abs(recovered_output-self.TL_stack).std(axis=0)

        ## Flag that we have encoded outputs
        self.bool_encoded_output = True

class history_class:
    
    def __init__(self, history):
        
        self.history = history

class fake_scaler_TL:

    def inverse_transform(self, x):
        return x
        
class build_NN:
    
    def __init__(self, output_dir, dataset):
    
        """
        Collect dataset information
        """
        
        self.output_dir = output_dir
        self.dataset = dataset
        try:
            self.dataset._get_dataset_metadata()
            self.dataset_metadata = self.dataset.metadata
        except:
            self.dataset_metadata = pd.DataFrame()
        self.input_source_shape = (1,)
        self.input_wind_shape = dataset.encoded_input.shape[1:]
        self.output_shape = dataset.encoded_output.shape[1]
        self.mse_errs = pd.DataFrame()
        self.uncertainty = np.array([])
        self.all_accuracies = pd.DataFrame()

    def split_train_test(self, seed=1, test_size=0.25, full_random=False, type='nb_slice_orig', scaler_output='standard', orig_idx_test=np.array([])):
    
        """
        Build input / output training and testing datasets
        """
        
        """
        train_properties = model.dataset.properties.loc[model.dataset.properties.simu_id.isin(model.train_ids)].groupby('simu_id').first()
        test_properties = model.dataset.properties.loc[model.dataset.properties.simu_id.isin(model.test_ids)].groupby('simu_id').first()
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
        sns.kdeplot(data=train_properties, x="veff-20-50", fill=True, common_norm=True, color="blue", alpha=.5, linewidth=0, ax=ax, legend=False)
        sns.kdeplot(data=test_properties, x="veff-20-50", fill=True, common_norm=True, color="red", alpha=.5, linewidth=0, ax=ax, legend=False)
        plt.show()
        """
        
        ## If split is done using categorical features (e.g., nb_slice_orig, ceff)
        if self.all_accuracies.shape[0] > 0 and False:
            bp()
        
        else:
            if not full_random:
            
                ## To avoid overfitting we put all the slices corresponding to the same original slice in the same set (training or validation)
                if type == 'nb_slice_orig':
                    orig_slice = self.dataset.properties.orig_slice.unique()
                    if orig_idx_test.size > 0:
                        self.test_orig_slice = orig_idx_test
                        self.train_orig_slice = np.setdiff1d(orig_slice, self.test_orig_slice)
                    else:
                        self.train_orig_slice, self.test_orig_slice = train_test_split(orig_slice, test_size=test_size, random_state=seed)
                    #self.train_ids = self.dataset.properties.loc[(self.dataset.properties.orig_slice.isin(self.train_orig_slice)) & (self.dataset.properties.ranges==0.), 'simu_id'].values
                    #self.test_ids = self.dataset.properties.loc[(self.dataset.properties.orig_slice.isin(self.test_orig_slice)) & (self.dataset.properties.ranges==0.), 'simu_id'].values
                    self.train_ids = self.dataset.properties.groupby('simu_id').first().reset_index()
                    self.train_ids = self.train_ids.loc[(self.train_ids.orig_slice.isin(self.train_orig_slice))].index.tolist()
                    self.train_ids = np.array(self.train_ids)
                    self.test_ids = self.dataset.properties.groupby('simu_id').first().reset_index()
                    self.test_ids = self.test_ids.loc[(self.test_ids.orig_slice.isin(self.test_orig_slice))].index.tolist()
                    self.test_ids = np.array(self.test_ids)
                    
                elif type == 'veff-20-50':
                    self.train_ids, self.test_ids = [], []
                    nb_range_values = 5
                    range_values = np.linspace(self.dataset.properties[type].min(), self.dataset.properties[type].max(), nb_range_values)
                    pd_cut = pd.cut(self.dataset.properties[type], bins=nb_range_values)#, labels=range_values)
                    self.dataset.properties['q-' + type] = [(pd_cut_val.left + pd_cut_val.right)/2 for pd_cut_val in pd_cut]
                    for q_value, data in self.dataset.properties.groupby('simu_id').first().reset_index().groupby('q-' + type): self.test_ids += data['simu_id'].sample(frac=test_size).values.tolist() 
                    self.train_ids = self.dataset.properties.loc[~self.dataset.properties.simu_id.isin(self.test_ids), 'simu_id'].unique().tolist()
                 
                else:
                    sys.exit('Train/test splitting type unknown')
                    
                ## Split the data using the simulation ids
                self.train_input_wind = self.dataset.encoded_input[self.train_ids,:]
                self.test_input_wind  = self.dataset.encoded_input[self.test_ids,:]
                self.train_input_f0 = self.dataset.f0s[self.train_ids]
                self.test_input_f0  = self.dataset.f0s[self.test_ids]
                self.train_input_slice_no = np.array(self.dataset.slice_nos)[self.train_ids]
                self.test_input_slice_no  = np.array(self.dataset.slice_nos)[self.test_ids]
                self.train_output = self.dataset.encoded_output[self.train_ids,:]
                self.test_output  = self.dataset.encoded_output[self.test_ids,:]
                
            else:
                orig_slice = self.dataset.properties.groupby('simu_id').first().orig_slice.values
                ids = self.dataset.properties.loc[self.dataset.properties.ranges==0.].simu_id.values
                (self.train_input_wind, self.test_input_wind, \
                    self.train_input_f0, self.test_input_f0, \
                    self.train_input_slice_no, self.test_input_slice_no, \
                    self.train_output, self.test_output, \
                    self.train_ids, self.test_ids, \
                    self.train_orig_slice, self.test_orig_slice) = \
                        train_test_split(self.dataset.encoded_input, self.dataset.f0s, self.dataset.slice_nos, self.dataset.encoded_output, ids, orig_slice, test_size=test_size, random_state=seed)
        
        ## Scale outputs using training dataset
        if self.dataset.method_output_name == 'linear':
        
            if scaler_output == 'standard':
                self.scaler_TL = StandardScaler()
                self.train_output = self.scaler_TL.fit_transform(self.train_output)
                self.test_output = self.scaler_TL.transform(self.test_output)
            elif scaler_output == 'minmax':
                self.scaler_TL = MinMaxScaler()
                self.train_output = self.scaler_TL.fit_transform(self.train_output)
                self.test_output = self.scaler_TL.transform(self.test_output)
            elif scaler_output == 'None':
                self.scaler_TL = fake_scaler_TL()
            else:
                sys.exit('Output scaler not recognized')
         
        if False:
            #self.mean_wind_ = self.dataset.wind_stack.mean(axis=0); self.std_wind_ = self.dataset.wind_stack.std(axis=0); input_scaler_wind_ = lambda winds: (winds - self.mean_wind_)/self.std_wind_; self.mean_temp = self.dataset.temp_stack.mean(axis=0); self.std_temp = self.dataset.temp_stack.std(axis=0); input_scaler_temp = lambda temp: (temp - self.mean_temp)/self.std_temp
            #id=10; wind_test=input_scaler_wind_(self.dataset.wind_stack[id,:]); temp_test=input_scaler_temp(self.dataset.temp_stack[id,:]); plt.plot(wind_test[:,0], self.dataset.altitudes); plt.plot(temp_test[:,0], self.dataset.altitudes); plt.show()
            #plt.plot(self.dataset.temp_stack[0,:,0]-self.dataset.wind_stack[0,:,0], self.dataset.altitudes); plt.plot(self.dataset.temp_stack[0,:,0], self.dataset.altitudes); plt.show()
            
            #self.mean_wind = self.train_input_wind.mean(axis=0); self.std_wind = self.train_input_wind.std(axis=0)
            self.mean_wind = self.train_input_wind.mean(axis=0); self.std_wind = self.train_input_wind.std(axis=0)
            self.max_wind = self.train_input_wind.min(axis=0); self.min_wind = self.train_input_wind.max(axis=0)
            self.input_scaler_wind = lambda winds: (winds - self.mean_wind)/self.std_wind
            #self.input_scaler_wind = lambda winds: (winds - self.min_wind)/(self.max_wind - self.min_wind)
            self.train_input_wind = self.input_scaler_wind(self.train_input_wind)
            self.test_input_wind = self.input_scaler_wind(self.test_input_wind)
                
        if False:
            self.mean_f0 = self.train_input_f0.mean(axis=0); self.std_f0 = self.train_input_f0.std(axis=0)
            self.max_f0 = self.train_input_f0.min(axis=0); self.min_f0 = self.train_input_f0.max(axis=0)
            self.input_scaler_f0 = lambda f0: (f0 - self.mean_f0)/self.std_f0
            #self.input_scaler_f0 = lambda f0: (f0 - self.min_f0)/(self.max_f0 - self.min_f0)
            self.train_input_f0 = self.input_scaler_f0(self.train_input_f0)
            self.test_input_f0 = self.input_scaler_f0(self.test_input_f0)
            
            """
            self.mean_inputs_one = np.mean(self.train_output, axis=0); 
            self.std_inputs_one  = np.std(self.train_output, axis=0); 
            m_aa   = np.repeat(np.repeat(mean_inputs_one[...,None], one_dataset.temp_stack.shape[2], axis=1)[None,...], one_dataset.temp_stack.shape[0], axis=0)
            std_aa = np.repeat(np.repeat(std_inputs_one[...,None], one_dataset.temp_stack.shape[2], axis=1)[None,...], one_dataset.temp_stack.shape[0], axis=0)
            """
        
    def build_model(self, type_model='FCN', doubleFCN=False, dense_layers_winds=[(10, 'relu'), (5, 'relu')], 
                    dense_layers_all=[(5, 'relu'), (5, 'relu')], CNN_layers=[((32, 3, 3), 'relu'), ((16, 3, 3), 'relu')],
                    activation_result='linear', learning_rate=1e-3, decay=1e-3 / 200, dropout_factor=0.2, 
                    loss_metric="mean_absolute_percentage_error", metrics=['mse'],
                    nb_trees=500, seed=1, oob_score=True, bootstrap=True, max_depth=100):

        """
        Select appropriate routine to build a ML model
        """
        
        print('Building model')

        self.type_model = type_model
        if type_model == 'FCN':
            self.build_model_FCN(doubleFCN=doubleFCN, dense_layers_winds=dense_layers_winds, 
                    dense_layers_all=dense_layers_all, activation_result=activation_result, 
                    learning_rate=learning_rate, decay=decay, 
                    loss_metric=loss_metric, metrics=metrics, dropout_factor=dropout_factor)
                    
        elif type_model == 'CNN':
            self.build_model_CNN(doubleFCN=doubleFCN, CNN_layers=CNN_layers, 
                    dense_layers_all=dense_layers_all, activation_result=activation_result, 
                    learning_rate=learning_rate, decay=decay, 
                    loss_metric=loss_metric, metrics=metrics, dropout_factor=dropout_factor)
        
        elif type_model == '1dCNN':
            self.build_model_1dCNN( CNN_layers=CNN_layers, 
                    dense_layers_all=dense_layers_all, activation_result=activation_result, 
                    learning_rate=learning_rate, decay=decay, 
                    loss_metric=loss_metric, metrics=metrics, dropout_factor=dropout_factor)
        
        elif type_model == 'transfer':
            self.build_model_transfer(dense_layers_all=dense_layers_all, 
                    activation_result=activation_result, learning_rate=learning_rate, decay=decay, 
                    loss_metric=loss_metric, metrics=metrics, dropout_factor=dropout_factor)
        
        elif type_model == 'forest':
            self.build_RF(nb_trees=nb_trees, seed=seed, oob_score=oob_score, bootstrap=bootstrap, 
                          max_depth=max_depth)

        else:
            sys.exit('ML model not found')

    def build_RF(self, nb_trees=500, seed=1, oob_score=True, bootstrap=True, 
                 max_depth=100):
                 
        """
        Build RF model
        """
    
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.pipeline import make_pipeline
        
        self.model = \
            make_pipeline(StandardScaler(),
                          ExtraTreesRegressor(n_estimators=nb_trees, random_state=seed, 
                                        oob_score=oob_score, bootstrap=bootstrap, 
                                        max_depth=max_depth))

    def build_model_FCN(self, doubleFCN=False, dense_layers_winds=[(10, 'relu'), (5, 'relu')], 
                    dense_layers_all=[(5, 'relu'), (5, 'relu')], dropout_factor=0.,
                    activation_result='linear', learning_rate=1e-3, decay=1e-3 / 200, 
                    loss_metric="mean_absolute_percentage_error", metrics=['mse']):

        """
        Build FCN architecture
        """

        ## import ML librairies
        #from tensorflow.keras.models import Sequential
        #from tensorflow.keras.layers import BatchNormalization
        #from tensorflow.keras.layers import Conv2D
        #from tensorflow.keras.layers import MaxPooling2D
        #from tensorflow.keras.layers import Activation
        #from tensorflow.keras.layers import Dropout
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import Flatten
        from tensorflow.keras.layers import Input
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dropout
        from tensorflow.keras.layers import concatenate
        from tensorflow.keras.optimizers import Adam, RMSprop
        from tensorflow.keras.layers.experimental import preprocessing
        
        print('loss_metric:', loss_metric)
        print('metrics:', metrics)
        
        ## define two sets of inputs
        inputA = Input(shape=self.input_wind_shape)
        inputB = Input(shape=self.input_source_shape)
        
        ## Data preprocessing
        normalizer_winds = preprocessing.Normalization()
        normalizer_winds.adapt(self.dataset.encoded_input)
        normalizer_f0s = preprocessing.Normalization()
        normalizer_f0s.adapt(self.dataset.f0s)
        
        ## the first branch operates on the first input
        #x = normalizer_winds(inputA)
        x = inputA
        x = Flatten()(x)
        if doubleFCN:
            for size, activation in dense_layers_winds:
                x = Dense(size, activation=activation)(x)

        ## Second branch for f0 -> only normalization
        y = inputB
        #y = normalizer_f0s(inputB)

        ## combine the output of the two branches
        combined = concatenate([x, y])

        ## apply a FC layer and then a regression prediction on the combined outputs
        z = combined
        if dropout_factor > 0.:
            z = Dropout(dropout_factor)(z)
        for size, activation in dense_layers_all:
            z = Dense(size, activation=activation)(z)
        
        ## Output layer
        z = Dense(self.output_shape, activation=activation_result)(z)

        ## our model will accept the inputs of the two branches and then output a TL1d map
        self.model = Model(inputs=[inputA, inputB], outputs=z)

        ## Optimizer
        opt = Adam(lr=learning_rate)#, decay=decay)
        self.model.compile(loss=loss_metric, optimizer=opt, metrics=metrics)
        
    def build_model_transfer(self, dense_layers_all=[(50, 'relu'), (5, 'relu')], dropout_factor=0.,
                    activation_result='linear', learning_rate=1e-3, decay=1e-3 / 200, 
                    loss_metric="mean_absolute_percentage_error", metrics=['mse']):
        
        """
        Use VG16 pretrained model to encode wind models
        """
        
        #from tensorflow.keras.models import Sequential
        #from tensorflow.keras.layers import BatchNormalization
        #from tensorflow.keras.layers import Conv2D
        #from tensorflow.keras.layers import MaxPooling2D
        #from tensorflow.keras.layers import Activation
        #from tensorflow.keras.layers import Dropout
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import Flatten
        from tensorflow.keras.layers import Input
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dropout
        from tensorflow.keras.layers import concatenate
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.applications import VGG16
        from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_VGG16
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50
        from tensorflow.keras.layers.experimental import preprocessing
        
        ## pretrained model
        #pretrained_model = VGG16
        #preprocess_model = preprocess_VGG16
        pretrained_model = ResNet50
        preprocess_model = preprocess_resnet50
        conv_base = pretrained_model(weights='imagenet', include_top=False, pooling=False, input_shape=self.input_wind_shape)
        for layer in conv_base.layers:
            layer.trainable = True
        
        ## define two sets of inputs
        inputA = Input(shape=self.input_wind_shape)
        inputB = Input(shape=self.input_source_shape)
        
        ## Data preprocessing
        normalizer_f0s = preprocessing.Normalization()
        normalizer_f0s.adapt(self.dataset.f0s)
        
        ## the first branch operates on the first input
        x = preprocess_model(inputA)
        x = conv_base(x)
        x = Flatten()(x)
        if dropout_factor > 0.:
            x = Dropout(dropout_factor)(x)
            
        ## Second branch for f0 -> only normalization
        y = normalizer_f0s(inputB)

        ## combine the output of the two branches
        combined = concatenate([x, y])
        
        z = combined
        for size, activation in dense_layers_all:
            z = Dense(size, activation=activation)(z)

        ## Output layer
        z = Dense(self.output_shape, activation=activation_result)(z)

        ## our model will accept the inputs of the two branches and then output a TL1d map
        self.model = Model(inputs=[inputA, inputB], outputs=z)

        ## Optimizer
        opt = Adam(lr=learning_rate)#, decay=decay)
        self.model.compile(loss=loss_metric, optimizer=opt, metrics=metrics)

    def build_model_CNN(self, doubleFCN=False, CNN_layers=[((32, 3, 3), 'relu'), ((16, 3, 3), 'relu')], 
                    dense_layers_all=[(5, 'relu'), (5, 'relu')], dropout_factor=0.,
                    activation_result='linear', learning_rate=1e-3, decay=1e-3 / 200, 
                    loss_metric="mean_absolute_percentage_error", metrics=['mse'], padding='same'):

        """
        Build CNN architecture
        """

        ## import ML librairies
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import BatchNormalization
        from tensorflow.keras.layers import Conv2D
        from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
        from tensorflow.keras.layers import Activation
        from tensorflow.keras.layers import Dropout
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import Flatten
        from tensorflow.keras.layers import Input
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import concatenate
        from tensorflow.keras.optimizers import Adam, RMSprop
        from tensorflow.keras.layers.experimental import preprocessing
        
        print('loss_metric:', loss_metric)
        print('metrics:', metrics)
        
        ## define two sets of inputs
        inputA = Input(shape=self.input_wind_shape)
        inputB = Input(shape=self.input_source_shape)
        
        ## Data preprocessing
        normalizer_winds = preprocessing.Normalization(axis=-1)
        normalizer_winds.adapt(self.train_input_wind)
        #wind_min = self.train_input_wind.min()
        #minmax_wind = self.train_input_wind.max() - wind_min
        #offset = - wind_min/minmax_wind
        normalizer_f0s = preprocessing.Normalization()
        normalizer_f0s.adapt(self.train_input_f0)
        
        """
        m_aa_one=np.mean(one_dataset.temp_stack, axis=(0,2)); 
        std_aa_one=np.std(one_dataset.temp_stack, axis=(0,2)); 
        plt.plot(m_aa, one_dataset.altitudes); plt.show()
        m_aa=np.repeat(np.repeat(np.mean(one_dataset.temp_stack, axis=(0,2))[...,None], one_dataset.temp_stack.shape[2], axis=1)[None,...], one_dataset.temp_stack.shape[0], axis=0)
        std_aa=np.repeat(np.repeat(np.std(one_dataset.temp_stack, axis=(0,2))[...,None], one_dataset.temp_stack.shape[2], axis=1)[None,...], one_dataset.temp_stack.shape[0], axis=0)
        aa=(one_dataset.temp_stack-m_aa)/std_aa; plt.plot((one_dataset.temp_stack[1000,:,0]-m_aa_one)/std_aa_one, one_dataset.altitudes); plt.plot(aa[1000,:,0], one_dataset.altitudes); plt.show()
        """
        
        ## the first branch operates on the first input
        print('new_pooling')
        x = normalizer_winds(inputA)
        #x = inputA
        #x = preprocessing.Rescaling(1.0/minmax_wind, offset=offset)(inputA)
        for size, activation in CNN_layers:
            x = Conv2D(size[0], size[1:], activation=activation, padding=padding)(x)
            x = BatchNormalization()(x)
            #x = MaxPooling2D((2, 2), padding=padding)(x)
            #x = AveragePooling2D((2, 2), padding=padding)(x)
            x = AveragePooling2D((5, 5), padding=padding)(x) # New

        x = Flatten()(x)
        #if dropout_factor > 0.:
        #    x = Dropout(dropout_factor)(x)
        
        ## Second branch for f0 -> only normalization
        y = normalizer_f0s(inputB)

        ## combine the output of the two branches
        combined = concatenate([x, y])
        
        ## apply a FC layer and then a regression prediction on the combined outputs
        z = combined
        for size, activation in dense_layers_all:
            z = Dense(size, activation=activation)(z)
            if dropout_factor > 0.:
                z = Dropout(dropout_factor)(z)
            
        ## Output layer
        z = Dense(self.output_shape, activation=activation_result)(z)

        ## our model will accept the inputs of the two branches and then output a TL1d map
        self.model = Model(inputs=[inputA, inputB], outputs=z)

        ## Optimizer
        opt = Adam(lr=learning_rate)#, decay=decay)
        self.model.compile(loss=loss_metric, optimizer=opt, metrics=metrics)
        
    def build_model_1dCNN(self, CNN_layers=[((32, 3), 'relu'), ((16, 3), 'relu')], 
                    dense_layers_all=[(5, 'relu'), (5, 'relu')], dropout_factor=0.,
                    activation_result='linear', learning_rate=1e-3, decay=1e-3 / 200, 
                    loss_metric="mean_absolute_percentage_error", metrics=['mse']):

        """
        Build 1d CNN RNN architecture
        """

        ## import ML librairies
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import BatchNormalization
        from tensorflow.keras.layers import Conv1D
        from tensorflow.keras.layers import GRU
        from tensorflow.keras.layers import MaxPooling1D, AveragePooling1D
        from tensorflow.keras.layers import Activation
        from tensorflow.keras.layers import Dropout
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import Flatten
        from tensorflow.keras.layers import Input
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import TimeDistributed
        from tensorflow.keras.layers import concatenate
        from tensorflow.keras.optimizers import Adam, RMSprop
        from tensorflow.keras.layers.experimental import preprocessing
        
        print('loss_metric:', loss_metric)
        print('metrics:', metrics)
        
        ## define two sets of inputs
        inputA = Input(shape=self.input_wind_shape)
        inputB = Input(shape=self.input_source_shape)
        
        ## Data preprocessing
        print('new norm')
        normalizer_winds = preprocessing.Normalization(axis=-1)
        normalizer_winds.adapt(self.train_input_wind)
        normalizer_f0s = preprocessing.Normalization()
        normalizer_f0s.adapt(self.train_input_f0)
        
        ## the first branch operates on the first input
        x = normalizer_winds(inputA)
        #x = tensorflow.transpose(normalizer_winds(inputA),perm = [0,2,1,3])
        #bp()
        for size, activation in CNN_layers:
            x = TimeDistributed(Conv1D(size[0], size[1], activation=activation))(x)
            x = BatchNormalization()(x)
            #x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
            x = TimeDistributed(AveragePooling1D(pool_size=2))(x)
        x = Flatten()(x)
        
        
        ## Second branch for f0 -> only normalization
        y = normalizer_f0s(inputB)

        ## combine the output of the two branches
        combined = concatenate([x, y])
        
        ## apply a FC layer and then a regression prediction on the combined outputs
        z = combined
        for size, activation in dense_layers_all:
            z = Dense(size, activation=activation)(z)
            if dropout_factor > 0.:
                z = Dropout(dropout_factor)(z)
            
        ## Output layer
        z = Dense(self.output_shape, activation=activation_result)(z)

        ## our model will accept the inputs of the two branches and then output a TL1d map
        self.model = Model(inputs=[inputA, inputB], outputs=z)

        ## Optimizer
        opt = Adam(lr=learning_rate)#, decay=decay)
        self.model.compile(loss=loss_metric, optimizer=opt, metrics=metrics)
        
    def train_model(self, epochs=100, batch_size=10, verbose=2, factor=0.5, min_lr=1e-8, cooldown=3, patience=10):
    
        """
        Fit model to training data
        """

        ## Save training history
        self.history = None
        
        ## Neural networks
        if self.type_model in ['FCN', 'CNN', 'transfer', '1dCNN']:
            from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
            
            ## Fittin and stopping conditions
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=factor, min_lr=min_lr, cooldown=cooldown)
            earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=patience, restore_best_weights=True)
            self.history = self.model.fit(
                x = [self.train_input_wind, self.train_input_f0], 
                y = self.train_output,
                validation_data = ([self.test_input_wind, self.test_input_f0], self.test_output),
                callbacks = [earlystop, reduce_lr],
                epochs = epochs,
                batch_size = batch_size,        
                verbose = verbose
            )
            
        ## Random forests
        elif self.type_model == 'RF':
            data_input = self.train_input_wind.reshape((self.train_input_wind.shape[0], self.train_input_wind.shape[1]*self.train_input_wind.shape[2]))
            data_input = np.concatenate((data_input, self.train_input_f0[..., None]), axis=1)
            self.model.fit(data_input, self.train_output)
        
        else:
            sys.exit('ML model not recognized')
        
    def save_model(self, model_dir):
    
        """
        Dump KERAS model and its history
        """
        
        ## Get dataset metadata
        if self.dataset_metadata.shape[0] > 0:
            self.dataset_metadata.to_csv(model_dir + 'dataset_metadata.csv', header=True, index=False)
        
        ## Save Keras model
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        self.model.save(model_dir)
        np.save(model_dir + 'model_history.npy', self.model.history.history)
        self.all_accuracies.to_csv(model_dir + 'all_accuracies.csv', header=True, index=False)
        
        ## Save indices used for validation
        np.savetxt(model_dir + 'test_ids.csv', self.test_ids, newline=" ")
        #self.mse_errs.to_csv(output_dir + 'uncertainty.csv', header=False, index=False)
        
        ## Save performance metrics
        self.mse_errs.to_csv(model_dir + 'mse_errs.csv', header=True, index=False)
        if self.uncertainty.size > 0:
            np.savetxt(model_dir + 'uncertainty.csv', self.uncertainty, newline=" ")
        np.savetxt(model_dir + 'type_model.txt', [self.type_model], fmt='%s')
        
    def load_model(self, model_dir):
    
        """
        Load KERAS model and its history
        """
        
        from tensorflow.keras.models import load_model
        self.model = load_model(model_dir)
        history = np.load(model_dir + 'model_history.npy', allow_pickle='TRUE').item()
        self.history = history_class(history)
        
        #self.uncertainty = pd.read_csv(model_dir + 'uncertainty.csv', header=None, delim_whitespace=True)
        self.type_model = str(np.loadtxt(model_dir + 'type_model.txt', dtype=str))
        try:
            self.mse_errs = pd.read_csv(model_dir + 'mse_errs.csv', header=[0], sep=',')
            self.uncertainty = np.loadtxt(model_dir + 'uncertainty.csv')
            
        except:
            pass
            
        try:
            self.test_ids = np.loadtxt(model_dir + 'test_ids.csv')
            self.dataset_metadata = pd.read_csv(model_dir + 'dataset_metadata.csv', header=[0], sep=',')
            nb_elements_dataset = self.dataset_metadata.iloc[0].TL_stack_shape[0]
            self.train_ids = np.setdiff1d(np.arange(nb_elements_dataset), self.test_ids)
            
        except:
            pass
        
        try:
            self.all_accuracies = pd.read_csv(model_dir + 'all_accuracies.csv', header=[0])
            if isinstance(self.all_accuracies.val_loss_accuracy.iloc[0], str):
                print('Conversion val accuracy')
                
                for ii, one_acc in self.all_accuracies.iterrows():
                    #print( one_acc.test_ids[1:-1].split())
                    test_ids = np.array([int(val) for val in one_acc.test_ids[1:-1].split()])
                    
                    self.all_accuracies.loc[self.all_accuracies.index == ii, 'test_ids'] = test_ids
                    val_loss_accuracy = np.array([float(val) for val in one_acc.val_loss_accuracy[1:-1].split(',')])
                    self.all_accuracies.loc[self.all_accuracies.index == ii, 'val_loss_accuracy'] = val_loss_accuracy
                    loss_accuracy = np.array([float(val) for val in one_acc.loss_accuracy[1:-1].split(',')])
                    self.all_accuracies.loc[self.all_accuracies.index == ii, 'loss_accuracy'] = loss_accuracy
                    
                
        except:
            pass
            
    def get_uncertainty(self, mov_mean=20, nb_ranges=5, nb_frequencies=5, list_quantiles=[0.05, 0.25, 0.5, 0.75, 0.95], model_dir = ''):
    
        """
        Compute range-dependent uncertainty
        """
        
        """
        mov_mean=20
        nb_ranges=5
        nb_frequencies=5
        list_quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
        ids_test = model.test_ids
        all_errors = model.dataset.properties.loc[model.dataset.properties.simu_id.isin(ids_test)]
        ids = all_errors.simu_id.unique(); 
        nb_simulations = len(ids)
        model.mse_errs = utils_figures.compute_TL_error_with_ids(model, mov_mean, nb_ranges, ids, get_range_uncertainty=True);
        model.mse_errs.to_csv('/staff/quentin/Documents/Projects/ML_attenuation_prediction/models/model_b32_d20_u4_t0.15_scaling_and_norm//mse_errs.csv', header=True, index=False)
        reload(utils_figures) 
        uncertainties, uncertainty_freqs = utils_figures.determine_uncertainty_from_errs(model.mse_errs, list_quantiles, nb_frequencies)
        """
        
        ## Find simulations from testing dataset
        #ids_test = self.test_ids; all_errors = self.dataset.properties.loc[self.dataset.properties.simu_id.isin(ids_test)]
        #ids_test = self.test_ids; 
        #all_errors = self.dataset.properties.loc[self.dataset.properties.simu_id.isin(ids_test)]
        #ids = all_errors.simu_id.unique(); 
        #nb_simulations = len(ids)
        
        ## Compute relative error and determine uncertainty
        #file = model_dir + 'mse_errs.csv'
        self.mse_errs = utils_figures.compute_TL_error_with_ids(self, mov_mean, nb_ranges, self.test_ids, get_range_uncertainty=True);
        #self.mse_errs = pd.read_csv(file, header=[0])
   
        ## Determine uncerainty
        self.uncertainty, self.uncertainty_freqs = utils_figures.determine_uncertainty_from_errs(self.mse_errs, list_quantiles, nb_frequencies)
        
    def predict(self, input_wind, input_f0, use_encoded_inputs=False):
    
        """
        Encode a wind model and give a decoded 1d TL profile
        Input rows should correspond to different wind slices
        """
        
        ## If input already encoded or not
        if use_encoded_inputs:
            encoded_inputs = input_wind
        else:
            encoded_inputs = self.dataset.SVD_input.transform(input_wind)
            args = {'coef_downsample': self.dataset.coef_downsample, 'coef_upsample': self.dataset.coef_upsample, 'n_components_input': self.dataset.n_components, 'method': self.dataset.method}
            encoded_inputs = self.dataset.encode_inputs(input_wind=input_wind, **args)
        #encoded_outputs = self.model.predict([encoded_input, input_f0])
        #recovered_output = self.SVD_output.inverse_transform(encoded_outputs)
        
        ## Transform normalized ML output to physical db unit
        encoded_outputs = self.model.predict([encoded_input, input_f0])
        if not self.dataset.method_output_name == 'linear':
            recovered_output_pred = self.dataset.SVD_output.inverse_transform(encoded_outputs)
        else:
            recovered_output_pred = encoded_outputs
        recovered_output_pred = self.dataset.scaler_TL.inverse_transform(recovered_output_pred)
        
        return recovered_output_pred

def get_args_from_opt_param(opt_par, type_model='FCN', kernel_size=3):

    """
    Convert a list of arguments found through optimization to arguments to build a NN instance 
    """

    ## Standard parameters
    args_NN = {}
    id = 0
    args_NN['learning_rate'] = opt_par[id]
    id += 1
    num_dense_layers = opt_par[id]
    id += 1
    num_dense_nodes = opt_par[id]
    id += 1
    factor_num_dense_layers = opt_par[id]
    id += 1
    activation = opt_par[id]
    
    ## Parameters related to wind encoding using FCN
    if type_model == 'FCN':
        
        id += 1
        num_dense_layers_winds = opt_par[id]
        id += 1
        num_dense_nodes_winds = opt_par[id]
        id += 1
        factor_num_dense_layers_winds = opt_par[id]
    
        vect_num_dense_nodes_winds = [int(num_dense_nodes_winds*(factor_num_dense_layers_winds**ilayer)) for ilayer in range(num_dense_layers_winds)]
        args_NN['dense_layers_winds'] = [(vect_num_dense_nodes_winds[ilayer], activation) for ilayer in range(num_dense_layers_winds) if vect_num_dense_nodes_winds[ilayer] > 0]
        
    ## Parameters related to wind encoding using CNN
    elif type_model in ['1dCNN', 'CNN']:
    
        id += 1
        num_CNN_layers = opt_par[id]
        id += 1
        min_filters_power2 = opt_par[id]
        id += 1
        factor_num_CNN_layers = opt_par[id]
    
        ## TODO: FIX THIS
        vect_num_CNN_layers = [int((2**min_filters_power2)*(factor_num_CNN_layers**ilayer)) for ilayer in range(num_CNN_layers)]
        if type_model == '1dCNN':
            args_NN['CNN_layers'] = [((vect_num_CNN_layers[ilayer], kernel_size), activation) for ilayer in range(num_CNN_layers) if vect_num_CNN_layers[ilayer] > 0]
        
        else:
            args_NN['CNN_layers'] = [((vect_num_CNN_layers[ilayer], kernel_size, kernel_size), activation) for ilayer in range(num_CNN_layers) if vect_num_CNN_layers[ilayer] > 0]
        #args_NN['CNN_layers'] = [((2**(ilayer+2), 5, 5), activation) for ilayer in range(num_CNN_layers)]
        
    vect_num_dense_nodes = [int(num_dense_nodes*(factor_num_dense_layers**ilayer)) for ilayer in range(num_dense_layers)]
    args_NN['dense_layers_all'] = [(vect_num_dense_nodes[ilayer], activation) for ilayer in range(num_dense_layers) if vect_num_dense_nodes[ilayer] > 0]
    
    return args_NN

def get_dataset(output_dir, construct_new_dataset=False, 
                coef_downsample=20, coef_upsample=4, type_model='CNN', ext='', 
                n_components_input=10, n_components_output=500, r_normalization=5,
                method_output='linear', method_input='downsample_1channel',
                method='tloss', nb_CPU=20, range_distance=[2., 999.], use_ceff_inputs=False,
                alt_veffs=[[5., 20.], [20., 50.], [40., 50.], [50., 100.]], ranges=np.linspace(0., 1000, 10),
                file_name = 'dataset_{model}_{input}_{output}_{ncomp}_{n_components_output}{ext}.pkl',
                name_file='list_simulations.csv'):

    """
    Load or build new TL dataset
    """

    ## Name of dataset
    ncomp = str(coef_downsample)+'_'+str(coef_upsample) if type_model in ['CNN', 'transfer'] else str(n_components_input)
    #file_name = 'dataset_{model}_{input}_{output}_{ncomp}_{n_components_output}_d{coef_downsample}_u{coef_upsample}.pkl'
    #file_name = 'dataset_{model}_{input}_{output}_{ncomp}_{n_components_output}{ext}.pkl'
    
    ## Build/load input/output dataset
    if construct_new_dataset:
        one_dataset = build_dataset(output_dir, method=method, nb_CPU=nb_CPU, nb_pts_TL_interp=n_components_output, \
                                    range_distance=range_distance, max_profiles=n_components_input, use_ceff_inputs=use_ceff_inputs,
                                    name_file=name_file, r_normalization=r_normalization)
        one_dataset.determine_dataset_properties(alt_veffs=alt_veffs, ranges=ranges)
        one_dataset.encode_inputs(n_components_input=n_components_input, method=method_input, coef_downsample=coef_downsample, coef_upsample=coef_upsample)
        if one_dataset.contains_output:
            one_dataset.encode_outputs(n_components_output, method=method_output)
        
        file_to_store = open(output_dir + file_name.format(model=type_model, input=method_input, output=method_output, ncomp=ncomp, n_components_output=n_components_output, coef_downsample=coef_downsample, coef_upsample=coef_upsample, ext=ext), 'wb')
        pickle.dump(one_dataset, file_to_store)
    
    else:
        try:
            file = file_name.format(model=type_model, input=method_input, output=method_output, ncomp=ncomp, n_components_output=n_components_output, ext=ext)
            file_to_load = open(output_dir + file, 'rb')
            
        except:
            file = file_name.format(model=type_model, input=method_input, output=method_output, ncomp=ncomp, n_components_output=n_components_output, coef_downsample=coef_downsample, coef_upsample=coef_upsample, ext=ext)
            file_to_load = open(output_dir + file, 'rb')
        
        print('#######################')
        print('#######################')
        print('Load dataset {file}'.format(file=file))
        print('#######################')
        print('#######################')
        one_dataset = pickle.load(file_to_load)
    
    return one_dataset

def load_or_train_ML_model(one_dataset, output_dir, model_dir='', save_model=True, full_random=False, 
                           coef_downsample=20, coef_upsample=4, type_model='CNN', test_size=0.15, type_split='nb_slice_orig', scaler_output='standard',
                           optimization=False, opt_par = [1e-3, 2, 50, 2., 'relu', 3, 5, 2.], seed_split=2, seed_tf=1, kfold_sets=[],
                           batch_size=32, verbose=2, factor=0.5, min_lr=1e-8, cooldown=3, patience=12,
                           loss_metric='mse', metrics=['mse'], activation_result=None, dropout_factor=0., epochs=120, kernel_size=5, ext='',
                           name_model='model_b{batch_size}_d{coef_downsample}_u{coef_upsample}_t{test_size}_seeds{npseed}.{tfseed}{ext}/'):

    """
    Load tensorflow model or train new model
    """
    
    ## Load model directory 
    if model_dir:
        model = build_NN(output_dir, one_dataset)
        model.load_model(model_dir)
        
    ## Train keras model
    else:
        
        ## Initialize ML model and split dataset for training
        model = build_NN(output_dir, one_dataset)
        print('after init')
        model.split_train_test(seed=seed_split, test_size=test_size, full_random=full_random, type=type_split, scaler_output=scaler_output)
        print('after split')
        
        ## Find optimal network parameters
        #opt_par = [0.003415614830416963, 3, 40, 1., 'relu', 7, 37, 1.] # SVD
        #opt_par = [0.00041791928002602813, 5, 40, 2.0, 'relu', 1, 5, 2.0] # PCA 1
        #opt_par = [0.0007138800045292168, 5, 40, 2.0, 'relu', 5, 5, 1.1187164278757962] # PCA 2
        #opt_par = [1e-4, 2, 40, 1., 'relu', -1, -1- -1] # transfer
        #opt_par = [[0.001, 2, 50, 2.0, 'relu', 3, 5, 2.0]] # CNN predicting raw TL
        #opt_par = [1e-4, 2, 50, 2., 'relu', 3, 5, 2.] # CNN 0.3 acc
        #opt_par = [1e-3, 2, 50, 2., 'relu', 3, 5, 2.] # CNN test
        if optimization:
            n_calls = 100
            opt_par = optimization_machine_learning.optimize(model, type_model, activation_result=activation_result, loss_metric=loss_metric, 
                     epochs=20, batch_size=64, acq_func='EI', n_calls=n_calls, default_parameters=opt_par,
                     range_learning_rate=[1e-6, 1e-3], range_num_dense_layers=[1,5], range_num_dense_layers_winds=[1,5], 
                     range_factor_num_dense_layers=[0.5, 2], range_factor_num_dense_layers_winds=[0.5, 2],
                     range_num_dense_nodes_winds=[10, 50], range_num_dense_nodes=[10, 50], 
                     range_num_CNN_layers=[2, 3], range_min_filters_power2=[3, 5], range_factor_num_CNN_layers=[0.5, 2],
                     range_activation=['relu', 'sigmoid'])

        print('----------------------')
        print('Optimal parameters:', opt_par)
        print('----------------------')
        
        ## Build neural network from optimal parameters
        if type_model == 'FCN': 
            model = build_NN(output_dir, one_dataset)
            model.split_train_test(seed=seed_split, test_size=test_size, full_random=full_random, type=type_split)
            args_NN = get_args_from_opt_param(opt_par, type_model, kernel_size)
            model.build_model(type_model=type_model, doubleFCN=True, **args_NN,
                             activation_result=activation_result,
                             loss_metric=loss_metric, metrics=metrics, dropout_factor=dropout_factor)
            model.train_model(epochs=epochs, batch_size=batch_size, verbose=verbose)
        
        elif type_model in ['1dCNN', 'CNN']:
        
            root_dir = '/staff/quentin/Documents/Projects/ML_attenuation_prediction/models/'
            sensitivity = False
            if sensitivity:
            
                coef_downsample, coef_upsample = 20, 4
                test_size = 0.15
                batch_size = 32
                
                batch_sizes = [256, 128, 64, 32, 16]
                test_sizes = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
                #l_dusamples = [(10, 4), (10, 2), (5, 4), (5, 2)]
                #l_dusamples = [(10, 2), (5, 4), (5, 2)]
                #l_dusamples = [(5, 4), (5, 2), (10, 4), (10, 2), (20, 2)]
                
                for batch_size in batch_sizes:
                #for test_size in test_sizes:
                #for coef_downsample, coef_upsample in l_dusamples:
                
                    """
                    one_dataset.encoded_input = one_dataset.wind_stack[:]
                    one_dataset.encode_inputs(n_components=n_components_input, method=method_input, coef_downsample=coef_downsample, coef_upsample=coef_upsample)
                    file_to_store = open(output_dir + file_name.format(model=type_model, input=method_input, output=method_output, ncomp=ncomp, n_components_output=n_components_output, coef_downsample=coef_downsample, coef_upsample=coef_upsample, ext=''), 'wb')
                    pickle.dump(one_dataset, file_to_store)
                    file_to_store.close()
                    """
                    
                    ## Generate new NN model
                    model = build_NN(output_dir, one_dataset)
                    model.split_train_test(seed=seed_split, test_size=test_size, full_random=full_random)
                    args_NN = get_args_from_opt_param(opt_par, type_model, kernel_size)
                    print(args_NN)
                    model.build_model(type_model=type_model, **args_NN,
                                     activation_result=activation_result,
                                     loss_metric=loss_metric, metrics=metrics, dropout_factor=dropout_factor)
                    model.train_model(epochs=epochs, batch_size=batch_size, verbose=verbose, factor=factor, min_lr=min_lr, cooldown=cooldown, patience=patience)
                    
                    if save_model:
                        model.save_model(root_dir + name_model.format(batch_size=batch_size, coef_downsample=coef_downsample, coef_upsample=coef_upsample, test_size=test_size, ext=ext, npseed=seed_split, tfseed=seed_tf))
                    
            else:
                
                args_NN = get_args_from_opt_param(opt_par, type_model, kernel_size)
                print(args_NN)
                seed_splits_loc = [seed_split] + kfold_sets
                
                ## Build list of index with fixed number of Kfolds
                idx_test_Kfold = []
                if len(seed_splits_loc) > 1:
                    orig_slice = one_dataset.properties.orig_slice.unique()
                    from sklearn.model_selection import KFold
                    kf = KFold(n_splits=int(orig_slice.size//np.ceil(orig_slice.size*test_size)), shuffle=True, random_state=seed_split)
                    idx_test_Kfold = [test_index for _, test_index in kf.split(orig_slice)]
                    
                all_accuracies = pd.DataFrame()
                best_accuracy = 1e10
                for iseed, seed_split_loc in enumerate(seed_splits_loc):
                
                    ## TODO: to remove
                    #seed_split_loc = 2
                    
                    ## If Kfold index already built, we select them instead of randomizing later on
                    idx_test_Kfold_loc = np.array([])
                    if idx_test_Kfold:
                        idx_test_Kfold_loc = idx_test_Kfold[iseed]
                    
                    ## Build and train model for this train/test split
                    model = build_NN(output_dir, one_dataset)
                    ## In the paper, seed_split_loc=2 hardcoded here
                    """
                    test_size=0.15
                    seed_split=3
                    orig_slice = one_dataset.properties.orig_slice.unique()
                    from sklearn.model_selection import KFold
                    kf = KFold(n_splits=int(orig_slice.size//np.ceil(orig_slice.size*test_size)), shuffle=True, random_state=seed_split)
                    idx_test_Kfold = [test_index for _, test_index in kf.split(orig_slice)]
                    iseed = 2
                    idx_test_Kfold_loc = idx_test_Kfold[iseed]
                    model2.split_train_test(seed=2, test_size=0.15, full_random=False, type='nb_slice_orig', orig_idx_test=idx_test_Kfold_loc)
                    """
                    model.split_train_test(seed=seed_split_loc, test_size=test_size, full_random=full_random, type=type_split, orig_idx_test=idx_test_Kfold_loc)
                    model.build_model(type_model=type_model, **args_NN,
                                     activation_result=activation_result,
                                     loss_metric=loss_metric, metrics=metrics, dropout_factor=dropout_factor)
                    model.train_model(epochs=epochs, batch_size=batch_size, verbose=verbose, factor=factor, min_lr=min_lr, cooldown=cooldown, patience=patience)
                    
                    val_accuracy = model.model.history.history['val_loss']
                    accuracy = model.model.history.history['loss']
                    loc_dict = {'seed_split': seed_split_loc, 'test_ids': model.test_ids, 'loss_accuracy': accuracy, 'val_loss_accuracy': val_accuracy}
                    all_accuracies = all_accuracies.append( [loc_dict] )
                    if best_accuracy > val_accuracy[-1]:
                        best_accuracy = val_accuracy[-1]
                        best_model = model
               
                ## Only save the best model
                model = best_model
                model.all_accuracies = all_accuracies.reset_index(drop=True)
                if save_model:
                    model.save_model(root_dir + name_model.format(batch_size=batch_size, coef_downsample=coef_downsample, coef_upsample=coef_upsample, test_size=test_size, ext=ext, npseed=seed_split_loc, tfseed=seed_tf))
                    
        elif type_model == 'transfer':
            model = build_NN(output_dir, one_dataset)
            args_NN = get_args_from_opt_param(opt_par, type_model, kernel_size)
            model.build_model(type_model=type_model, **args_NN,
                             activation_result=activation_result,
                             loss_metric=loss_metric, metrics=metrics, dropout_factor=dropout_factor)
            model.train_model(epochs=epochs, batch_size=batch_size, verbose=verbose)
            if save_model:
                model.save_model(root_dir + 'transfer_learning')
        
        ## Build RF
        elif type_model == 'RF':
            model.build_model(type_model=type_model, nb_trees=500, seed=1, oob_score=True, bootstrap=True, 
                              max_depth=100)
            model.train_model()
            
    return model

def get_sub_dict_parameters(list_idx, parameters):
    
    """
    Return subdictionnary of main dictionnary "parameters" composed of entries listed in "idx"
    """
    
    sub_parameters = {}
    for idx in list_idx:
        sub_parameters[idx] = parameters[idx]
        
    return sub_parameters