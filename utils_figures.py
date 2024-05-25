#!/usr/bin/env python3
from importlib import reload 
import numpy as np
from pdb import set_trace as bp
import pandas as pd
import os 
from obspy.core.utcdatetime import UTCDateTime

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from scipy.interpolate import griddata
import string 
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy import interpolate
from matplotlib.colors import ListedColormap

from multiprocessing import get_context
from functools import partial

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import build_machine_learning
from skimage.transform import resize
import time

def determine_dataset_properties(model, output_dir, alt_veffs=[[0., 20.], [20., 50.], [50., 100.]], ranges=np.linspace(0., 1000, 10), use_MSIS=True):
        
    """
    Compute range dependent Veff ratio for a given altiude
    """
    
    ## Determine sound speed for veff computation
    if not use_MSIS:
        c_ground = np.sqrt(401.87430086589046*fluids.atmosphere.ATMOSPHERE_1976(0.).T)
            
    ## Wind lateral stds
    lateral_std_winds = model.wind_stack.std(axis=2)
    
    ## Determine veff_ratio at various altitudes
    model.properties = pd.DataFrame()
    for ialt_veff, alt_veff_bounds in enumerate(alt_veffs):
        
        str_alt = str(int(alt_veff_bounds[0])) + '-' + str(int(alt_veff_bounds[1]))
        ialt     = np.argmin(abs(model.altitudes - alt_veff_bounds[0]))
        ialt_end = np.argmin(abs(model.altitudes - alt_veff_bounds[1]))
        print(alt_veff_bounds, ialt, ialt_end)
        data_shape = model.wind_stack.shape
        
        nb_alts = ialt_end - ialt + 1
        if use_MSIS:
            c_alt = np.zeros((model.slice_nos.size, nb_alts))
            for ii, islice in enumerate(model.slice_nos):
                c_file = '{output_dir}slice_no_{islice:0.0f}/slice_no_{islice:0.0f}_velocity.csv'.format(islice=islice)
                c_profile = pd.read_csv(c_file, header=[0])
                c_alt[ii,:] = c_profile['c'].values[ialt:ialt_end+1]
        
        bp()
        
        std = 0.
        c_alt = np.zeros((nb_alts,))
        for idx, i in enumerate(range(ialt, ialt_end+1)):
            ## Sound velocity at a given altitude
            #c_alt[idx] = fluids.atmosphere.ATMOSPHERE_1976(self.altitudes[i]*1e3).v_sonic
            c_alt[idx] = np.sqrt(401.87430086589046*fluids.atmosphere.ATMOSPHERE_1976(model.altitudes[i]*1e3).T)
        c_alt = np.repeat([c_alt], ranges.size, axis=0).T
        
        ## Compute veff ratio
        veff_ratio = np.max(model.wind_stack[:, ialt:ialt_end+1, :] + c_alt, axis=1) / (model.wind_stack[:, 0, :] + c_ground)
            
        ## Compute lateral std
        std = lateral_std_winds[:, ialt:ialt_end+1].max(axis=1)
        
        ## Convert 2d array into dataframe
        veff_ratio = pd.DataFrame(data=veff_ratio)
        veff_ratio.columns = ranges.tolist()
        veff_ratio = veff_ratio.unstack().reset_index()
        veff_ratio.columns = ['ranges', 'simu_id', 'veff-'+str_alt]
        if ialt_veff == 0:
            model.properties = veff_ratio.copy()
        else:
            model.properties['veff-'+str_alt] = veff_ratio['veff-'+str_alt]
        
        model.properties['std-'+str_alt] = np.repeat(std, ranges.size)
        
    model.properties['f0s'] = np.tile(model.f0s, ranges.size)
    model.properties['orig_slice'] = np.tile(model.list_orig_slice, ranges.size)

def find_simulations_without_upwind(wind_stack, threshold, range_index):

    """
    Attempt to find simulations with similar wind profiles (upwind or downwind and Gardner)
    """

    links = pd.DataFrame()
    N_simulations = wind_stack.shape[0]
    for ii in range_index:
        print('Simulation {ii}'.format(ii=ii))
        diff = wind_stack[ii+1:] - wind_stack[ii,:]
        diff_inv = wind_stack[ii+1:] + wind_stack[ii,:]
        diff_mean = abs(diff).mean(axis=2).mean(axis=1)
        diff_mean_inv = abs(diff_inv).mean(axis=2).mean(axis=1)
        idx_diff = np.where(abs(diff_mean <= threshold))[0]
        idx_diff_inv = np.where(abs(diff_mean_inv <= threshold))[0]
        
        if idx_diff.size > 0:
            diff_mean = diff_mean[idx_diff]
            loc_diff = pd.DataFrame()
            loc_diff['score'] = diff_mean
            loc_diff['type'] = 'downwind'
            loc_diff['model_A'] = ii
            loc_diff['model_B'] = np.arange(ii+1, N_simulations)[idx_diff]
            links = links.append( loc_diff )
        
        if idx_diff_inv.size > 0:
            diff_mean_inv = diff_mean_inv[idx_diff_inv]
            loc_diff = pd.DataFrame()
            loc_diff['score'] = diff_mean_inv
            loc_diff['type'] = 'upwind'
            loc_diff['model_A'] = ii
            loc_diff['model_B'] = np.arange(ii+1, N_simulations)[idx_diff_inv]
            links = links.append( loc_diff )
        
    links.reset_index(drop=True, inplace=True)
    return links
    
def find_simu_CPUs(one_dataset, threshold=1e-6, nb_CPU=15):

    N_simulations = one_dataset.wind_stack.shape[0]
    find_simulations_without_upwind_partial = \
        partial(find_simulations_without_upwind, one_dataset.wind_stack, threshold)

    N = min(nb_CPU, N_simulations)
    ## If one CPU requested, no need for deployment
    if N == 1:
        range_index = np.arange(0, N_simulations-1)
        links = find_simulations_without_upwind_partial(range_index)

    ## Otherwise, we pool the processes
    else:

        list_of_lists = []
        for i in range(N):
            range_index = np.arange(i, N_simulations-1, N)
            #if i == N-1:
            #    range_index = np.arange(i, N_simulations-1, N_simulations)
            list_of_lists.append( range_index )
        
        with get_context("spawn").Pool(processes = N) as p:
            results = p.map(find_simulations_without_upwind_partial, list_of_lists)

        links = pd.DataFrame()
        for result in results:
            links = links.append( result )

    links.reset_index(drop=True, inplace=True)
    
    return links
    
def create_list_simulation_radial(file, azs, source_loc, nb_rad=10, max_range=1000.,
                                  nb_stds = 4, range_gardner = [25., 18., 10., 5.]):

    """
    Create list of points
    """

    from pyproj import Geod
    wgs84_geod = Geod(ellps='WGS84')
    ev_coords = np.repeat(np.array([source_loc]), azs.size, axis=0)
    #ranges = np.repeat(np.array([range_total]), azs.size, axis=0)
    ranges = np.linspace(0., max_range, nb_rad)
    
    dummy_lat, dummy_lon = 0., 0.
    dummy_date = UTCDateTime.now()
    template_loc_simu = {
        'nb_slice': -1,
        'lat': dummy_lat,
        'lon': dummy_lon,
        'range': -1,
        'year': dummy_date.year,
        'month': dummy_date.month,
        'day': dummy_date.day,
        'hour': dummy_date.hour,
        'file': file,
        'nb_slice_orig': -1,
        'f0': -1,
        'az': 370.,
        'ceff_no': -1.,
        'std-0': 0.,
        'std-1': 0.,
        'std-2': 0.,
        'std-3': 0.,
        'summary_file': 'N/A',
        'flagged_zero_wind': False,
        'new_entry': False,
        'simulation_exists': False
    }
    
    low_range = 1.
    stds = []
    for istd in range(nb_stds): template_loc_simu['std-'+str(istd)] = np.random.uniform(low=low_range, high=range_gardner[istd], size=(1,))[0]
        
    list_simulations = pd.DataFrame()
    ev_coords = np.repeat(np.array([source_loc]), ranges.size, axis=0)
    for iaz, az in enumerate(azs):
    #for one_range in ranges:
        #ranges = np.repeat(np.array([one_range]), azs.size, axis=0)
        az_repeat = np.repeat(np.array([az]), ranges.size, axis=0)
        endlon, endlat, _ = wgs84_geod.fwd(ev_coords[:,1], ev_coords[:,0], az_repeat, ranges*1e3)
        for one_range, lon, lat in zip(ranges, endlon, endlat):
            new_entry = template_loc_simu.copy()
            new_entry['lat'] = lat
            new_entry['lon'] = lon
            new_entry['nb_slice'] = iaz
            new_entry['az'] = az
            new_entry['range'] = one_range
            new_entry['nb_slice_orig'] = iaz
            list_simulations = list_simulations.append( [new_entry] )

    list_simulations.reset_index(drop=True, inplace=True)
    return list_simulations
    

def ML_predict_radial_from_dataset(one_dataset, model, f0, output_dir, plot_error='RMSE', use_radial_plot=True, fontsize=12,
                                   vmin=-80., vmax=-30., vmin_TL=-120., vmax_TL=-10., error_min=2.5, error_max=20.,
                                   nb_lon=500, nb_lat=500, az_to_plot=83., mov_mean = 20):

    """
    Read atmospheric models from dataset, encode them, and perform ML-based TL predictions
    """
    
    nb_plots = 3
    
    ## Read simulation list
    """
    ints = ['nb_slice', 'year', 'month', 'day', 'hour']; dtype = {}; 
    for int_ in ints: dtype[int_] = float
    list_simulations = pd.read_csv(output_dir + '/list_simulations.csv', sep=',', header=[0], keep_default_na=False, dtype=dtype)
    for int_ in ints: list_simulations[int_] = list_simulations[int_].astype(int)
    """
    list_simulations = one_dataset.list_simulations.copy()
    list_simulations = list_simulations.groupby('nb_slice').first().reset_index()
    
    ## Predict TL
    dataset = one_dataset.encoded_input
    #dataset = model.dataset.encoded_input[simu_id:simu_id+1,:]
    #winds = model.dataset.wind_stack[simu_id]
    #f0=1.18
    f0s = np.repeat([f0], dataset.shape[0])
    recovered_output_pred = model.model.predict([dataset, f0s])
    recovered_output_pred = model.scaler_TL.inverse_transform(recovered_output_pred)
    #recovered_output_pred = model.model.predict([one_dataset.encoded_input, f0s])
    #recovered_output_pred = model.dataset.scaler_TL.inverse_transform(recovered_output_pred)
    distances = model.dataset.distances
    
    ## Compute RMSE errors if needed
    if plot_error == 'RMSE':
        TL_avg = []
        for idx_simu in range(one_dataset.TL_stack.shape[0]):
            TL_avg.append( get_avg_profile(one_dataset.TL_stack[idx_simu,:], distances, mov_mean) )
        TL_avg = np.array(TL_avg)
        RMSE = np.sqrt((1./recovered_output_pred.shape[1])*((recovered_output_pred - TL_avg)**2).sum(axis=1))
        RMSE_theta = np.radians(list_simulations.az.values)
    
    ## Retrieve point locations
    from pyproj import Geod
    wgs84_geod = Geod(ellps='WGS84')
    source_loc = list_simulations.iloc[0][['lat', 'lon']].values.astype('float')
    nb_points = recovered_output_pred.shape[0]*recovered_output_pred.shape[1]
    ev_coords = np.repeat(source_loc[None,:], nb_points, axis=0)
    ranges = np.tile(one_dataset.distances, nb_points//one_dataset.distances.shape[0])
    azs = np.repeat(list_simulations.az.values, nb_points//list_simulations.shape[0])
    grid_lon, grid_lat, _ = wgs84_geod.fwd(ev_coords[:,1], ev_coords[:,0], azs, ranges*1e3)
    grid_lon = grid_lon.reshape(recovered_output_pred.shape)
    grid_lat = grid_lat.reshape(recovered_output_pred.shape)
    
    ## Interpolate ML predictions
    import scipy.ndimage as ndimage
    lon_i = np.linspace(grid_lon.min(), grid_lon.max(), nb_lon)
    lat_i = np.linspace(grid_lat.min(), grid_lat.max(), nb_lat)
    grid_lon_i, grid_lat_i = np.meshgrid(lon_i, lat_i)
    points = np.c_[grid_lon.ravel(), grid_lat.ravel()]
    TL_i = griddata(points, recovered_output_pred.ravel(), (grid_lon_i, grid_lat_i), method='linear')
    TL_i = ndimage.gaussian_filter(TL_i, sigma=2, order=0)
    
    if use_radial_plot:
        nb_points_i = grid_lon_i.ravel().size
        ev_coords = np.repeat(source_loc[None,:], nb_points_i, axis=0)
        azs_reshape, _, ranges_reshape = wgs84_geod.inv(ev_coords[:,1], ev_coords[:,0], grid_lon_i.ravel(), grid_lat_i.ravel())
        azs_reshape = np.radians(azs_reshape.reshape(grid_lon_i.shape))
        ranges_reshape = ranges_reshape.reshape(grid_lon_i.shape)/1e3
        max_range = ranges_reshape.max()
    
    ## Interpolate PE predictions
    if one_dataset.TL_stack.size > 0:
        lon_i = np.linspace(grid_lon.min(), grid_lon.max(), nb_lon)
        lat_i = np.linspace(grid_lat.min(), grid_lat.max(), nb_lat)
        grid_lon_i, grid_lat_i = np.meshgrid(lon_i, lat_i)
        points = np.c_[grid_lon.ravel(), grid_lat.ravel()]
        TL_PE_i = griddata(points, one_dataset.TL_stack.ravel(), (grid_lon_i, grid_lat_i), method='linear')
        TL_PE_i = ndimage.gaussian_filter(TL_PE_i, sigma=2, order=0)
    
    else:
        nb_plots -= 1
    
    ## Interpolate effective velocities
    if not plot_error == 'RMSE':
        if plot_error == 'map':
            map_last = abs(TL_PE_i - TL_i) 
        else:
            ev_coords = np.repeat(source_loc[None,:], one_dataset.properties.shape[0], axis=0)
            one_dataset.properties['az'] = np.tile(list_simulations.az.values, one_dataset.properties.shape[0]//list_simulations.shape[0])
            grid_lon_veff, grid_lat_veff, _ = wgs84_geod.fwd(ev_coords[:,1], ev_coords[:,0], one_dataset.properties['az'].values, one_dataset.properties['ranges'].values*1e3)
            points = np.c_[grid_lon_veff, grid_lat_veff]
            map_last = griddata(points, one_dataset.properties['veff-40-50'].values, (grid_lon_i, grid_lat_i), method='linear')
    
    ## Plot maps
    fig = plt.figure(figsize=(7,4)); 
    grid = fig.add_gridspec(6, 3)
    subplot_kw = {}
    if use_radial_plot:
        subplot_kw = {'projection': 'polar'}
    axs = [fig.add_subplot(grid[:3, i], **subplot_kw) for i in range(2)]
    if plot_error == 'RMSE' or use_radial_plot:
        subplot_kw = {'projection': 'polar'}
        axs.append(fig.add_subplot(grid[:3, -1], **subplot_kw))
    else:
        axs.append(fig.add_subplot(grid[:3, -1]))
        
    ax_TL = fig.add_subplot(grid[4:, 1])
    simu = list_simulations.loc[abs(list_simulations.az-az_to_plot) == abs(list_simulations.az-az_to_plot).min()]
    az_found = simu.az.iloc[0]
    range_max = 960#one_dataset.distances.max()
    from pyproj import Geod
    wgs84_geod = Geod(ellps='WGS84')
    lon_pt, lat_pt, _ = wgs84_geod.fwd(source_loc[1], source_loc[0], az_found, range_max*1e3)
    
    #print(simu)
    idx_simu = simu.nb_slice.iloc[0]
    #TL_avg = one_dataset.TL_stack[idx_simu,:]
    TL_avg = get_avg_profile(one_dataset.TL_stack[idx_simu,:], distances, mov_mean)
    ax_TL.plot(one_dataset.distances, TL_avg, color='tab:red', alpha=0.4); 
    
    TL_1d = recovered_output_pred[idx_simu,:]
    """
    TL_1d = 10**(TL_1d/20.)
    f = interpolate.interp1d(one_dataset.distances, TL_1d, fill_value='extrapolate')
    TL_1d /= f(1)
    TL_1d = 20*np.log10(TL_1d)
    """
    ax_TL.plot(one_dataset.distances, TL_1d, color='tab:blue'); 
    ax_TL.set_xlabel('Range (km)', fontsize=fontsize-2.)
    ax_TL.set_ylabel('TL (db)', fontsize=fontsize-2.)
    
    ## Plot wind profiles
    ax_winds = fig.add_subplot(grid[3, 1])
    max_wind = one_dataset.wind_stack[idx_simu,:].max()
    delta_range = (one_dataset.distances[-1] - one_dataset.distances[0]) / one_dataset.wind_stack[idx_simu,:].shape[1]
    range_winds = np.arange(one_dataset.distances[0], one_dataset.distances[-1], delta_range)
    for irange, range_value in enumerate(range_winds):
        winds = one_dataset.wind_stack[idx_simu, :, irange] * (0.5 * delta_range * 0.9) / max_wind
        ax_winds.plot(range_value+delta_range/2. + winds, model.dataset.altitudes, color='tab:blue')
    ax_winds.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
    print('--> TL_i')
    
    bounds_v = {'vmin': vmin, 'vmax': vmax, 'shading': 'auto', 'cmap': 'inferno'}
    if use_radial_plot:
        axs[0].set_theta_zero_location("N")
        axs[0].set_theta_direction(-1)
        sc = axs[0].pcolormesh(azs_reshape, ranges_reshape, TL_i, **bounds_v);
        axs[0].set_rmax(1000.)
        axs[0].set_rticks([])  # Less radial ticks
        axs[0].set_thetagrids([], [], fontsize=12)
        #axs[0].set_rlabel_position(112)  # Move radial labels away from plotted line
        axs[0].grid(False)
    else:
        sc = axs[0].pcolormesh(grid_lon_i, grid_lat_i, TL_i, **bounds_v); 
        axs[0].set_xlabel('Longitude', fontsize=fontsize)
        axs[0].set_ylabel('Latitude', fontsize=fontsize)
    
    axins = inset_axes(axs[0], width="70%", height="3.5%", loc='lower left', bbox_to_anchor=(0.15, 1.03, 1, 1.), bbox_transform=axs[0].transAxes, borderpad=0)
    axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
    cbar = plt.colorbar(sc, cax=axins, orientation='horizontal', extend='both')
    cbar.ax.set_xlabel('TL (db)', labelpad=1., fontsize=fontsize-2.)
    cbar.ax.xaxis.tick_top()
    cbar.ax.xaxis.set_label_position('top')
    axs[0].set_title('ML', pad=45.)
    
    print('--> TL_PE_i')
    if one_dataset.TL_stack.size > 0:
        if use_radial_plot:
            axs[1].set_theta_zero_location("N")
            axs[1].set_theta_direction(-1)
            axs[1].pcolormesh(azs_reshape, ranges_reshape, TL_PE_i, **bounds_v);
            axs[1].set_rmax(range_max)
            axs[1].set_rticks([])  # Less radial ticks
            #thetatick_locs = [0., 90., 180., 270.]
            #thetatick_labels = ['N', 'E', 'S', 'W']
            axs[1].set_thetagrids([], [], fontsize=12)
            #axs[1].set_rlabel_position(112)  # Move radial labels away from plotted line
            axs[1].grid(False)
        else:
            axs[1].pcolormesh(grid_lon_i, grid_lat_i, TL_PE_i, **bounds_v);
        axs[1].set_title('PE', pad=45.)
    
    
    print('--> error')
    ## Either plot map with errors or winds or plot RMSE vs azimuth
    if not plot_error == 'RMSE':
        bounds_last = {}
        if plot_error:
            bounds_last = {'vmin': error_min, 'vmax': error_max}
        if use_radial_plot:
            axs[-1].set_theta_zero_location("N")
            axs[-1].set_theta_direction(-1)
            sc = axs[-1].pcolormesh(azs_reshape, ranges_reshape, map_last, shading='auto', **bounds_last);
            axs[-1].set_rmax(1000.)
            axs[-1].set_rticks([500., 1000.])  # Less radial ticks
            thetatick_locs = [0., 90., 180., 270.]
            thetatick_labels = ['N', 'E', 'S', 'W']
            axs[-1].set_thetagrids(thetatick_locs, thetatick_labels, fontsize=12)
            axs[-1].set_rlabel_position(112)  # Move radial labels away from plotted line
            axs[-1].grid(True)
        
        else:
            sc = axs[-1].pcolormesh(grid_lon_i, grid_lat_i, map_last, shading='auto', **bounds_last);
        axins = inset_axes(axs[-1], width="70%", height="3.5%", loc='lower left', bbox_to_anchor=(0.15, 1.03, 1, 1.), bbox_transform=axs[-1].transAxes, borderpad=0)
        axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
        cbar = plt.colorbar(sc, cax=axins, orientation='horizontal', extend='both')
        if plot_error:
            cbar.ax.set_xlabel('Error (db)', labelpad=1., fontsize=fontsize-2.)
            axs[-1].set_title('|ML-PE|', pad=45.)
        else:
            cbar.ax.set_xlabel('Stratosphere (40-50 km)', labelpad=1., fontsize=fontsize-2.)
            axs[-1].set_title('Effective velocity ratio', pad=45.)
        cbar.ax.xaxis.tick_top()
        cbar.ax.xaxis.set_label_position('top')
    else:
        axs[-1].set_theta_zero_location("N")
        axs[-1].set_theta_direction(-1)
        axs[-1].plot(RMSE_theta, RMSE)
        axs[-1].set_rmax(RMSE.max())
        axs[-1].set_rticks([5., 20., 40.])  # Less radial ticks
        thetatick_locs = [0., 90., 180., 270.]
        thetatick_labels = ['N', 'E', 'S', 'W']
        axs[-1].set_thetagrids(thetatick_locs, thetatick_labels, fontsize=12)
        axs[-1].set_rlabel_position(112)  # Move radial labels away from plotted line
        axs[-1].grid(True)
        axs[-1].set_title('RMSE (db)', pad=45.)
    
    if model.uncertainty.size > 0:
        uncertainty = model.uncertainty.loc[(model.uncertainty.fmin <= f0) 
                & (model.uncertainty.fmax >= f0)
                & (model.uncertainty.q == 0.5), 
                ~model.uncertainty.columns.isin(['fmin', 'fmax', 'q'])].values[:,0]
        lower_bound = recovered_output_pred.T[:,0] - uncertainty
        upper_bound = recovered_output_pred.T[:,0] + uncertainty
        ax_TL.fill_between(model.dataset.distances, lower_bound, upper_bound, alpha=0.3, color='tab:blue', zorder=1, label='uncertainty')
    
    ## Plot slice orientation shown below
    if not use_radial_plot:
        for ax in axs[:-1]:
            ax.plot([source_loc[1], lon_pt], [source_loc[0], lat_pt], linestyle='--', linewidth=2., color='tab:green')
    else:
        print(az_found)
        az_found = np.radians(az_found)
        print(az_found)
        for ax in axs[:-1]:
            ax.plot([az_found, az_found], [0., range_max], linestyle='--', linewidth=2., color='tab:green')
            ax.set_rmax(range_max)
            
    ## Remove frame around radial plot and show panel labels
    for iax, ax in enumerate(axs):
        alphabet = string.ascii_lowercase
        if not use_radial_plot and iax < 2 or (iax == 2 and not plot_error == 'RMSE'):
            ax.axis('off')
        ax.text(-0.075, 1.015, alphabet[iax]+')', ha='right', va='bottom', transform=ax.transAxes, 
                   bbox=dict(facecolor='w', edgecolor='w', pad=0.1, alpha=0.5), fontsize=15., fontweight='bold')
    
    ax_TL.text(-0.075, 1.015, alphabet[iax+1]+')', ha='right', va='bottom', transform=ax_TL.transAxes, 
                   bbox=dict(facecolor='w', edgecolor='w', pad=0.1, alpha=0.5), fontsize=15., fontweight='bold')
    ax_TL.set_ylim([vmin_TL, vmax_TL])
    
    fig.subplots_adjust(top=0.8, bottom=0.2)
    
    fig.set_rasterized(True)
    plt.savefig(output_dir + 'radial_comparison_az{az:d}.pdf'.format(az=az_to_plot))
    plt.close('all')

def convert_columns_to_rows(properties, column_name='veff', col_labels={}, remove_cols=['40-50']):

    df_to_plot = pd.DataFrame()
    for col in properties.columns:
        if not column_name in col:
            continue
        
        skip = False
        for col_to_remove in remove_cols:
            if col_to_remove in col:
                skip = True
        
        if skip:
            continue
        
        loc_df = pd.DataFrame()
        loc_df['value'] = properties[col]
        #print(column_name, loc_df)
        col_label = col
        alt_range = '-'.join(col.split('-')[-2:])
        if alt_range in col_labels:
            col_label = col_labels[alt_range]
        loc_df['layer'] = col_label
        df_to_plot = df_to_plot.append( loc_df )

    return df_to_plot
    
def read_slices(output_dir_existing):
        
    ## Recreate list_slices file from folder containing NETCDF files
    ints = ['nb_slice', 'year', 'month', 'day', 'hour']
    dtype = {}
    for int_ in ints:
        dtype[int_] = float
    list_slices_existing = pd.read_csv(output_dir_existing + 'list_slices.csv', sep=',', header=[0], dtype=dtype)
    for int_ in ints:
        list_slices_existing[int_] = list_slices_existing[int_].astype(int)

    return list_slices_existing

def move_legend(ax, new_loc, format_legend='{nb:.1g}', convert_to_float=True, **kws):

    old_legend = ax.legend_
    handles = old_legend.legendHandles
    if not convert_to_float:
        labels = [format_legend.format(nb=t.get_text()) for t in old_legend.get_texts()]
    
    else:
        labels = [format_legend.format(nb=float(t.get_text())) for t in old_legend.get_texts()]
    #title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, **kws)

def compute_probas_distrib(dataset, TL_vals=np.linspace(0., -200., 200), subsample=10, init_dist=10):
    
    idx = np.arange(dataset.f0s.shape[0])#[dataset.f0s>0.5]
    bins = np.arange(0,dataset.TL_stack.shape[1])[init_dist::subsample]
    from sklearn.neighbors import KernelDensity
    kdes = []
    for ikde in bins: kdes.append( KernelDensity(kernel='gaussian', bandwidth=0.2).fit(dataset.TL_stack[idx,ikde:ikde+1]) )
    scores = []
    for kde in kdes: scores.append( kde.score_samples(np.array([TL_vals]).T) )
    scores = np.array(scores)
    
    return dataset.distances[bins], TL_vals, scores.T

def plot_convergence(model, output_dir, ax=None):

    """
    Plot RMSE convergence vs epochs
    """

    Nepochs = len(model.history.history['val_mse'])
    epochs = np.arange(1, Nepochs+1)
    
    new_figure = False
    if ax == None:
        new_figure = True
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    ax.plot(epochs, np.sqrt(model.history.history['mse']), label='training', color='tab:blue')
    ax.plot(epochs, np.sqrt(model.history.history['val_mse']), label='validation - best fold', color='tab:orange')
    if model.all_accuracies.shape[0] > 1:
        for imodel, one_model in model.all_accuracies.iterrows():
            label_dict = {}
            if imodel == 0:
                label_dict = {'label': 'validation - other folds'}
            val_loss_accuracy = one_model.val_loss_accuracy
            if isinstance(one_model.val_loss_accuracy, str):
                val_loss_accuracy = np.array([float(val) for val in one_model.val_loss_accuracy[1:-1].split(',')])
            ax.plot(np.arange(len(val_loss_accuracy))+1, np.sqrt(val_loss_accuracy), color='tab:orange', alpha=0.3, **label_dict)
            
    #ax.grid()
    ax.set_ylabel('RMSE (db)')
    ax.legend(frameon=False)
    
    """
    axs[1].plot(epochs, model.history.history['lr'])
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Learning rate')
    axs[1].grid()
    
    fig.align_labels()
    fig.subplots_adjust(left=0.15)
    """
    
    if new_figure:
        file_name = 'convergence_{type_model}.pdf'
        plt.savefig(output_dir + file_name.format(type_model=model.type_model))

def determine_uncertainty_from_errs(mse_errs, list_quantiles, nb_frequencies):

    ## Compute relative error and determine uncertainty
    #self.mse_errs = utils_figures.compute_TL_error_with_ids(self, mov_mean, nb_ranges, ids, get_range_uncertainty=True);
    #self.mse_errs.to_csv('/staff/quentin/Documents/Projects/ML_attenuation_prediction/models/model_b32_d20_u4_t0.15/mse_errs.csv', header=True, index=False)
    #self.mse_errs = pd.read_csv('/staff/quentin/Documents/Projects/ML_attenuation_prediction/models/model_b32_d20_u4_t0.15/mse_errs.csv', header=[0])

    ## Frequency ranges
    if nb_frequencies > 1:
        uncertainty_freqs = np.linspace(mse_errs.f0.min(), mse_errs.f0.max(), nb_frequencies+1)
    else:
        uncertainty_freqs = [-1]
    
    ## Determine uncerainty
    uncertainties = pd.DataFrame()
    for q in list_quantiles:
        for ifreq, freq in enumerate(uncertainty_freqs[1:]):
            loc_mse_errs = mse_errs.loc[(mse_errs.f0 >= uncertainty_freqs[ifreq]) & (mse_errs.f0 < freq), ~mse_errs.columns.isin(['f0', 'simu_id'])]
            uncertainty = pd.DataFrame([abs(loc_mse_errs).quantile(q=q, axis=0).values])
            uncertainty['fmin'] = uncertainty_freqs[ifreq]
            uncertainty['fmax'] = freq
            uncertainty['q'] = q
            uncertainties = pd.concat([uncertainties, uncertainty])

    return uncertainties, uncertainty_freqs
    
def compare_TL(ids, model, output_dir, additional_f0s=[0.2, 3.], additional_homogeneous=[1], show_uncertainty=True,
               dataset='test', ylim=[-130., 0.], show_predictions=True, axs=None, legend=True, bbox_to_anchor_winds=(0.5, -1.3),
               label_predicted='ML', ext_name_file='', show_ceff=True, show_labels=True, show_LP12=False, show_PE=True):

    """
    Plot predicted and true TL profiles for a given id in the test dataset along with uncertainty from encoding
    """
    
    for id in ids:
    
        new_figure = False
        if axs == None:
            fig = plt.figure()
            grid = fig.add_gridspec(4, 1)
            ax_winds = fig.add_subplot(grid[0, :])
            ax_TL = fig.add_subplot(grid[1:, :])
            new_figure = True
            
        else:
            ax_winds = axs[0]
            ax_TL = axs[-1]
            
        cmap = sns.color_palette("rocket", n_colors=len(additional_f0s))

        ## Test over one profile
        simu_id, simu_slice_no, input_wind, input_f0, output = load_one_id(id, model, dataset)
            
        ceff = model.dataset.properties.loc[model.dataset.properties.simu_id==simu_id, 'veff-35-60'].values.mean()
        distances = model.dataset.distances
        altitudes = model.dataset.altitudes
        
        #encoded_inputs = model.dataset.SVD_input.transform(input_wind)
        ## Encoded inputs/outputs
        encoded_input = input_wind
        #time_start = time.time()
        encoded_outputs = model.model.predict([encoded_input, input_f0])
        if not model.dataset.method_output_name == 'linear':
            recovered_output_pred = model.dataset.SVD_output.inverse_transform(encoded_outputs)
        else:
            recovered_output_pred = encoded_outputs
        recovered_output_pred = model.scaler_TL.inverse_transform(recovered_output_pred)
        #time_end = time.time()
        #print('time one prediction', time_end-time_start)
        additional_outputs = []
        for f0 in additional_f0s:
            additional_outputs.append( model.model.predict([encoded_input, np.array([[f0]])]) )
            
        additional_outputs_homogeneous = []
        for id_profile in additional_homogeneous:
            encoded_input_temp = np.repeat(encoded_input[:,:,id_profile:id_profile+1,:], encoded_input.shape[2], axis=2)
            additional_outputs_homogeneous.append( model.model.predict([encoded_input_temp, input_f0]) )
            
        ## Decoded ML predicted output
        additional_recovered_output_pred = []
        for additional_output in additional_outputs:
            if not model.dataset.method_output_name == 'linear':
                temp_output = model.dataset.SVD_output.inverse_transform(additional_output)
            else:
                temp_output = additional_output
            temp_output = model.scaler_TL.inverse_transform(temp_output)
            additional_recovered_output_pred.append( temp_output )
        
        additional_outputs_homogeneous_pred = []
        for additional_output in additional_outputs_homogeneous:
            if not model.dataset.method_output_name == 'linear':
                temp_output = model.dataset.SVD_output.inverse_transform(additional_output)
            else:
                temp_output = additional_output
            temp_output = model.scaler_TL.inverse_transform(temp_output)
            additional_outputs_homogeneous_pred.append( temp_output )
        
        ## Decoded TL output
        if not model.dataset.method_output_name == 'linear':
            recovered_output = model.dataset.SVD_output.inverse_transform(output)
        else:
            recovered_output = model.dataset.TL_stack[simu_id:simu_id+1]
        recovered_output = model.scaler_TL.inverse_transform(recovered_output)
        true_output = model.dataset.TL_stack[simu_id]
        
        ## Decoded wind profiles
        if model.dataset.method_encoding in ['PCA', 'SVD']:
            recovered_input  = model.dataset.SVD_input.inverse_transform(input_wind.reshape(input_wind.shape[1:]))
            recovered_input  = model.dataset.scaler_winds.inverse_transform(recovered_input)
        else:
            surface_v = 0.
            if show_ceff:
                recovered_input = model.dataset.temp_stack[simu_id:simu_id+1,:]
                surface_v = recovered_input[0,0,0]
                recovered_input -= surface_v
                #print('surface_v', surface_v, recovered_input.min(), recovered_input.max())
                
            else:
                recovered_input = model.dataset.wind_stack[simu_id:simu_id+1,:]
            recovered_input = recovered_input.transpose(2,1,0)
            recovered_input = recovered_input[:,:,0]
            
        ## Compute predictions for one std over the standard input accounting for SVD/PCA encoding errors
        if model.dataset.method_encoding in ['PCA', 'SVD']:
            std = np.repeat(model.dataset.std_altitude_input[:,None], recovered_input.T.shape[1], axis=1)
            upper_std_wind_profiles = recovered_input.T + std/1.
            lower_std_wind_profiles = recovered_input.T - std/1.
            
            encoded_upper_std_wind = model.dataset.scaler_winds.transform(upper_std_wind_profiles.transpose(1,0))
            encoded_upper_std_wind = model.dataset.SVD_input.transform(encoded_upper_std_wind)
            
            pred_upper_std = model.model.predict([encoded_upper_std_wind[None,:], input_f0])
            if not model.dataset.method_output_name == 'linear':
                recovered_pred_upper_std = model.dataset.SVD_output.inverse_transform(pred_upper_std)
            else:
                recovered_pred_upper_std = pred_upper_std
            recovered_pred_upper_std = model.scaler_TL.inverse_transform(recovered_pred_upper_std)
            time_end = time.time()
            print('Prediction time: ', time_end-time_start)
            
            encoded_lower_std_wind = model.dataset.scaler_winds.transform(lower_std_wind_profiles.transpose(1,0))
            encoded_lower_std_wind = model.dataset.SVD_input.transform(encoded_lower_std_wind)
            
            pred_lower_std = model.model.predict([encoded_lower_std_wind[None,:], input_f0])
            if not model.dataset.method_output_name == 'linear':
                recovered_pred_lower_std = model.dataset.SVD_output.inverse_transform(pred_lower_std)
            else:
                recovered_pred_lower_std = pred_lower_std
            recovered_pred_lower_std = model.scaler_TL.inverse_transform(recovered_pred_lower_std)
            
        ## Plot wind profiles
        #print('-->', recovered_input.mean(axis=0).shape)
        #mean_vals = recovered_input.mean(axis=1)
        #recovered_input -= mean_vals
        max_wind = abs(recovered_input).max()
        #print('max_wind', max_wind)
        delta_range = (distances[-1] - distances[0]) / recovered_input.shape[0]
        range_winds = np.arange(distances[0], distances[-1], delta_range)
        #cmap = sns.color_palette("rocket", n_colors=len(range_winds))
        for irange, range_value in enumerate(range_winds):
            winds = (recovered_input.T[:, irange]) * (0.5 * delta_range) / max_wind
            #print('conv max_wind', winds.max(), delta_range)
            if model.dataset.method_encoding in ['PCA', 'SVD']:
                winds_upper = upper_std_wind_profiles[:, irange] * (0.5 * delta_range * 0.9) / max_wind
                winds_lower = lower_std_wind_profiles[:, irange] * (0.5 * delta_range * 0.9) / max_wind
            ax_winds.plot(range_value+delta_range/2.+ winds, model.dataset.altitudes, color='tab:blue')
            if model.dataset.method_encoding in ['PCA', 'SVD']:
                ax_winds.fill_betweenx(model.dataset.altitudes,range_value+delta_range/2. + winds_lower, range_value+delta_range/2. + winds_upper, facecolor='tab:blue', alpha=0.2)
            
            """
            label = '_nolegend_'
            if iprofile == 0:
                label = 'zonal'
            ax_winds.plot(range_value+delta_range/2.+u_winds, profile.z, color=cmap[5], label=label)
            label = '_nolegend_'
            if iprofile == 0:
                label = 'merid.'
            ax_winds.plot(range_value+delta_range/2.+v_winds, profile.z, color=cmap[-4], label=label)
            """
            
        ## Add scale bar
        size = winds.max()
        size_str = str(np.round(surface_v+size *max_wind / (0.5 * delta_range * 0.9), 1)) + ' m/s'
        fontprops = fm.FontProperties(size=12, family='monospace')
        asb = AnchoredSizeBar(ax_winds.transData, size, size_str, loc='lower left', pad=0.1, borderpad=0., sep=2.5, fontproperties=fontprops, frameon=False, bbox_to_anchor=bbox_to_anchor_winds, bbox_transform=ax_winds.transAxes)
        ax_TL.add_artist(asb)
        
        ## Plot TL profiles
        if show_PE:
            if model.dataset.method_encoding in ['PCA', 'SVD']:
                ax_TL.plot(model.dataset.distances, recovered_output.T, label='encoded true', color='tab:red', zorder=10); 
            ax_TL.plot(model.dataset.distances, true_output, label='PE', color='tab:red', alpha=0.4, zorder=10); 
        
        ## Plot LP12 (if needed)
        if show_LP12:
            print('LP12')
            ## Plot LP12
            list_freq_available = np.array([0.1, 0.2, 0.5, 1., 2.])
            freq = list_freq_available[np.argmin(abs(list_freq_available-input_f0[0]))]
            file = './data_alexis/ATTEN_FREQ_{freq:g}.asc'.format(freq=freq)
            ranges_LP12, ceff_LP12, attenuation_LP12 = load_alexis_data(file)
            idx_ceff = np.argmin(abs(ceff-ceff_LP12))
            print(ceff_LP12[idx_ceff], freq, input_f0[0])
            ax_TL.plot(ranges_LP12, attenuation_LP12.iloc[idx_ceff].values, color='tab:green', label='LP12')
        
        ## Plot predictions
        if show_predictions:
            ax_TL.plot(model.dataset.distances, recovered_output_pred.T, label=label_predicted, color='tab:blue', zorder=10); 
            for iTL, TL in enumerate(additional_recovered_output_pred):
                linestyle = '-'
                if iTL == len(additional_f0s)-1:
                    linestyle = '--'
                ax_TL.plot(model.dataset.distances, TL.T, color='grey', alpha=0.5, linestyle=linestyle, label=str(additional_f0s[iTL])+' Hz', zorder=5);
        
            for iTL, TL in enumerate(additional_outputs_homogeneous_pred):
                linestyle = '-'
                if iTL == len(additional_f0s)-1:
                    linestyle = '--'
                ax_TL.plot(model.dataset.distances, TL.T, color='green', alpha=0.5, linestyle=linestyle, label='profile ' + str(additional_homogeneous[iTL]), zorder=25);
            
            ## Plot uncertainty            
            if model.uncertainty.size > 0 and show_uncertainty:
                #input_f0=[2.]
                #uncertainty = model.uncertainty.loc[(model.uncertainty.fmin <= input_f0[0]) & (model.uncertainty.fmax > input_f0[0]) & (model.uncertainty.q == 0.5), ~model.uncertainty.columns.isin(['fmin', 'fmax', 'q'])].values
                uncertainty = model.uncertainty.loc[(model.uncertainty.fmin <= input_f0[0]) 
                        & (model.uncertainty.fmax >= input_f0[0])
                        & (model.uncertainty.q == 0.5), 
                        ~model.uncertainty.columns.isin(['fmin', 'fmax', 'q'])].values[:,0]
                lower_bound = recovered_output_pred.T[:,0] - uncertainty
                upper_bound = recovered_output_pred.T[:,0] + uncertainty
                ax_TL.fill_between(model.dataset.distances, lower_bound, upper_bound, alpha=0.3, color='tab:blue', zorder=1, label='uncertainty')
                
            ## TODO: To fix 
            if False:
                ## Output error
                uncertainty = [
                    recovered_output_pred.T[:,0],
                    recovered_output_pred.T[:,0]-model.dataset.std_distance_output,
                    recovered_output_pred.T[:,0]+model.dataset.std_distance_output,
                ]
                
                ## Add SVD/PCA encoding uncertainties
                if model.dataset.method_encoding in ['PCA', 'SVD']:
                    uncertainty.append(recovered_pred_lower_std.T[:,0])
                    uncertainty.append(recovered_pred_upper_std.T[:,0])
                uncertainty = np.array(uncertainty)
                    
                lower_bound = uncertainty.min(axis=0)
                upper_bound = uncertainty.max(axis=0)
                ax_TL.fill_between(model.dataset.distances, lower_bound, upper_bound, alpha=0.1, color='tab:blue', zorder=1, label='uncertainty')
    
        if legend:
            ax_TL.legend(bbox_to_anchor=(1., 0.45), bbox_transform=ax_TL.transAxes, frameon=False, labelspacing=0.04, handlelength=0.4, handletextpad=0.2)
        ax_TL.set_xlabel('Range (km)')
        if show_labels:
            ax_TL.set_ylabel('TL\n(db)')
        else:
            ax_TL.tick_params(axis='both', which='both', labelleft=False, left=False)
        ax_TL.set_xlim([distances.min(), distances.max()])
        ax_TL.set_ylim(ylim)
        
        ax_winds.set_xlim([distances.min(), distances.max()])
        ax_winds.set_xticklabels([])
        if show_labels:
            ax_winds.set_ylabel('Altitude\n(km)')
        else:
            ax_winds.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
        ax_winds.set_title('['+str(simu_slice_no)+'] Transmmission-loss vs range with f = ' + str(np.round(input_f0[0], 3)) + 'Hz' )
        
        
        if new_figure:
            fig.align_labels()
            format_file = 'TL_predicted_{type_model}_simu_{no}{ext_name_file}.png'
            plt.savefig(output_dir + format_file.format(type_model=model.type_model, no=simu_slice_no, ext_name_file=ext_name_file))

def load_one_id(id, model, dataset):

    """
    Load one item from the training or testing dataset
    """

    ## Test over one profile
    if dataset == 'test':
        simu_id    = model.test_ids[id]
        
    elif dataset == 'train':
        simu_id    = model.train_ids[id]
        """
        simu_slice_no = model.train_input_slice_no[id]
        input_wind = model.train_input_wind[id:id+1,:]
        input_f0   = model.train_input_f0[id:id+1]
        output     = model.train_output[id:id+1, :]
        """
    else:
        sys.exit('Dataset type not recognized')
        
    simu_slice_no = model.dataset.slice_nos[simu_id]
    input_wind = model.dataset.encoded_input[simu_id:simu_id+1,:]
    input_f0   = model.dataset.f0s[simu_id:simu_id+1]
    output     = model.dataset.encoded_output[simu_id:simu_id+1, :]

    return simu_id, simu_slice_no, input_wind, input_f0, output

def plot_TL_examples(model, figure_dir, l_TL={'troposphere': ('test', 10)}, ext_name_file='', ylim=[-130., 0.], show_LP12=False, show_uncertainty=True, show_ceff=True):

    """
    Figure 3 - Plot TL comparisons requested by Jelle Assink
    """

    ## Label settings
    alphabet = string.ascii_lowercase
    
    ## Setup figure
    fig = plt.figure(figsize=(6.8, 2.5)); 
    h_wind  = 2
    h_TL    = 6
    grid = fig.add_gridspec(h_TL+h_wind, len(l_TL))
    
    ## Plot comparisons TL
    for iax, label_TL in enumerate(l_TL):
    
        dataset_type, id_TL = l_TL[label_TL]
        
        ## Setup grid
        params_winds, params_TL = {}, {}
        if iax > 0:
            params_winds = {'sharex': ax_winds, 'sharey': ax_winds}
            params_TL = {'sharex': ax_TL, 'sharey': ax_TL}
        ax_winds = fig.add_subplot(grid[:h_wind, iax], **params_winds)
        ax_TL = fig.add_subplot(grid[h_wind:, iax], **params_TL)
        axs = [ax_winds, ax_TL]
        simu_id, simu_slice_no, input_wind, input_f0, output = load_one_id(id_TL, model, dataset_type)
        
        ## Plot winds and TL
        params = {'additional_homogeneous':[], 'additional_f0s':[], 'dataset':dataset_type, 'show_predictions':True, 'show_ceff':show_ceff, 'axs':axs, 'legend':False, 'show_labels':False, 'ylim': ylim, 'bbox_to_anchor_winds': (0.25, -0.8), 'show_LP12': show_LP12, 'show_uncertainty': show_uncertainty}
        if iax == 0:
            params.update({'legend':True, 'show_labels':True})
        compare_TL([id_TL], model, figure_dir, **params)
        
        ## Labels
        ax_winds.set_title(label_TL)
        ax_winds.text(-0.05, 1.7, alphabet[iax]+')', ha='right', va='top', transform=ax_winds.transAxes, 
                   bbox=dict(facecolor='w', edgecolor='w', pad=0.1, alpha=0.5), fontsize=15., fontweight='bold')
        ax_TL.text(0.03, 0.03, '{f:.2f} Hz'.format(f=input_f0[0]), ha='left', va='bottom', transform=ax_TL.transAxes, fontsize=12.)
    
    fig.align_labels()
    fig.subplots_adjust(hspace=0.4, left=0.15, right=0.95, bottom=0.2)
    
    ## Save figure
    plt.savefig(figure_dir + 'comparison_TLs{ext_name_file}.pdf'.format(ext_name_file=ext_name_file))

def plot_error_distributions(all_errors, fig, iax, ax_f0, ax_range, ax_ceff, ax_cbar, plot_ceff=True):

    """
    Create error distributions for a given grid plot
    
    Example:
    fig = plt.figure(); 
    iax = 0
    w_main_plots = 3
    w_small_plots = 2
    grid = fig.add_gridspec(2,w_main_plots*2)
    grid_f0 = grid[0,:w_main_plots]
    ax_f0 = fig.add_subplot(grid_f0)
    grid_range = grid[0,w_main_plots:]
    ax_range = fig.add_subplot(grid_range)
    grids_ceff = [grid[1:, w_small_plots*icol:w_small_plots*(icol+1)] for icol in range(3)]
    ax_ceff = append( fig.add_subplot(one_grid) for one_grid in grids_ceff )
    grid_cbar = grid[1, -1]
    ax_cbar = fig.add_subplot(one_grid)
    utils_figures.plot_error_distributions(all_errors, fig, iax, ax_f0, ax_range, ax_ceff, ax_cbar, plot_ceff=True)
    """

    alphabet = string.ascii_lowercase

    ## Plot error f0
    print('f0')
    iax += 1
    g = sns.kdeplot(data=all_errors, x="rmse_err", hue="q-f0s", fill=True, common_norm=False, palette="mako", alpha=.4, linewidth=1, ax=ax_f0, legend=True)
    move_legend(ax_f0, "lower left", bbox_to_anchor=(-0.0, 1.2), frameon=False, title='$f_0$ (Hz)', ncol=2, format_legend='{nb:.2g}')
    ax_f0.set_xlim(xlim_err)
    ax_f0.set_xlabel('RMSE (db)', labelpad=0.)
    ax_f0.set_ylabel('Error\ndistribution')
    ax_f0.tick_params(axis='x', which='major', pad=0.1)
    plt.setp(ax_f0, yticks=[])
    ax_f0.text(-0.05, 1.4, alphabet[iax]+')', ha='right', va='top', transform=ax_f0.transAxes, 
               bbox=dict(facecolor='w', edgecolor='w', pad=0.1, alpha=0.5), fontsize=15., fontweight='bold')
    ax_f0.text(0.95, 0.95, '$f_0$', ha='right', va='top', transform=ax_f0.transAxes, 
               bbox=dict(facecolor='w', edgecolor='w', pad=0.1, alpha=0.5), fontsize=12.)
    ax_f0.xaxis.tick_top()
    ax_f0.xaxis.set_label_position('top')
    ax_f0.minorticks_on()
    
    ## Plot error range
    print('range')
    iax += 1
    col_labels = {}
    errors_range = pd.DataFrame()
    for col in all_errors.columns:
        if not 'mse_err_' in col:
            continue
            
        col_labels[col] = col.split('_')[-1]
        print(col)
        errors_range[col] = np.sqrt(all_errors[col])
    errors_range = convert_columns_to_rows(errors_range, column_name='mse_err_', col_labels=col_labels)
    print('done')

    g = sns.kdeplot(data=errors_range, x="value", hue="layer", fill=True, common_norm=False, palette="rocket", alpha=.4, linewidth=1, ax=ax_range, legend=True)
    move_legend(ax_range, "lower left", bbox_to_anchor=(-0.0, 1.2), frameon=False, title='range (km)', ncol=2, format_legend='{nb:.3g}')
    ax_range.set_xlim(xlim_err)
    ax_range.set_xlabel('RMSE (db)', labelpad=0.05)
    ax_range.tick_params(axis='x', which='major', pad=0.1)
    plt.setp(ax_range, yticks=[])
    ax_range.set_ylabel('')
    ax_range.text(-0.05, 1.4, alphabet[iax]+')', ha='right', va='top', transform=ax_range.transAxes, 
               bbox=dict(facecolor='w', edgecolor='w', pad=0.1, alpha=0.5), fontsize=15., fontweight='bold')
    ax_range.text(0.95, 0.95, 'range', ha='right', va='top', transform=ax_range.transAxes, 
               bbox=dict(facecolor='w', edgecolor='w', pad=0.1, alpha=0.5), fontsize=12.)
    ax_range.xaxis.tick_top()
    ax_range.xaxis.set_label_position('top')
    ax_range.minorticks_on()
    
    ## Plot error ceff
    if plot_ceff:
        print('Plotting ceff')
        all_errors_loc = all_errors.groupby('simu_id').mean().reset_index()
        cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
        for icol, col_ceff in enumerate(cols_ceff):
            iax += 1
            
            cbar = {}
            if icol == len(cols_ceff)-1:
                ax_cbar.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
                cbar = {'cbar': True, 'cbar_kws': {'extend': 'both', 'format': '%.2f'}, 'cbar_ax': ax_cbar}
            g = sns.kdeplot(data=all_errors_loc, x="rmse_err", y=col_ceff, n_levels=5, **cbar, fill=True, common_norm=True, cmap=sns.color_palette("ch:s=-.2,r=.6", as_cmap=True), alpha=1., ax=ax_ceff[icol], legend=False)
            
            ax_ceff[icol].set_xlim([0., 40.])
            ax_ceff[icol].set_ylim([0.8, 1.2])
            ax_ceff[icol].text(0.95, 0.95, '$\overline{{c}}_{{eff}}$ {alt}'.format(alt=col_ceff.replace('veff-','')), ha='right', va='top', transform=ax_ceff[icol].transAxes, 
               bbox=dict(facecolor='w', edgecolor='w', pad=0.1, alpha=0.5), fontsize=12.)
            ax_ceff[icol].set_ylabel('$\overline{{c}}_{{eff}}$')
            ax_ceff[icol].set_xlabel('RMSE (db)')
            
                
            if icol > 0:
                ax_ceff[icol].tick_params(labelleft=False)
                ax_ceff[icol].set_ylabel('')
                loc_label = (-0.05, 1.2)
            else:
                loc_label = (-0.25, 1.2)
            ax_ceff[icol].text(loc_label[0], loc_label[1], alphabet[iax]+')', ha='right', va='top', transform=ax_ceff[icol].transAxes, 
               bbox=dict(facecolor='w', edgecolor='w', pad=0.1, alpha=0.5), fontsize=15., fontweight='bold')
            ax_ceff[icol].minorticks_on()

def plot_convergence_erors_comparisons(model, all_errors, figure_dir, cost, l_TL_1=[10], l_TL_2=[1000], 
                                       ext_name_file='', plot_examples=False, figsize=(6.4, 5.8), plot_ceff=True,
                                       cols_ceff=['veff-1-15', 'veff-35-60', 'veff-80-120'], xlim_err=[0., 40.],
                                       ylim_maps_ceff=[0.8, 1.2]):

    """
    Figure 3 - Plot a convergence of ML and errors
    """

    alphabet = string.ascii_lowercase
    col_labels = {'1-15': 'troposphere', '5-20': 'troposphere', '20-50': 'stratosphere', '35-60': 'stratosphere', '50-100': 'thermosphere', '80-120': 'thermosphere'}
    iax = -1

    ## Setup figure
    fig = plt.figure(figsize=figsize); 
    nb_preplot = 1
    h_error = 4
    h_sep_conv_cost = 2
    h_wind  = 0
    h_TL    = 0
    if plot_examples:
        h_wind  = 2
        h_TL    = 6
    h_ceff = 0
    if plot_ceff:
        h_ceff = 4
    h_shift = 1
    w_total = 19
    w_main_plots = (w_total-1)//2
    w_ceff_plots = (w_total-1)//3
    grid = fig.add_gridspec(h_TL+h_wind+h_ceff+(nb_preplot+1)*h_error+h_shift+h_sep_conv_cost, w_total)
    
    ## Plot convergence
    print('convergence')
    iax += 1
    ax_convergence = fig.add_subplot(grid[:h_error, :w_main_plots])
    plot_convergence(model, '', ax=ax_convergence)
    ax_convergence.text(-0.05, 1.4, alphabet[iax]+')', ha='right', va='top', transform=ax_convergence.transAxes, 
               bbox=dict(facecolor='w', edgecolor='w', pad=0.1, alpha=0.5), fontsize=15., fontweight='bold')
    ax_convergence.set_xlabel('Epoch')
    ax_convergence.tick_params(axis='x', which='major', pad=0.1)
    ax_convergence.xaxis.tick_top()
    ax_convergence.xaxis.set_label_position('top')
    ax_convergence.legend(loc='lower left', bbox_to_anchor=(-0.04, 1.3), frameon=False, labelspacing=0.04, handlelength=0.4, handletextpad=0.05)
    ax_convergence.minorticks_on()
    
    ## Plot cost
    print('cost')
    iax += 1
    #ax_cost = fig.add_subplot(grid[h_error+h_sep_conv_cost:h_error*2+h_sep_conv_cost, 0])
    ax_cost = fig.add_subplot(grid[:h_error, w_main_plots:2*w_main_plots])
    ax_cost.text(-0.05, 1.4, alphabet[iax]+')', ha='right', va='top', transform=ax_cost.transAxes, 
               bbox=dict(facecolor='w', edgecolor='w', pad=0.1, alpha=0.5), fontsize=15., fontweight='bold')
    ## WARNING: Le Pichon is actually PE here
    ax_cost.plot(cost.freq, cost['Le Pichon'], label='PE', color='tab:red', alpha=0.4)
    ax_cost.plot(cost.freq, cost['ML'], label='ML', color='tab:blue')
    ax_cost.set_yscale('log')
    ax_cost.set_xlabel('Frequency (Hz)')
    ax_cost.set_ylabel('Cost (s)')
    ax_cost.tick_params(axis='x', which='major', pad=0.1)
    ax_cost.tick_params(axis='y', which='minor')
    ax_cost.xaxis.tick_top()
    ax_cost.xaxis.set_label_position('top')
    ax_cost.yaxis.tick_right()
    ax_cost.yaxis.set_label_position('right')
    ax_cost.legend(loc='lower left', bbox_to_anchor=(-0.04, 1.3), frameon=False, labelspacing=0.04, handlelength=0.4, handletextpad=0.05)
    ax_cost.minorticks_on()
    
    ## Plot error f0
    print('f0')
    iax += 1
    ax_f0 = fig.add_subplot(grid[h_error*(nb_preplot)+h_shift+h_sep_conv_cost:(nb_preplot+1)*h_error+h_shift+h_sep_conv_cost, :w_main_plots])
    #df_to_plot = utils_figures.convert_columns_to_rows(all_errors, column_name='q-veff', col_labels=col_labels, remove_cols=['40-50'])
    g = sns.kdeplot(data=all_errors, x="rmse_err", hue="q-f0s", fill=True, common_norm=False, palette="mako", alpha=.4, linewidth=1, ax=ax_f0, legend=True)
    move_legend(ax_f0, "lower left", bbox_to_anchor=(-0.0, 1.2), frameon=False, title='$f_0$ (Hz)', ncol=2, format_legend='{nb:.2g}')
    ax_f0.set_xlim(xlim_err)
    ax_f0.set_xlabel('RMSE (db)', labelpad=0.)
    ax_f0.set_ylabel('Error\ndistribution')
    ax_f0.tick_params(axis='x', which='major', pad=0.1)
    plt.setp(ax_f0, yticks=[])
    ax_f0.text(-0.05, 1.4, alphabet[iax]+')', ha='right', va='top', transform=ax_f0.transAxes, 
               bbox=dict(facecolor='w', edgecolor='w', pad=0.1, alpha=0.5), fontsize=15., fontweight='bold')
    ax_f0.text(0.95, 0.95, '$f_0$', ha='right', va='top', transform=ax_f0.transAxes, 
               bbox=dict(facecolor='w', edgecolor='w', pad=0.1, alpha=0.5), fontsize=12.)
    ax_f0.xaxis.tick_top()
    ax_f0.xaxis.set_label_position('top')
    ax_f0.minorticks_on()
    
    ## Plot error range
    print('range')
    iax += 1
    ax_range = fig.add_subplot(grid[h_error*(nb_preplot)+h_shift+h_sep_conv_cost:(nb_preplot+1)*h_error+h_shift+h_sep_conv_cost, w_main_plots:2*w_main_plots])
    col_labels = {}
    errors_range = pd.DataFrame()
    for col in all_errors.columns:
        if not 'mse_err_' in col:
            continue
            
        col_labels[col] = col.split('_')[-1]
        print(col)
        errors_range[col] = np.sqrt(all_errors[col])
    errors_range = convert_columns_to_rows(errors_range, column_name='mse_err_', col_labels=col_labels)
    print('done')

    g = sns.kdeplot(data=errors_range, x="value", hue="layer", fill=True, common_norm=False, palette="rocket", alpha=.4, linewidth=1, ax=ax_range, legend=True)
    move_legend(ax_range, "lower left", bbox_to_anchor=(-0.0, 1.2), frameon=False, title='range (km)', ncol=2, format_legend='{nb:.3g}')
    ax_range.set_xlim(xlim_err)
    ax_range.set_xlabel('RMSE (db)', labelpad=0.05)
    ax_range.tick_params(axis='x', which='major', pad=0.1)
    plt.setp(ax_range, yticks=[])
    ax_range.set_ylabel('')
    ax_range.text(-0.05, 1.4, alphabet[iax]+')', ha='right', va='top', transform=ax_range.transAxes, 
               bbox=dict(facecolor='w', edgecolor='w', pad=0.1, alpha=0.5), fontsize=15., fontweight='bold')
    ax_range.text(0.95, 0.95, 'range', ha='right', va='top', transform=ax_range.transAxes, 
               bbox=dict(facecolor='w', edgecolor='w', pad=0.1, alpha=0.5), fontsize=12.)
    ax_range.xaxis.tick_top()
    ax_range.xaxis.set_label_position('top')
    ax_range.minorticks_on()
    
    ## Plot error ceff
    if plot_ceff:
        print('Plotting ceff')
        all_errors_loc = all_errors.groupby('simu_id').mean().reset_index()
        cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
        ax_ceff = []
        for icol, col_ceff in enumerate(cols_ceff):
            iax += 1
            if icol == len(cols_ceff)-1:
                #ax_ceff.append( fig.add_axes([0,0,1,1]) )
                ax_ceff.append( fig.add_subplot(grid[-h_ceff:, w_ceff_plots*icol:w_ceff_plots*(icol+1)]) )
            
            else:
                ax_ceff.append( fig.add_subplot(grid[-h_ceff:, w_ceff_plots*icol:w_ceff_plots*(icol+1)]) )
            
        #df_to_plot = utils_figures.convert_columns_to_rows(all_errors, column_name='q-veff', col_labels=col_labels, remove_cols=['40-50'])
            cbar = {}
            if icol == len(cols_ceff)-1:
                axins = fig.add_subplot(grid[-h_ceff:, -1:])
                #axins = inset_axes(ax_ceff[-1], width="4%", height="100%", loc='lower left', bbox_to_anchor=(1.05, 0., 1, 1.), bbox_transform=ax_ceff[-1].transAxes, borderpad=0)
                #axins = fig.add_axes([.8, .25, .03, .4]) 
                axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
                cbar = {'cbar': True, 'cbar_kws': {'extend': 'both', 'format': '%.2f'}, 'cbar_ax': axins}
            g = sns.kdeplot(data=all_errors_loc, x="rmse_err", y=col_ceff, n_levels=5, **cbar, fill=True, common_norm=True, cmap=sns.color_palette("ch:s=-.2,r=.6", as_cmap=True), alpha=1., ax=ax_ceff[-1], legend=False)
            
            ax_ceff[-1].set_xlim(xlim_err)
            ax_ceff[-1].set_ylim(ylim_maps_ceff)
            ax_ceff[-1].text(0.95, 0.95, '$\overline{{c}}_{{eff}}$ {alt}'.format(alt=col_ceff.replace('veff-','')), ha='right', va='top', transform=ax_ceff[-1].transAxes, 
               bbox=dict(facecolor='w', edgecolor='w', pad=0.1, alpha=0.5), fontsize=12.)
            ax_ceff[-1].set_ylabel('$\overline{{c}}_{{eff}}$')
            ax_ceff[-1].set_xlabel('RMSE (db)')
            
                
            if icol > 0:
                #plt.setp(ax_ceff[-1], yticks=[])
                ax_ceff[-1].tick_params(labelleft=False)
                ax_ceff[-1].set_ylabel('')
                #ax_ceff.set_xlabel('')
                loc_label = (-0.05, 1.2)
            else:
                loc_label = (-0.25, 1.2)
            ax_ceff[-1].text(loc_label[0], loc_label[1], alphabet[iax]+')', ha='right', va='top', transform=ax_ceff[-1].transAxes, 
               bbox=dict(facecolor='w', edgecolor='w', pad=0.1, alpha=0.5), fontsize=15., fontweight='bold')
            ax_ceff[-1].minorticks_on()
    
        #move_legend(ax_f0, "lower left", bbox_to_anchor=(1.1, 1.45), frameon=False, title='$f_0$ (Hz)')
        """
        move_legend(ax_f0, "lower left", bbox_to_anchor=(0.3, 1.45), frameon=False, title='$f_0$ (Hz)')
        ax_f0.set_xlim([0., 20.])
        ax_f0.set_xlabel('RMSE (db)', labelpad=0.)
        ax_f0.set_ylabel('Error\ndistribution')
        ax_f0.tick_params(axis='x', which='major', pad=0.1)
        plt.setp(ax_f0, yticks=[])
        ax_f0.text(-0.05, 1.0, alphabet[iax]+')', ha='right', va='top', transform=ax_f0.transAxes, 
                   bbox=dict(facecolor='w', edgecolor='w', pad=0.1, alpha=0.5), fontsize=15., fontweight='bold')
        ax_f0.text(0.95, 0.95, '$f_0$', ha='right', va='top', transform=ax_f0.transAxes, 
                   bbox=dict(facecolor='w', edgecolor='w', pad=0.1, alpha=0.5), fontsize=12.)
        ax_f0.xaxis.tick_top()
        ax_f0.xaxis.set_label_position('top')
        """
    
    ## Plot Transmission loss examples
    if plot_examples:
        ## Plot comparisons TL
        iax += 1
        ax_winds1 = fig.add_subplot(grid[(nb_preplot+1)*h_error+h_shift+h_sep_conv_cost:(nb_preplot+1)*h_error+h_wind+h_shift+h_sep_conv_cost, 0])
        ax_TL1 = fig.add_subplot(grid[(nb_preplot+1)*h_error+h_wind+h_shift+h_sep_conv_cost:, 0])
        axs = [ax_winds1, ax_TL1]
        simu_id, simu_slice_no, input_wind, input_f0, output = load_one_id(l_TL_1[0], model, 'test')
        compare_TL(l_TL_1, model, figure_dir, additional_homogeneous=[], additional_f0s=[], dataset='test', show_predictions=True, axs=axs)
        ax_winds1.set_title('')
        ax_winds1.text(-0.05, 1.7, alphabet[iax]+')', ha='right', va='top', transform=ax_winds1.transAxes, 
                   bbox=dict(facecolor='w', edgecolor='w', pad=0.1, alpha=0.5), fontsize=15., fontweight='bold')
        
        ax_TL1.text(0.03, 0.97, '{f:.2f} Hz'.format(f=input_f0[0]), ha='left', va='top', transform=ax_TL1.transAxes, fontsize=12.)
        
        ## Plot comparisons TL
        iax += 1
        ax_winds2 = fig.add_subplot(grid[(nb_preplot+1)*h_error+h_shift+h_sep_conv_cost:(nb_preplot+1)*h_error+h_wind+h_shift+h_sep_conv_cost, 1])
        ax_TL2 = fig.add_subplot(grid[(nb_preplot+1)*h_error+h_wind+h_shift+h_sep_conv_cost:, 1])
        axs = [ax_winds2, ax_TL2]
        simu_id, simu_slice_no, input_wind, input_f0, output = load_one_id(l_TL_2[0], model, 'test')
        compare_TL(l_TL_2, model, figure_dir, additional_homogeneous=[], additional_f0s=[], dataset='test', show_predictions=True, axs=axs, legend=False)
        plt.setp(ax_winds2, xticks=[], yticks=[])
        plt.setp(ax_TL2, yticks=[])
        ax_winds2.set_ylabel('')
        ax_TL2.set_ylabel('')
        ax_winds2.set_title('')
        ax_winds2.text(-0.05, 1.7, alphabet[iax]+')', ha='right', va='top', transform=ax_winds2.transAxes, 
                   bbox=dict(facecolor='w', edgecolor='w', pad=0.1, alpha=0.5), fontsize=15., fontweight='bold')
        
        ax_TL2.text(0.03, 0.97, '{f:.2f} Hz'.format(f=input_f0[0]), ha='left', va='top', transform=ax_TL2.transAxes, fontsize=12.)
    
    fig.align_labels()
    fig.subplots_adjust(hspace=2.3, wspace=4.5, left=0.15, bottom=0.1, top=0.85)
    
    plt.savefig(figure_dir + 'comparison_errors{ext_name_file}.pdf'.format(ext_name_file=ext_name_file))
    
def find_dataset_id_from_simu_id(model, simu_id):

    ## Test over one profile
    try:
        dataset = 'test'
        id = np.where(model.test_ids == simu_id)[0][0]
        _, simu_slice_no, input_wind, input_f0, output = load_one_id(id, model, 'test')
        
    except:
        dataset = 'train'
        id = np.where(model.train_ids == simu_id)[0][0]
        _, simu_slice_no, input_wind, input_f0, output = load_one_id(id, model, 'train')
    
    return id, simu_slice_no, input_wind, input_f0, output, dataset
    
def compute_map_TL_veff(model, f0, deviation_from_f0=0.1, max_veff_tropo=1.05, max_veff_thermo=1.1, min_veff_strato=1.,
                        tropo_col='veff-5-20', strato_col='veff-40-50', thermo_col='veff-50-100'):

    ## Retrieve simulation around the requested frequency
    list_simulations = model.dataset.properties.loc[(abs(f0 - model.dataset.properties.f0s) <= deviation_from_f0)
                                                    & (model.dataset.properties[tropo_col] <= max_veff_tropo)
                                                    & ((model.dataset.properties[thermo_col] <= max_veff_thermo)
                                                    | (model.dataset.properties[strato_col] >= min_veff_strato))].groupby('simu_id').first().reset_index()
    simu_ids = list_simulations.simu_id.values
    
    all_TLs = []
    all_TLs_true = []
    winds = []
    for simu_id in simu_ids:
    
        id, simu_slice_no, input_wind, input_f0, output, _ = find_dataset_id_from_simu_id(model, simu_id)
           
        ## Encoded inputs/outputs
        encoded_input = input_wind

        encoded_outputs = model.model.predict([encoded_input, input_f0])
        if not model.dataset.method_output_name == 'linear':
            recovered_output_pred = model.dataset.SVD_output.inverse_transform(encoded_outputs)
        else:
            recovered_output_pred = encoded_outputs
        
        recovered_output_pred = model.scaler_TL.inverse_transform(recovered_output_pred)
        recovered_output_true = model.dataset.TL_stack[simu_id][:, None].T
        
        all_TLs_true.append( recovered_output_true.T[:, 0] )
        all_TLs.append( recovered_output_pred.T[:, 0] )
        winds.append( encoded_input[:,0] )
        
    return simu_ids, np.array(all_TLs), np.array(all_TLs_true), np.array(winds)
    
def read_RD_profiles(summary_file, model_dir):
    
    """
    Read a list of vertical profiles from a summary plot
    """

    all_profiles = pd.DataFrame()
    summary = pd.read_csv(model_dir + summary_file, delim_whitespace=True, header=None)
    summary.columns = ['range', 'file']
    for iprofile, profile_info in summary.iterrows():
        file_profile = model_dir + profile_info.file
        profile = pd.read_csv(file_profile, comment='#', header=None, delim_whitespace=True)
        profile.columns = ['z', 'u', 'v', 't', 'rho', 'p']
        profile['range'] = profile_info.range
        all_profiles = all_profiles.append( profile )

    all_profiles.reset_index(drop=True, inplace=True)
    return all_profiles

def determine_computational_cost(model, input_frequencies, max_profiles=10, coef_downsample=20, coef_upsample=4,
                                 simu_dir='/staff/quentin/Documents/Projects/ML_attenuation_prediction/computational_cost_data/slice_no_1301646/'):

    ## Metadata of current simulation
    islice = int(simu_dir.split('_')[-1].replace('/',''))
    summary_file = 'slice_no_' + str(islice) + '_summary.dat'
    
    ## Load params
    with open(simu_dir + 'param.txt') as f:
        lines = f.readlines()[1].split()
        
    freq = float(lines[7])
    az_rad = np.radians(float(lines[5]))
    
    ## Retrieve profiles
    profiles = read_RD_profiles(summary_file, simu_dir)
    ranges = profiles.range.unique()
    for irange, range_ in enumerate(ranges):
        one_profile = profiles.loc[profiles.range==range_]
        winds = np.sin(az_rad) * one_profile.u.values + np.cos(az_rad) * one_profile.v.values
        if irange == 0:
            wind_stack_temp = winds
        else:
            wind_stack_temp = np.c_[wind_stack_temp, winds]

    wind_stack_temp = wind_stack_temp[:,:max_profiles]
    wind_stack = wind_stack_temp[None,...]
    
    ## Upsample/downsample
    image_shape = wind_stack.shape
    encoded_input = resize(wind_stack, 
                        (image_shape[0], image_shape[1]//coef_downsample, 
                            image_shape[2]*coef_upsample), 
                        anti_aliasing=True)     
    encoded_input = encoded_input[...,None]
    
    os.chdir(simu_dir)
    computational_time = pd.DataFrame()
    ## Loop over different frequencies to compute run time
    for freq in input_frequencies:
        loc_dict = {}
        
        loc_dict['freq'] = freq
        
        ## Run PE
        template = lines.copy()
        template[7] = str(freq)
        start = time.time()
        os.system(' '.join(template) + ' --npade 7')
        end = time.time()
        loc_dict['LP12'] = end - start
        
        ## Run ML
        start = time.time()
        input_f0 = np.array([[freq]])
        print(encoded_input.shape, input_f0.shape)
        _ = model.model.predict([encoded_input, input_f0])
        end = time.time()
        loc_dict['ML'] = end - start
        
        computational_time = computational_time.append( [loc_dict] )
    
    computational_time.reset_index(drop=True, inplace=True)
    
    return computational_time
    
def plot_2dTL_comparisons(model, all_errors, figure_dir, output_dir, 
                                xlim=[0.85, 1.2], ylim=[0., 50.], 
                                ceffs_low={}, error_Alexis_low={}, error_ML_low={}, 
                                ceffs_high={}, error_Alexis_high={}, error_ML_high={},
                                interpolation='linear', l_TL_1=[10], nb_z=100, nb_r=400,
                                ext_name_file=''):

    """
    Plot 2d TLs along with winds and ground TLs for large and small errors between ML and PE simulations
    """

    #utils_figures.plot_2dTL_comparisons(model, all_errors, figure_dir, output_dir, error_Alexis_low=error_Alexis_low, error_ML_low=error_ML_low, ceffs_low=ceffs_low, error_Alexis_high=error_Alexis_high, error_ML_high=error_ML_high, ceffs_high=ceffs_high)
    
    #data = model.dataset.properties.loc[model.dataset.properties.simu_id.isin(simu_ids_low)]; simus = data.groupby('simu_id').mean().reset_index(); ceffs = np.linspace(simus[ceff_pick_for_error].min(), simus[ceff_pick_for_error].max(), nb_ceff)
    
    ## Setup figure
    alphabet = string.ascii_lowercase
    fig = plt.figure()
    grid = fig.add_gridspec(4, 2)
    iax = -1
    
    """
    grid = fig.add_gridspec(7, 2)
    axs_errors_low = []
    for iax in range(3):
        axs_errors_low.append( fig.add_subplot(grid[iax, 0]) )
    axs_errors_high = []
    for iax in range(3):
        axs_errors_high.append( fig.add_subplot(grid[iax, 1]) )

    iax = -1
    
    ## Plot low frequencies
    if not errors_Alexis_low or not errors_ML_low or not ceffs_low:
        f0 = 0.15
        deviation_from_f0 = 0.05
        simu_ids, TLs, TLs_true, _ = \
            compute_map_TL_veff(model, f0, deviation_from_f0=deviation_from_f0, max_veff_tropo=2., max_veff_thermo=2.)
        data = model.dataset.properties.loc[model.dataset.properties.simu_id.isin(simu_ids)]
        simus = data.groupby('simu_id').mean().reset_index()
        file = './data_alexis/ATTEN_FREQ_0.1.asc'
        ranges, ceff, attenuation = load_alexis_data(file)
        DD, CEFF = np.meshgrid(ranges, ceff)
        #errors_Alexis_low, errors_ML_low, ceffs_low = get_errors_interpolated(ceff_errors, model, TLs, TLs_true, simus, nb_ceff, interpolation, DD, CEFF, attenuation)
        errors_Alexis_low, errors_ML_low, ceffs_low = get_errors_interpolated(ceff_unknown, model, TLs, TLs_true, simus, nb_ceff, interpolation, DD, CEFF, attenuation)
    plot_errors_interpolated(ceff_errors, simus, ceffs_low, errors_Alexis_low, errors_ML_low, xlim=xlim, ylim=ylim, axs=axs_errors_low, plot_legend=False)
    #utils_figures.plot_errors_interpolated(ceff_errors, simus, ceffs_high, errors_Alexis_high, errors_ML_high, xlim=xlim, ylim=ylim, plot_legend=False)
    
    for ax in axs_errors_low:
        iax += 1
        ax.text(-0.25, 1.0, alphabet[iax]+')', ha='right', va='bottom', transform=ax.transAxes, fontsize=15., fontweight='bold')
    for ax in axs_errors_low[1:]:
        plt.setp(ax, xticks=[])
    axs_errors_low[0].xaxis.tick_top()
    axs_errors_low[0].xaxis.set_label_position('top')
    axs_errors_low[0].set_xlabel('$c_{eff}$')
    axs_errors_low[-1].set_xlabel('')
    
    ## Plot high frequencies
    if not errors_Alexis_high or not errors_ML_high or not ceffs_high:
        f0 = 2.
        deviation_from_f0 = 0.25
        simu_ids, TLs, TLs_true, _ = utils_figures.compute_map_TL_veff(model, f0, deviation_from_f0=deviation_from_f0, max_veff_tropo=2., max_veff_thermo=2.)
        data = model.dataset.properties.loc[model.dataset.properties.simu_id.isin(simu_ids)]
        simus = data.groupby('simu_id').mean().reset_index()
        file = './data_alexis/ATTEN_FREQ_2.asc'
        ranges, ceff, attenuation = load_alexis_data(file)
        DD, CEFF = np.meshgrid(ranges, ceff)
        #errors_Alexis_high, errors_ML_high, ceffs_high = get_errors_interpolated(ceff_errors, model, TLs, TLs_true, simus, nb_ceff, interpolation, DD, CEFF, attenuation)
        errors_Alexis_high, errors_ML_high, ceffs_high = get_errors_interpolated(ceff_unknown, model, TLs, TLs_true, simus, nb_ceff, interpolation, DD, CEFF, attenuation)
    plot_errors_interpolated(ceff_errors, simus, ceffs_high, errors_Alexis_high, errors_ML_high, xlim=xlim, ylim=ylim, axs=axs_errors_high, plot_legend=False)
    
    for ax in axs_errors_high:
        iax += 1
        ax.text(-0.035, 1.0, alphabet[iax]+')', ha='right', va='bottom', transform=ax.transAxes, fontsize=15., fontweight='bold')
    for iax, ax in enumerate(axs_errors_high):
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        if iax > 0:
            plt.setp(ax, xticks=[])
    axs_errors_high[0].xaxis.tick_top()
    axs_errors_high[0].xaxis.set_label_position('top')
    axs_errors_high[0].set_xlabel('$c_{eff}$')
    axs_errors_high[-1].set_xlabel('')
    """
    
    ## Get simulation with largest error and plot
    iax += 1
    ax_winds1 = fig.add_subplot(grid[0, 0])
    ax_TL2d1 = fig.add_subplot(grid[1:3, 0])
    ax_TL1 = fig.add_subplot(grid[3, 0])
    axs = [ax_winds1, ax_TL1]
    
    f0 = 0.15
    ceff_pick_for_error = 'veff-40-50'
    ceff_max_error = ceffs_low[error_Alexis_low>0.8][error_Alexis_low[error_Alexis_low>0.8].argmax()]
    f0 = 2.
    ceff_max_error = 1.1
    simu_id = all_errors.sort_values(by='rmse_err'); simu_id = simu_id.loc[(abs(simu_id['f0s']-f0) < 0.05)]; simu_id = simu_id.loc[abs(simu_id[ceff_pick_for_error]-ceff_max_error) < 0.01]; simu_id = simu_id.loc[simu_id['veff-5-20'] == simu_id['veff-5-20'].max(), 'simu_id'].iloc[-1]; id, simu_slice_no, input_wind, input_f0, output, dataset = find_dataset_id_from_simu_id(model, simu_id)
    l_TL_1 = [id]
    print('-->', '{no:.0f}'.format(no=simu_slice_no))
    compare_TL(l_TL_1, model, figure_dir, additional_homogeneous=[], additional_f0s=[], dataset=dataset, show_predictions=True, axs=axs, bbox_to_anchor_winds=(0.5, 1.05), label_predicted='ML')
    ax_winds1.set_title('')
    ax_winds1.text(-0.05, 1.2, alphabet[iax]+')', ha='right', va='top', transform=ax_winds1.transAxes, fontsize=15., fontweight='bold')
    ax_winds1.set_ylabel('')
    
    file = './data_alexis/ATTEN_FREQ_0.1.asc'
    ranges, ceff, attenuation = load_alexis_data(file)
    idx_ceff = np.argmin(abs(ceff-ceff_max_error))
    #print(attenuation.shape, idx_ceff, attenuation.iloc[idx_ceff].values, ranges)
    ax_TL1.plot(ranges, attenuation.iloc[idx_ceff].values, color='tab:green', label='LP12')
    ax_TL1.legend(loc='lower left', bbox_to_anchor=(2.2, 0.), frameon=False, labelspacing=0.01, handlelength=0.4, handletextpad=0.05)
    
    ## Load 2d TL
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    file = output_dir + 'slice_no_{no:.0f}/tloss_2d.pe'
    simu_id, simu_slice_no, input_wind, input_f0, output = load_one_id(l_TL_1[0], model, dataset)
    file = file.format(no=simu_slice_no)
    TL2d = pd.read_csv(file, header=None, delim_whitespace=True)
    TL2d.columns = ['r', 'z', 'TL_real', 'TL_imag']
    r, z = TL2d.r.unique(), TL2d.z.unique()
    subsample = 5
    r_subsampled = r[::subsample]
    TL2d = TL2d.loc[TL2d.r.isin(r_subsampled)]
    r, z = TL2d.r.unique(), TL2d.z.unique()
    TL2d['TL'] = (np.sqrt(TL2d['TL_real']**2 + TL2d['TL_imag']**2))
    surface = TL2d.loc[(abs(TL2d.z-0.) == abs(TL2d.z-0.).min()), 'TL']
    f = interpolate.interp1d(r, surface, fill_value='extrapolate')
    TL2d.TL /= f(1)
    TL2d.TL = 20. * np.log(TL2d.TL)
    
    ## Interpolate TL2d to plot on regular cartesian grid
    r_i, z_i = np.linspace(r.min(), r.max(), nb_r), np.linspace(z.min(), z.max(), nb_z)
    grid_range, grid_z = np.meshgrid(r_i, z_i)
    points = np.c_[TL2d.r.values, TL2d.z.values]
    grid_TL2d = griddata(points, TL2d.TL.values, (grid_range, grid_z), method='linear')
    
    #plt.close('all')
    ax_TL2d1.pcolormesh(grid_range, grid_z, grid_TL2d, cmap=cmap, vmin=-150., vmax=-50., shading='auto')
    ax_TL2d1.set_ylabel('Altitude (km)')
    ax_TL2d1.set_ylim([0., 50.])
    plt.setp(ax_TL2d1, xticks=[])
    
    fig.align_ylabels([ax_TL1, ax_TL2d1, ax_winds1])
    
    ## Get simulation with largest error and plot
    iax += 1
    ax_winds2 = fig.add_subplot(grid[0, 1])
    ax_TL2d2 = fig.add_subplot(grid[1:3, 1])
    ax_TL2 = fig.add_subplot(grid[3, 1])
    axs = [ax_winds2, ax_TL2]
    
    f0 = 2.
    #ceff_max_error = ceffs_high[ceff_pick_for_error][errors_Alexis_high[ceff_pick_for_error].argmax()]
    #ceff_max_error = 1.1
    ceff_max_error = ceffs_high[error_Alexis_high>0.8][error_Alexis_high[error_Alexis_high>0.8].argmax()]
    simu_id = all_errors.sort_values(by='rmse_err'); simu_id = simu_id.loc[(abs(simu_id['f0s']-f0) < 0.25)]; simu_id = simu_id.loc[abs(simu_id[ceff_pick_for_error]-ceff_max_error) < 0.01]; simu_id = simu_id.loc[simu_id['veff-5-20'] == simu_id['veff-5-20'].max(), 'simu_id'].iloc[-1]; id, simu_slice_no, input_wind, input_f0, output, dataset = find_dataset_id_from_simu_id(model, simu_id)
    l_TL_2 = [id]
    #print('->',simu_id, simu_slice_no)
    compare_TL(l_TL_2, model, figure_dir, additional_homogeneous=[], additional_f0s=[], dataset=dataset, show_predictions=True, axs=axs, legend=False,  bbox_to_anchor_winds=(0.5, 1.05), label_predicted='ML')
    ax_winds2.set_title('')
    ax_winds2.text(-0.05, 1.2, alphabet[iax]+')', ha='right', va='top', transform=ax_winds2.transAxes, fontsize=15., fontweight='bold')
    ax_winds2.set_ylabel('')
    plt.setp(ax_winds2, yticks=[])
    
    file = './data_alexis/ATTEN_FREQ_2.asc'
    ranges, ceff, attenuation = load_alexis_data(file)
    idx_ceff = np.argmin(abs(ceff-ceff_max_error))
    ax_TL2.plot(ranges, attenuation.iloc[idx_ceff].values, color='tab:green')
    
    ## Load 2d TL
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    file = output_dir + 'slice_no_{no:.0f}/tloss_2d.pe'
    simu_id, simu_slice_no, input_wind, input_f0, output = load_one_id(l_TL_2[0], model, dataset)
    file = file.format(no=simu_slice_no)
    TL2d = pd.read_csv(file, header=None, delim_whitespace=True)
    TL2d.columns = ['r', 'z', 'TL_real', 'TL_imag']
    r, z = TL2d.r.unique(), TL2d.z.unique()
    r_subsampled = r[::subsample]
    TL2d = TL2d.loc[TL2d.r.isin(r_subsampled)]
    r, z = TL2d.r.unique(), TL2d.z.unique()
    TL2d['TL'] = (np.sqrt(TL2d['TL_real']**2 + TL2d['TL_imag']**2))
    surface = TL2d.loc[(abs(TL2d.z-0.) == abs(TL2d.z-0.).min()), 'TL']
    f = interpolate.interp1d(r, surface, fill_value='extrapolate')
    TL2d.TL /= f(1)
    TL2d.TL = 20. * np.log(TL2d.TL)
    
    ## Interpolate TL2d to plot on regular cartesian grid
    r_i, z_i = np.linspace(r.min(), r.max(), nb_r), np.linspace(z.min(), z.max(), nb_z)
    grid_range, grid_z = np.meshgrid(r_i, z_i)
    points = np.c_[TL2d.r.values, TL2d.z.values]
    grid_TL2d = griddata(points, TL2d.TL.values, (grid_range, grid_z), method='linear')
    
    #plt.close('all')
    ax_TL2d2.pcolormesh(grid_range, grid_z, grid_TL2d, cmap=cmap, vmin=-150., vmax=-50., shading='auto')
    ax_TL2d2.set_ylabel('Altitude (km)')
    ax_TL2d2.set_ylim([0., 50.])
    ax_TL2d2.set_ylabel('')
    plt.setp(ax_TL2d2, xticks=[], yticks=[])
    ax_TL2.set_ylabel('')
    plt.setp(ax_TL2, yticks=[])
    
    fig.subplots_adjust(left=0.15, right=0.82)
    
    plt.savefig(figure_dir + 'comparison_2dTL{ext_name_file}.pdf'.format(ext_name_file=ext_name_file))

def plot_errors_interpolated(ceff_errors, simus, ceffs, error_Alexis, error_ML,
                            xlim=[0.85, 1.2], ylim=[0., 50.], axs=None, plot_legend=False, plot_layers=True):

    """
    Plot ML and Alexis' RMSE error vs ceff in different atmospheric layer for a given frequency
    """
    
    print('plot_errors_interpolated')
    
    ## Find non nan values in Alexis' data
    good_idx = np.where(~np.isnan(error_Alexis))
    
    ## Setup figure
    new_figure = False
    if axs == None:
        new_figure = True
        fig, axs = plt.subplots(nrows=len(ceff_errors), ncols=1, sharex=True, sharey=True)
    
    col_labels = {'1-15': 'troposphere', '5-20': 'troposphere', '20-50': 'stratosphere', '35-60': 'stratosphere', '50-100': 'thermosphere', '80-120': 'thermosphere'}
    colors = sns.diverging_palette(220, 20)
    colors_alexis = sns.color_palette("crest", 5)
    
    #key = [key for key in ceffs.keys()][0]
    #index = np.arange(ceffs[key].size) + 0.3
    index = np.arange(ceffs.size) + 0.3
    bar_width = 0.2
    y_offset = 0.
    for iceff, ceff_unknown in enumerate(ceff_errors):
    
        layer = ceff_unknown.replace('veff-', '')
        
        #ceffs, distances, grid_range, grid_ceff, grid_TL_loc, grid_TL_true_loc = \
        #    get_veff_one_layer(model, TLs, TLs_true, simus, ceff_unknown, nb_ceff, interpolation)
        #ceffs = np.linspace(simus[ceff_unknown].min(), simus[ceff_unknown].max(), nb_ceff)
        
        #error_Alexis = errors_Alexis[ceff_unknown]
        #error_ML = errors_ML[ceff_unknown]

        label_alexis = '_nolegend_'
        label_ML = '_nolegend_'
        if iceff == 0:
            label_alexis = 'LP12'
            label_ML = 'ML'
          
        ## 
        f = interpolate.interp1d(simus['veff-40-50'].values, simus[ceff_unknown].values, fill_value='extrapolate')
        ceffs_loc = f(ceffs)
        idx_ceffs = np.argsort(ceffs_loc)
        
        #km0, km1 = ceff_profile_unknown.split('-')[1], ceff_profile_unknown.split('-')[2]
        #ax_ceff.scatter(simus[ceff_profile_unknown], simus[ceff_unknown], s=10, color='black')
        #ax_ceff.set_ylim([ceffs.min(), ceffs.max()])
        #if ylim:
        #    ax_ceff.set_ylim(ylim)
        #ax_ceff.invert_yaxis()
        #ax_ceff.set_title('$C_{{eff}}$ {km0} - {km1} km'.format(km0=km0, km1=km1))
        
        ## 
        axs[iceff].plot(ceffs_loc[idx_ceffs], error_Alexis[idx_ceffs], color=colors[-1], label=label_alexis); 
        axs[iceff].fill_between(ceffs_loc[idx_ceffs], error_Alexis[idx_ceffs], color=colors[-1], alpha=0.3)
        axs[iceff].plot(ceffs_loc[idx_ceffs], error_ML[idx_ceffs], color=colors[0], label=label_ML); 
        axs[iceff].fill_between(ceffs_loc[idx_ceffs], error_ML[idx_ceffs   ], color=colors[0], alpha=0.3)
        if plot_layers:
            axs[iceff].text(0.99, 0.97, col_labels[layer], ha='right', va='top', transform=axs[iceff].transAxes, fontsize=12.)
        #axs[iceff].bar(index, error_Alexis, bar_width, color=colors[iceff], label=label_alexis)
        #axs[iceff].bar(index, error_ML, bar_width, bottom=error_Alexis, color=colors_alexis[iceff], label=label_alexis)
        #y_offset = y_offset + data[row]
        
    if plot_legend:
        axs[0].legend(frameon=False, loc='upper center')
    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim)
    axs[-1].set_xlabel('$c_{eff}$')
    axs[-1].set_ylabel('RMSE (db)')
    
    if new_figure:
        plt.savefig('./test_figure_errors.png')
        plt.close('all')
    
def get_errors_interpolated(ceff_unknown, model, TLs, TLs_true, simus, nb_ceff, interpolation, 
                            DD, CEFF, attenuation):

    """
    Retrieve ML and Alexis' RMSE error vs ceff in different atmospheric layer for a given frequency
    """
    
    """
    errors_ML, errors_Alexis, ceffs = {}, {}, {}
    for iceff, ceff_unknown in enumerate(ceff_errors):
    
        layer = ceff_unknown.replace('veff-', '')
        
        ceffs[ceff_unknown], distances, grid_range, grid_ceff, grid_TL_loc, grid_TL_true_loc = \
            get_veff_one_layer(model, TLs, TLs_true, simus, ceff_unknown, nb_ceff, interpolation)
        #ceffs[ceff_unknown], distances, grid_range, grid_ceff, grid_TL_loc, grid_TL_true_loc = utils_figures.get_veff_one_layer(model, TLs, TLs_true, simus, ceff_unknown, nb_ceff, interpolation)
        points = np.c_[DD.ravel(), CEFF.ravel()]
        grid_TL_alexis = griddata(points, attenuation.values.ravel(), (grid_range, grid_ceff), method=interpolation)
        errors_Alexis[ceff_unknown] = np.sqrt(np.mean((grid_TL_alexis - grid_TL_true_loc)**2, axis=1))
        errors_ML[ceff_unknown] = np.sqrt(np.mean((grid_TL_loc - grid_TL_true_loc)**2, axis=1))
    """
    
    #errors_ML, errors_Alexis, ceffs = {}, {}, {}
    #for iceff, ceff_unknown in enumerate(ceff_errors):
    #
    #    layer = ceff_unknown.replace('veff-', '')
    #print('dasd')
    ceffs, _, grid_range, grid_ceff, grid_TL_loc, grid_TL_true_loc = \
        get_veff_one_layer(model, TLs, TLs_true, simus, ceff_unknown, nb_ceff, interpolation)
    points = np.c_[DD.ravel(), CEFF.ravel()]
    grid_TL_alexis = griddata(points, attenuation.values.ravel(), (grid_range, grid_ceff), method=interpolation)
    errors_Alexis = np.sqrt(np.mean((grid_TL_alexis - grid_TL_true_loc)**2, axis=1))
    errors_ML     = np.sqrt(np.mean((grid_TL_loc - grid_TL_true_loc)**2, axis=1))
    
    return errors_Alexis, errors_ML, ceffs
    
def get_veff_one_layer(model, TLs, TLs_true, simus, ceff_unknown, nb_ceff, interpolation):
    
    """
    Return interpolated TLs
    """
    
    ceffs = np.linspace(simus[ceff_unknown].min(), simus[ceff_unknown].max(), nb_ceff)
    distances = model.dataset.distances
    
    grid_range, grid_ceff = np.meshgrid(distances, ceffs)
    
    points = np.c_[np.tile(distances, TLs.shape[0]), np.repeat(simus[ceff_unknown].values, TLs.shape[1])]
    values = TLs.ravel()

    grid_TL = griddata(points, values, (grid_range, grid_ceff), method=interpolation)
    if TLs_true.size > 0:
        grid_TL_true = griddata(points, TLs_true.ravel(), (grid_range, grid_ceff), method=interpolation)
    
    return ceffs, distances, grid_range, grid_ceff, grid_TL, grid_TL_true
    
def plot_map_TL_veff(dir_figures, simus, TLs, model, f0, ceff_unknown='veff-40-50', ceff_profile_unknowns=['veff-5-20'], 
                     ceff_errors=['veff-5-20', 'veff-20-50', 'veff-50-100'],
                     nb_ceff=100, interpolation='cubic', vmin=-100., vmax=0., TLs_true=np.array([]), axs=None,
                     plot_ceff=False, ranges=pd.DataFrame(), ceff=pd.DataFrame(), attenuation=pd.DataFrame(), colorbar=True, ylim=[],
                     plot_legend=False, error_Alexis=np.array([]), error_ML=np.array([]), plot_median=True, linestyle_median='-',
                     levels_LP12=[-80., -70., -60., -50., -40.], xlim_ceff=[0.8, 1.125]):
    
    """
    Plot ML and PE predictions along with contours from Alexis' model
    """
    
    from scipy.interpolate import griddata
    
    """
    data = model.dataset.properties.loc[model.dataset.properties.simu_id.isin(simu_ids_low)]; simus = data.groupby('simu_id').mean().reset_index(); ceffs = np.linspace(simus[ceff_pick_for_error].min(), simus[ceff_pick_for_error].max(), nb_ceff)
    """
    
    ## Interpolate values on a regular grid
    print('Interpolation map')
    ceffs, distances, grid_range, grid_ceff, grid_TL, grid_TL_true = get_veff_one_layer(model, TLs, TLs_true, simus, ceff_unknown, nb_ceff, interpolation)
    
    ## Setup new figure
    new_figure = False
    if axs == None:
        new_figure = True
        fig = plt.figure()
        if TLs_true.size > 0:
            nb_plots = 8
            #if plot_ceff:
            #    nb_plots += 1
            if error_Alexis.size > 0:
                nb_plots += 1
            if plot_median:
                nb_plots += 1
            grid = fig.add_gridspec(1, nb_plots)
            ax = fig.add_subplot(grid[0, :3])
            #fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True)
            #ax = axs[0]
            #fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True); plt.pcolormesh(TLs_true);
            ax_true = fig.add_subplot(grid[0, 3:6], sharex=ax)
            ax_last = ax_true
            if plot_ceff:
                ax_ceff = fig.add_subplot(grid[0, 6])
            if error_Alexis.size > 0:
                ax_error = fig.add_subplot(grid[0, 7])
            if plot_median:
                ax_medianTLceff = fig.add_subplot(grid[0, -1])
                
        else:
            grid = fig.add_gridspec(1, 5)
            #fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
            ax = fig.add_subplot(grid[0, :3])
            ax_last = ax
            id_ax = 3
            if plot_ceff:
                ax_ceff = fig.add_subplot(grid[0, id_ax])
                id_ax += 1
            if plot_median:
                ax_medianTLceff = fig.add_subplot(grid[0, id_ax])
    
    else:
        ax = axs[0]
        ax_true = axs[1]
        ax_last = ax_true
        id_ax = 2
        if plot_ceff:
            ax_ceff = axs[id_ax]
            id_ax += 1
            
        if error_Alexis.size > 0:
            ax_error = axs[id_ax]
            id_ax += 1
        
        if plot_median:
            ax_medianTLceff = axs[id_ax]
        
    print('Plotting map')
        
    km0, km1 = ceff_unknown.split('-')[1], ceff_unknown.split('-')[2]
    
    #cmap = sns.color_palette("flare", as_cmap=True)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    sc = ax.pcolormesh(distances, ceffs, grid_TL, shading='auto', vmin=vmin, vmax=vmax, cmap=cmap); 
    ax.minorticks_on()
    
    ## Plot Alexis' data as contours
    if attenuation.shape[0] > 0:
        DD, CEFF = np.meshgrid(ranges, ceff)
        #error_Alexis[np.isnan(error_Alexis)] = 0.
        a = 0.7
        cmap_contour = cmap(np.arange(cmap.N))
        cmap_contour[:,0:3] *= a 
        cmap_contour = ListedColormap(cmap_contour)
        CS = ax_true.contour(DD, CEFF, attenuation.values, levels=levels_LP12, vmin=vmin, vmax=vmax, cmap=cmap_contour)
        ax_true.clabel(CS, inline=True, fontsize=10, colors='black')

        if plot_legend:
            label = 'LP12 isocontour'
            CS.collections[-1].set_label(label)
            ax_true.legend(loc='lower left', bbox_to_anchor=(-0.1, 0.95), frameon=False, labelspacing=0.01, handlelength=0.4, handletextpad=0.05)

    ax.set_xlim([distances.min(), distances.max()])
    ax.set_ylim([ceffs.min(), ceffs.max()])
    if ylim:
        ax.set_ylim(ylim)
    ax.set_title('ML predictions at {f0} Hz\n'.format(f0=np.round(f0,2)))
    ax.set_xlabel('Range (km)')
    ax.set_ylabel('$\overline{{c}}_{{eff}}$ {km0} - {km1} km'.format(km0=km0, km1=km1))
    ax.invert_yaxis()
    if TLs_true.size > 0:
        ax_true.pcolormesh(distances, ceffs, grid_TL_true, shading='auto', vmin=vmin, vmax=vmax, cmap=cmap);
        ax_true.set_title('{f0:.2g} Hz\nPE simulations\n'.format(f0=f0))
        if ylim:
            ax_true.set_ylim(ylim)
        ax_true.invert_yaxis()
        #plt.setp(ax_true, yticks=[])
        ax_true.tick_params(labelleft=False)
        ax_true.minorticks_on()
        
    if colorbar:
        if error_Alexis.size > 0:
            #plt.setp(ax_true, xticks=[])
            ax_true.tick_params(labelleft=False)
            axins = inset_axes(ax_last, width="80%", height="3.5%", loc='lower left', bbox_to_anchor=(0.1, -0.05, 1, 1.), bbox_transform=ax_last.transAxes, borderpad=0)
            axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
            cbar = plt.colorbar(sc, cax=axins, orientation='horizontal', extend='both')
            cbar.ax.set_xlabel('TL (db)', labelpad=4.) 
            
        else:
            axins = inset_axes(ax_last, width="4%", height="100%", loc='lower left', bbox_to_anchor=(1.05, 0., 1, 1.), bbox_transform=ax_last.transAxes, borderpad=0)
            axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
            cbar = plt.colorbar(sc, cax=axins, orientation='vertical', extend='both')
            cbar.ax.set_ylabel('TL (db)', rotation=270., labelpad=10.) 
    
    ## Plot other ceff along ceff strato
    if plot_ceff:
        simus_loc = simus.groupby('simu_id').first().reset_index()
        colors_add_ceff = sns.diverging_palette(220, 20, n=2)
        for iceff, ceff_profile_unknown in enumerate(ceff_profile_unknowns[::-1]):
            km0, km1 = ceff_profile_unknown.split('-')[1], ceff_profile_unknown.split('-')[2]
            ax_ceff.scatter(simus_loc[ceff_profile_unknown], simus_loc[ceff_unknown], s=5, alpha=0.5, color=colors_add_ceff[iceff], 
                label='{km0}-{km1} km'.format(km0=km0, km1=km1))
            #m, b = np.polyfit(simus_loc[ceff_profile_unknown].values, simus_loc[ceff_unknown].values, 1)
            #ax_ceff.plot(simus_loc[ceff_profile_unknown].values, m*simus_loc[ceff_profile_unknown].values + b, color=colors_add_ceff[iceff])
        
        ##
        ax_ceff.axvline(0.95, linestyle='--', color='black', alpha=0.6)
        
        #ax_ceff.set_ylim([ceffs.min(), ceffs.max()])
        ax_ceff.set_xlabel('$c_{eff}$')
        if plot_legend:
            ax_ceff.legend(loc='lower left', bbox_to_anchor=(-0.9, 1.1), frameon=False, labelspacing=0.01, handletextpad=0.01)
        
        if ylim:
            ax_ceff.set_ylim(ylim)
        ax_ceff.invert_yaxis()
        ax_ceff.set_xlim(xlim_ceff)
        #plt.setp(ax_ceff, yticks=[])
        ax_ceff.tick_params(labelleft=False)
    
    ## Plot error with ceff
    if error_Alexis.size > 0:
        colors_errors = sns.color_palette("flare", 2)
        ax_error.plot(error_Alexis, ceffs, color=colors_errors[0], label='LP12'); 
        ax_error.fill_betweenx(ceffs, error_Alexis, error_Alexis*0., color=colors_errors[0], alpha=0.3)
        ax_error.plot(error_ML, ceffs, color=colors_errors[1], label='ML'); 
        ax_error.fill_betweenx(ceffs, error_ML, error_ML*0., color=colors_errors[1], alpha=0.3)
        
        ax_error.set_ylim([ceffs.min(), ceffs.max()])
        if plot_legend:
            ax_error.legend(loc='lower left', bbox_to_anchor=(0.1, 1.1), frameon=False, labelspacing=0.01, handlelength=0.2, handletextpad=0.05)
        if ylim:
            ax_error.set_ylim(ylim)
        ax_error.invert_yaxis()
        ax_error.set_xlim([0., 50.])
        ax_error.set_xlabel('error\n(db)')
        #plt.setp(ax_error, yticks=[])
        ax_error.tick_params(labelleft=False)
        ax_error.minorticks_on()
    
    ## Plot low frequency median TL
    if plot_median:
        colors_errors = sns.color_palette("flare", 2)
        std_TL_ML = grid_TL.std(axis=1)
        std_TL = attenuation.std(axis=1)
        #print(ceffs.shape, std_TL_ML.shape, std_TL.shape, np.median(grid_TL, axis=1).shape, np.median(attenuation, axis=1).shape)
        ax_medianTLceff.plot(np.median(grid_TL, axis=1), ceffs, color=colors_errors[1], linewidth=2., linestyle=linestyle_median)
        ax_medianTLceff.plot(np.median(grid_TL_true, axis=1), ceffs, label='PE', color='black', linewidth=0.5, linestyle=linestyle_median)
        ax_medianTLceff.plot(np.median(attenuation, axis=1), ceff, color=colors_errors[0], linewidth=2., linestyle=linestyle_median)
        ax_medianTLceff.fill_betweenx(ceff, np.median(attenuation, axis=1)-std_TL, np.median(attenuation, axis=1)+std_TL, alpha=0.3, color=colors_errors[0])
        ax_medianTLceff.fill_betweenx(ceffs, np.median(grid_TL, axis=1)-std_TL_ML, np.median(grid_TL, axis=1)+std_TL_ML, color=colors_errors[1], alpha=0.3)
        ax_medianTLceff.set_ylim(ylim)
        ax_medianTLceff.invert_yaxis()
        #plt.setp(ax_medianTLceff, yticks=[])
        ax_medianTLceff.tick_params(labelleft=False)
        ax_medianTLceff.set_xlim([-130., -30.])
        ax_medianTLceff.set_xlabel('TL\n(db)')
        ax_medianTLceff.minorticks_on()
        if plot_legend:
            ax_medianTLceff.legend(loc='lower left', bbox_to_anchor=(0., 1.1), frameon=False, labelspacing=0.01, handlelength=0.2, handletextpad=0.05)
        
    if new_figure:
        fig.subplots_adjust(bottom=0.15)
        plt.savefig(dir_figures + 'comparison_veff_TL_{f0}_{veff}.pdf'.format(f0=f0, veff=ceff_unknown))
    
def load_alexis_data(file):
    
    """
    Load Alexis' TL data in a Dataframe
    """
    
    ranges = pd.read_csv('/staff/quentin/Documents/Projects/ML_attenuation_prediction/data_alexis/RANGE.asc', header=None, delim_whitespace=True)
    ranges = ranges.iloc[0].values
    
    ceff = pd.read_csv('/staff/quentin/Documents/Projects/ML_attenuation_prediction/data_alexis/CSIM.asc', header=None, delim_whitespace=True)
    ceff = ceff.iloc[0].values
    
    attenuation = pd.read_csv(file, header=None, delim_whitespace=True)
    for ii in range(attenuation.shape[0]):
        #A = 20log10(P/P0)
        attenuation.loc[attenuation.index==ii] = 10**(attenuation.loc[attenuation.index==ii]/20.)
        #print(attenuation.loc[attenuation.index==ii].values.T.shape, ranges.shape)
        f = interpolate.interp1d(ranges, attenuation.loc[attenuation.index==ii].values.T[:,0], fill_value='extrapolate')
        attenuation.loc[attenuation.index==ii] /= f(1)
        attenuation.loc[attenuation.index==ii] = 20. * np.log10( attenuation.loc[attenuation.index==ii] )
    
    return ranges, ceff, attenuation
    
def plot_composite_sensitivity_ML(figure_dir, l_files_all, name_types, one_dataset, output_dir, xlim=[0, 60], ylim=[0.5, 1]):
    
    """
    Plot convergence RMSE validation for different ML parameters
    """

    nrows, ncols = 2, 2
    fig = plt.figure()
    grid = fig.add_gridspec(nrows, ncols)
    alphabet = string.ascii_lowercase
            
    for itype, type in enumerate(l_files_all):
        ax = fig.add_subplot(grid[itype//ncols, np.mod(itype, nrows)])
        l_files = l_files_all[type]
        params = []
        for file in l_files:
            split_file = file.split('_')[1:]
            for no_split, isplit in enumerate(split_file):
                if isplit[0] == type:
                    if type == 'd':
                        params.append( isplit[1:] + 'x' + split_file[no_split+1][1:] )
                    else:
                        params.append( isplit[1:].replace('/','') )
        plot_loss_sensitivity(figure_dir, type, l_files, name_types, params, one_dataset, output_dir, ax=ax)
    
        ax.text(-0.1, 1.015, alphabet[itype]+')', ha='right', va='bottom', transform=ax.transAxes, fontsize=15., fontweight='bold')
    
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('')
        ax.set_ylabel('')
        
    ax.set_xlabel('Epochs')
    ax.set_ylabel('RMSE')
    
    plt.savefig(figure_dir + 'comparison_sensitivity.pdf')
    plt.close('all')
    
def plot_loss_sensitivity(figure_dir, type, l_files, name_types, params, one_dataset, output_dir, ax=None):

    """
    Plot validation RMSE vs epochs for various choices of batch size
    """

    new_figure = False
    if ax == None:  
        new_figure = True
        fig, ax = plt.subplots(nrows=1, ncols=1)
    
    cmap = sns.color_palette("rocket", n_colors=len(params))
    for ibatch, (batch_size, file) in enumerate(zip(params, l_files)):
        
        model = build_machine_learning.build_NN(output_dir, one_dataset)
        model.load_model(file)
       
        Nepochs = len(model.history.history['val_mse'])
        epochs = np.arange(1, Nepochs+1)
       
        ax.plot(epochs, np.sqrt(model.history.history['val_mse']), label=str(batch_size), color=cmap[ibatch], linewidth=2.)
    
    ax.set_xlim([epochs.min(), epochs.max()])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('RMSE validation')
    ax.legend(frameon=False, ncol=2, title=name_types[type])
    
    if new_figure:
        plt.savefig(figure_dir + 'comparison_batch_size.pdf')
        plt.close('all')
    
def plot_ceff_maps(model, figure_dir, max_veff_tropo=0.97, max_veff_thermo=1.04, min_veff_strato=1.,
                   vmin=-80., vmax=-30., interpolation='linear',
                   ceff_unknown='veff-40-50', ylim=[0.8, 1.2], ceff_profile_unknowns=['veff-5-20', 'veff-50-100'],
                   tropo_col='veff-5-20', strato_col='veff-40-50', thermo_col='veff-50-100',
                   nb_ceff=100, simu_ids_high=np.array([]), TLs_high=np.array([]), TLs_true_high=np.array([]), error_Alexis_high=np.array([]), 
                   error_ML_high=np.array([]), simu_ids_low=np.array([]), TLs_low=np.array([]), TLs_true_low=np.array([]), error_Alexis_low=np.array([]), error_ML_low=np.array([])):
    
    """
    Plot Figure paper showing comparisons between ML, PE, and Alexis' model at two different frequencies
    """
    
    print('Start plotting maps')
    
    #simu_ids_high, TLs_high, TLs_true_high, error_Alexis_high, error_ML_high, simu_ids_low, TLs_low, TLs_true_low, error_Alexis_low, error_ML_low = utils_figures.plot_ceff_maps(model, figure_dir, max_veff_tropo=0.97, max_veff_thermo=1.04, vmin=-80., vmax=-20., interpolation='linear', nb_ceff=100)
    #_ = utils_figures.plot_ceff_maps(model, figure_dir, max_veff_tropo=0.97, max_veff_thermo=1.04, vmin=-80., vmax=-20., interpolation='linear', nb_ceff=100, error_Alexis_high=error_Alexis_high, error_ML_high=error_ML_high, error_Alexis_low=error_Alexis_low, error_ML_low=error_ML_low, simu_ids_high=simu_ids_high, TLs_high=TLs_high, TLs_true_high=TLs_true_high, simu_ids_low=simu_ids_low, TLs_low=TLs_low, TLs_true_low=TLs_true_low)
    
    ## Setup figure
    fig = plt.figure(figsize=(6.4, 6.8)); 
    nb_total_cols = 9
    plot_ceff = True
    if not ceff_profile_unknowns:
        nb_total_cols -= 1
        plot_ceff = False
    grid = fig.add_gridspec(4, nb_total_cols)
    alphabet = string.ascii_lowercase
    iax = -1
    h_map = 2
    size_map = 3
    
    ## Plot low frequencies
    ax_ML1 = fig.add_subplot(grid[:h_map, :size_map]); ax_PE1 = fig.add_subplot(grid[:h_map, size_map:2*size_map]);
    id_ax = 0
    if plot_ceff:
        ax_ceff1 = fig.add_subplot(grid[:h_map, 2*size_map]);
        id_ax += 1
    ax_error1 = fig.add_subplot(grid[:h_map, 2*size_map+id_ax]); 
    id_ax += 1
    ax_medianTLceff1 = fig.add_subplot(grid[:h_map, 2*size_map+id_ax]); 
    if plot_ceff:
        axs = [ax_ML1, ax_PE1, ax_ceff1, ax_error1, ax_medianTLceff1]
    else:
        axs = [ax_ML1, ax_PE1, ax_error1, ax_medianTLceff1]
    f0 = 0.15
    deviation_from_f0 = 0.05
    
    print('Load Alexis 0.1 Hz')
    file = './data_alexis/ATTEN_FREQ_0.1.asc'
    ranges_low, ceff_low, attenuation_low = load_alexis_data(file)
    DD, CEFF = np.meshgrid(ranges_low, ceff_low)
    if TLs_true_low.size == 0:
        print('test')
        simu_ids1, TLs, TLs_true, _ = compute_map_TL_veff(model, f0, deviation_from_f0=deviation_from_f0, max_veff_tropo=max_veff_tropo, max_veff_thermo=max_veff_thermo, tropo_col=tropo_col, strato_col=strato_col, thermo_col=thermo_col, min_veff_strato=min_veff_strato)
    else:
        simu_ids1, TLs, TLs_true = simu_ids_low, TLs_low, TLs_true_low
    data = model.dataset.properties.loc[model.dataset.properties.simu_id.isin(simu_ids1)]
    simus = data.groupby('simu_id').mean().reset_index()
    #simus['veff-q'] = pd.cut(simus['veff-1-15'], bins=CEFF[:,0])
    #simus['veff-q-1-15-val'] = simus.groupby('veff-q')['veff-1-15'].transform('median')
    #simus['veff-q-35-60-val'] = simus.groupby('veff-q')['veff-35-60'].transform('median')
    
    print('Plotting 0.1 Hz')
    if error_Alexis_low.size == 0:
        print('test2')
        error_Alexis_low, error_ML_low, ceffs_high = get_errors_interpolated(ceff_unknown, model, TLs, TLs_true, simus, nb_ceff, interpolation, DD, CEFF, attenuation_low)
    plot_map_TL_veff(figure_dir, simus, TLs, model, f0, ceff_unknown=ceff_unknown, ceff_profile_unknowns=ceff_profile_unknowns, 
                                   nb_ceff=nb_ceff, vmin=vmin, vmax=vmax, interpolation=interpolation, TLs_true=TLs_true, axs=axs, plot_ceff=plot_ceff, 
                                   ranges=ranges_low, ceff=ceff_low, attenuation=attenuation_low, ylim=ylim, error_Alexis=error_Alexis_low, error_ML=error_ML_low, plot_legend=True, colorbar=False, levels_LP12=[-50., -45., -40.])
    
    #plot_map_TL_veff(figure_dir, simu_ids, TLs, model, f0, ceff_unknown='veff-40-50', ceff_profile_unknown='veff-50-100', nb_ceff=nb_ceff, vmin=vmin, vmax=vmax, interpolation=interpolation, TLs_true=TLs_true, axs=axs, plot_ceff=False, ranges=ranges, ceff=ceff, attenuation=attenuation, ylim=[0.8, 1.2], error_Alexis=errors_Alexis_high, error_ML=errors_ML_high)
    ax_ML1.set_title('ML predictions\n')
    iax += 1
    ax_ML1.text(-0.1, 1.015, alphabet[iax]+')', ha='right', va='bottom', transform=ax_ML1.transAxes, fontsize=15., fontweight='bold')
    iax += 1
    ax_PE1.text(-0.075, 1.015, alphabet[iax]+')', ha='right', va='bottom', transform=ax_PE1.transAxes, fontsize=15., fontweight='bold')
             
    if plot_ceff:
        iax += 1
        ax_ceff1.text(-0.0, 1.015, alphabet[iax]+')', ha='right', va='bottom', transform=ax_ceff1.transAxes, fontsize=15., fontweight='bold')
               
    iax += 1
    ax_error1.text(-0.0, 1.015, alphabet[iax]+')', ha='right', va='bottom', transform=ax_error1.transAxes, fontsize=15., fontweight='bold')
    
    iax += 1
    ax_medianTLceff1.text(-0.0, 1.015, alphabet[iax]+')', ha='right', va='bottom', transform=ax_medianTLceff1.transAxes, fontsize=15., fontweight='bold')
          
    #plt.setp(ax_ML1, xticks=[])
    ax_ML1.tick_params(labelbottom=False) 
    ax_ML1.set_xlabel('')
    #plt.setp(ax_PE1, xticks=[])
    ax_PE1.tick_params(labelbottom=False) 
    ax_PE1.set_xlabel('')
    
    #plt.setp(ax_medianTLceff1, xticks=[])
    ax_medianTLceff1.tick_params(labelbottom=False)
    ax_medianTLceff1.set_xlabel('')
    
    if plot_ceff:
        #plt.setp(ax_ceff1, xticks=[])
        ax_ceff1.set_xlabel('')
        ax_ceff1.tick_params(labelbottom=False)
    #plt.setp(ax_error1, xticks=[])
    ax_error1.tick_params(labelbottom=False)
    ax_error1.set_xlabel('')
    
    print('before 2 Hz')
    
    ## Plot high frequencies
    ax_ML2 = fig.add_subplot(grid[h_map:2*h_map, :size_map]); 
    ax_PE2 = fig.add_subplot(grid[h_map:2*h_map, size_map:2*size_map]); 
    id_ax = 0
    if plot_ceff:
        ax_ceff2 = fig.add_subplot(grid[h_map:2*h_map, 2*size_map]); 
        id_ax += 1
    ax_error2 = fig.add_subplot(grid[h_map:2*h_map, 2*size_map+id_ax]);  
    id_ax += 1
    ax_medianTLceff2 = fig.add_subplot(grid[h_map:2*h_map, 2*size_map+id_ax]);  
    if plot_ceff:
        axs = [ax_ML2, ax_PE2, ax_ceff2, ax_error2, ax_medianTLceff2]
    else:
        axs = [ax_ML2, ax_PE2, ax_error2, ax_medianTLceff2]
    #ax_ML2 = fig.add_subplot(grid[1, 0]); ax_PE2 = fig.add_subplot(grid[1, 1]); axs = [ax_ML2, ax_PE2]
    f0 = 2.; deviation_from_f0 = 0.25
    
    print('Load Alexis 2 Hz')
    file = './data_alexis/ATTEN_FREQ_2.asc'; ranges_high, ceff, attenuation_high = load_alexis_data(file)
    DD, CEFF = np.meshgrid(ranges_high, ceff)
    if TLs_true_high.size == 0:
        simu_ids2, TLs2, TLs_true2, _ = compute_map_TL_veff(model, f0, deviation_from_f0=deviation_from_f0, max_veff_tropo=max_veff_tropo, max_veff_thermo=max_veff_thermo, tropo_col=tropo_col, strato_col=strato_col, thermo_col=thermo_col, min_veff_strato=min_veff_strato)
    else:
        simu_ids2, TLs2, TLs_true2 = simu_ids_high, TLs_high, TLs_true_high
    
    print('Plotting 2 Hz')
    data = model.dataset.properties.loc[model.dataset.properties.simu_id.isin(simu_ids2)]
    simus = data.groupby('simu_id').mean().reset_index()
    if error_Alexis_high.size == 0:
        error_Alexis_high, error_ML_high, _ = get_errors_interpolated(ceff_unknown, model, TLs2, TLs_true2, simus, nb_ceff, interpolation, DD, CEFF, attenuation_high)
    
    plot_map_TL_veff(figure_dir, simus, TLs2, model, f0, ceff_unknown=ceff_unknown, ceff_profile_unknowns=ceff_profile_unknowns, 
                     nb_ceff=nb_ceff, interpolation=interpolation, vmin=vmin, vmax=vmax, TLs_true=TLs_true2, axs=axs, plot_ceff=plot_ceff, 
                     ranges=ranges_high, ceff=ceff, attenuation=attenuation_high, colorbar=True, ylim=ylim, plot_legend=False, 
                     error_Alexis=error_Alexis_high, error_ML=error_ML_high, levels_LP12=[-80., -70., -60., -50., -40.])
    ax_ML2.set_title('')
    ax_PE2.set_title('2 Hz\n')
    #ax_medianTLceff2.set_xlabel('')

    iax += 1
    ax_ML2.text(-0.1, 1.015, alphabet[iax]+')', ha='right', va='bottom', transform=ax_ML2.transAxes, fontsize=15., fontweight='bold')
    iax += 1
    ax_PE2.text(-0.075, 1.015, alphabet[iax]+')', ha='right', va='bottom', transform=ax_PE2.transAxes, fontsize=15., fontweight='bold')
    ax_PE2.tick_params(labelbottom=False) 
               
    if plot_ceff:
        iax += 1
        ax_ceff2.text(-0.0, 1.015, alphabet[iax]+')', ha='right', va='bottom', transform=ax_ceff2.transAxes, fontsize=15., fontweight='bold')
               
    iax += 1
    ax_error2.text(-0.0, 1.015, alphabet[iax]+')', ha='right', va='bottom', transform=ax_error2.transAxes, fontsize=15., fontweight='bold')
               
    iax += 1
    ax_medianTLceff2.text(-0.0, 1.015, alphabet[iax]+')', ha='right', va='bottom', transform=ax_medianTLceff2.transAxes, fontsize=15., fontweight='bold')
      
    fig.align_ylabels(axs)
      
    ## Plot TL median vs range
    if False:
        
        plt.setp(ax_ML2, xticks=[])
        ax_ML2.set_xlabel('')
        plt.setp(ax_PE2, xticks=[])
        ax_PE2.set_xlabel('')
    
        ## Plot median TLs below each simulation type
        ax_TLmedianML = fig.add_subplot(grid[2*h_map:, :size_map]); ax_TLmedianPE = fig.add_subplot(grid[2*h_map:, size_map:2*size_map]); axs = [ax_TLmedianML, ax_TLmedianPE]
        distances = model.dataset.distances
        
        std_TL = TLs.std(axis=0)
        std_TL2 = TLs2.std(axis=0)
        ax_TLmedianML.plot(distances, np.median(TLs, axis=0), label='.15 Hz ML', color='violet', linewidth=2.)
        ax_TLmedianML.plot(distances, np.median(TLs2, axis=0), label='2.0 Hz ML', color='purple', linestyle=':', linewidth=2.)
        ax_TLmedianML.plot(distances, np.median(TLs_true, axis=0), label='.15 Hz PE', color='black')
        ax_TLmedianML.plot(distances, np.median(TLs_true2, axis=0), label='2.0 Hz PE', color='black', linestyle=':')
        ax_TLmedianML.fill_between(distances, np.median(TLs, axis=0)-std_TL, np.median(TLs, axis=0)+std_TL, alpha=0.3, color='violet')
        ax_TLmedianML.fill_between(distances, np.median(TLs2, axis=0)-std_TL2, np.median(TLs2, axis=0)+std_TL2, alpha=0.3, color='purple')
        
        ax_TLmedianML.set_xlabel('Range (km)')
        ax_TLmedianML.set_ylabel('TL (db)')
        ax_TLmedianML.set_xlim([np.min(distances), np.max(distances)])
        ax_TLmedianML.set_ylim([-100., 0.])
        ax_TLmedianML.legend(loc='lower left', bbox_to_anchor=(2.1, -0.1), ncol=2, frameon=False, labelspacing=0.04, handlelength=0.4, handletextpad=0.2, columnspacing=0.2)
        
        std_TL = attenuation_low.std(axis=0)
        std_TL2 = attenuation_high.std(axis=0)
        ax_TLmedianPE.plot(ranges_low, np.median(attenuation_low, axis=0), label='.15 Hz LP', color='lightcoral', linewidth=2.)
        ax_TLmedianPE.plot(ranges_high, np.median(attenuation_high, axis=0), label='2.0 Hz LP ', color='brown', linestyle=':', linewidth=2.)
        ax_TLmedianPE.fill_between(ranges_low, np.median(attenuation_low, axis=0)-std_TL, np.median(attenuation_low, axis=0)+std_TL, alpha=0.3, color='lightcoral')
        ax_TLmedianPE.fill_between(ranges_high, np.median(attenuation_high, axis=0)-std_TL2, np.median(attenuation_high, axis=0)+std_TL2, alpha=0.3, color='red')
        ax_TLmedianPE.plot(distances, np.median(TLs_true, axis=0), color='black')
        ax_TLmedianPE.plot(distances, np.median(TLs_true2, axis=0), color='black', linestyle=':')
        
        ax_TLmedianPE.set_xlabel('Range (km)')
        ax_TLmedianPE.set_xlim([np.min(ranges_low), np.max(ranges_low)])
        ax_TLmedianPE.set_ylim([-100., 0.])
        ax_TLmedianPE.legend(loc='lower left', bbox_to_anchor=(1.07, -0.35), ncol=2, frameon=False, labelspacing=0.04, handlelength=0.4, handletextpad=0.2, columnspacing=0.2)
        
        iax += 1
        ax_TLmedianML.text(-0.1, 1.015, alphabet[iax]+')', ha='right', va='bottom', transform=ax_TLmedianML.transAxes, fontsize=15., fontweight='bold')
        
        iax += 1
        ax_TLmedianPE.text(-0.075, 1.015, alphabet[iax]+')', ha='right', va='bottom', transform=ax_TLmedianPE.transAxes, fontsize=15., fontweight='bold')
        
        plt.setp(ax_TLmedianPE, yticks=[])
        
        fig.align_ylabels([ax_ML1, ax_ML2, ax_TLmedianML])
        
    fig.subplots_adjust(right=0.95, hspace=0.68, bottom=0.12)
               
    plt.savefig(figure_dir + 'comparison_TL_alexis_ML_{ceff}.pdf'.format(ceff=ceff_unknown))
    plt.close('all')

    return simu_ids2, TLs2, TLs_true2, error_Alexis_high, error_ML_high, simu_ids1, TLs, TLs_true, error_Alexis_low, error_ML_low
    
def optimize_encoding(one_dataset, range_n_components, encoders):

    """
    Compute MSE over the whole wind dataset for a different number of SVD or PCA components
    """

    ## Reshape wind stack for fitting
    shape_init    = one_dataset.wind_stack.shape
    shape_for_SVD = (shape_init[1], shape_init[0]*shape_init[2])
    reshaped_winds = one_dataset.wind_stack.transpose(1,0,2)
    reshaped_winds = reshaped_winds.reshape(shape_for_SVD).T
    
    ## Build scaler to remove mean and scale with std before SVD
    scaler_winds = StandardScaler()
    reshaped_winds = scaler_winds.fit_transform(reshaped_winds)

    ## Compute errors
    error_inputs = pd.DataFrame()
    for n_components in range_n_components:
        for name_encoder in encoders:
            encoder = encoders[name_encoder]
            encoder_input = encoder(n_components=n_components)
            
            encoded_input   = encoder_input.fit_transform(reshaped_winds)
            recovered_input = encoder_input.inverse_transform(encoded_input)
            
            loc_dict = {
                'encoder': name_encoder,
                'encoder_func': encoder_input,
                'n_components': n_components,
                'mse': mean_squared_error(recovered_input, reshaped_winds),
            }
            error_inputs = error_inputs.append( [loc_dict] )
    
    return error_inputs

def plot_encoding_errors(one_dataset, error_inputs, id_wind_profiles, output_dir, comp_to_highlight=10, encoders_to_plot='PCA'):

    """
    Plot encoding error vs components and their impact on a given slice of winds
    """
    
    ## Error to plot
    error_to_plot = error_inputs.loc[error_inputs.encoder == encoders_to_plot]
    range_components = error_to_plot.n_components.unique()
    
    ## Wind profiles to plot
    winds = one_dataset.wind_stack[id_wind_profiles, :, :]
    
    ## Reshape wind stack for fitting
    shape_init    = one_dataset.wind_stack.shape
    shape_for_SVD = (shape_init[1], shape_init[0]*shape_init[2])
    reshaped_winds = one_dataset.wind_stack.transpose(1,0,2)
    reshaped_winds = reshaped_winds.reshape(shape_for_SVD).T
    
    ## Build scaler to remove mean and scale with std before SVD
    scaler_winds = StandardScaler()
    reshaped_winds = scaler_winds.fit_transform(reshaped_winds)
    
    ## Plotting formatting
    distances = one_dataset.distances
    altitudes = one_dataset.altitudes
    max_wind = winds.max()
    original_winds = np.linspace(distances[0], distances[-1], winds.shape[1])
    range_winds = np.linspace(distances[0], distances[-1], 3)
    delta_range = (distances[-1] - distances[0]) / len(range_winds)
    cmap = sns.color_palette("rocket", n_colors=len(range_components))
    
    ## Plot profiles, reconstruted profiles and error
    #fig, axs = plt.subplots(nrows=len(encoders_to_plot), ncols=winds.shape[1]+1)
    #for iencoder, encoder in enumerate(encoders_to_plot):
    fig = plt.figure()
    grid = fig.add_gridspec(1, 4)
    ax_error = fig.add_subplot(grid[0, 0])
    ax_winds = fig.add_subplot(grid[0:, 1:])
    
    ax_error.plot(error_to_plot.n_components, error_to_plot.mse)
    ax_error.set_xlabel('N$_{comp}$')
    ax_error.set_ylabel('MSE')
    ax_error.set_ylim([error_to_plot.mse.min(), error_to_plot.mse.max()])
    
    for irange, range_value in enumerate(range_winds):
        irange_original = np.argmin(abs(original_winds - range_value))
        
        winds_scaled = scaler_winds.transform([winds[:, irange_original]])
        for icomponent, n_components in enumerate(range_components):
            encoder_input = error_to_plot.loc[error_to_plot.n_components==n_components, 'encoder_func'].iloc[0]
            encoded_input = encoder_input.transform(winds_scaled)
            recovered_input = encoder_input.inverse_transform(encoded_input)
            recovered_input = scaler_winds.inverse_transform(recovered_input)
            recovered_input = recovered_input[0, :]
            normalized_profile = recovered_input * (0.5 * delta_range * 0.9) / max_wind
            label = '_nolegend_'
            linestyle = '-'
            if irange == 0:
                label = str(n_components)
            if icomponent == comp_to_highlight:
                linestyle = '--'
                
            ax_winds.plot(range_value + delta_range/2. + normalized_profile, altitudes, color=cmap[icomponent], label=label, linestyle=linestyle)
        
        normalized_profile = winds[:, irange_original] * (0.5 * delta_range * 1.2) / max_wind
        ax_winds.plot(range_value + delta_range/2. + normalized_profile, altitudes, color='black', linestyle=':')
        
    #ax_winds.legend(loc='lower left', bbox_to_anchor=(0., 1.01), ncol=9, handleheight=0.2, handlelength=0.1, labelspacing=0.01, columnspacing=0.05)
    
    ## Create custom colormap
    n = len(range_components)
    norm = matplotlib.colors.Normalize(vmin=1, vmax=n)
    cmap = sns.color_palette("rocket", n_colors=len(range_components), as_cmap=True)
    #cmap = matplotlib.cm.cool
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    axins = inset_axes(ax_winds, width="100%", height="3%", loc='lower left', 
                       bbox_to_anchor=(0., 1.02, 1, 1.), bbox_transform=ax_winds.transAxes, borderpad=0)
    axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
    cbar = plt.colorbar(sm, ticks=range_components, cax=axins, extend='both', orientation='horizontal')
    cbar.solids.set_edgecolor("face")
    cbar.ax.set_xlabel('N$_{comp}$', labelpad=1) 
    cbar.ax.xaxis.tick_top()
    cbar.ax.xaxis.set_label_position('top')
    
    ax_winds.yaxis.tick_right()
    ax_winds.yaxis.set_label_position("right")
    ax_winds.set_ylabel('Altitude\n(km)')
    ax_winds.set_xlabel('Range (km)')
    ax_winds.set_ylim([altitudes[0], altitudes[-1]])
        
    plt.savefig(output_dir + 'errors_' +encoders_to_plot+ '.pdf')
    
def compare_downsampling(one_dataset, id, output_dir, max_range=1000.):

    """
    3 panels comparison between true wind profiles, downsampled wind profiles and difference between the two
    """

    coef_downsample = 20
    #coef_upsample   = 4
    ranges_upsample = np.linspace(0., max_range, one_dataset.encoded_input.shape[2])
    ranges = np.linspace(0., max_range, one_dataset.wind_stack.shape[-1])
    altitudes = one_dataset.altitudes[::coef_downsample]
    altitudes[-1] = one_dataset.altitudes[-1]
    encoded_wind = one_dataset.encoded_input[id, :, :, 0]
    
    ## Interpolate coarse model over finer grid
    x, y = np.meshgrid(ranges_upsample, altitudes)
    coords = np.c_[x.ravel(), y.ravel()]
    XI, YI = np.meshgrid(ranges, one_dataset.altitudes)
    encoded_wind_interpolated = griddata(coords, encoded_wind.ravel(), (XI, YI), method='cubic') 
    
    ## Setup figure
    fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True)
    cmap = sns.color_palette("rocket", as_cmap=True)
    
    vmin, vmax = encoded_wind.min(), encoded_wind.max()
    axs[0].pcolormesh(ranges, one_dataset.altitudes, one_dataset.wind_stack[id, :, :], vmin=vmin, vmax=vmax, cmap=cmap)
    axs[0].set_title('True')
    axs[1].pcolormesh(ranges_upsample, altitudes, encoded_wind, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[1].set_title('Encoded')
    sc = axs[2].pcolormesh(ranges, one_dataset.altitudes, encoded_wind_interpolated-one_dataset.wind_stack[id, :, :], vmin=vmin, vmax=vmax, cmap=cmap)
    contours = axs[2].contour(XI, YI, encoded_wind_interpolated-one_dataset.wind_stack[id, :, :], levels=[-10., -5., 5., 10.], color='black')
    axs[2].clabel(contours, inline=True, fontsize=10)

    axs[2].set_title('Difference\n(encoded - true)')
    axs[0].set_ylabel('Altitude (km)')
    axs[0].set_xlabel('Range (km)')
    
    for i in range(1,3):
        axs[i].set_xticklabels([])
    
    axins = inset_axes(axs[2], width="7%", height="100%", loc='lower left', 
                       bbox_to_anchor=(1.05, 0., 1, 1.), bbox_transform=axs[2].transAxes, borderpad=0)
    axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
    cbar = plt.colorbar(sc, cax=axins, extend='both', orientation='vertical')
    cbar.ax.set_ylabel('wind (m/s)', rotation=270.) 
    
    fig.subplots_adjust(top=0.88, wspace=0.05, left=0.1, right=0.85)
    
    file_name = 'downsampled_winds_{id}.pdf'
    plt.savefig(output_dir + file_name.format(id=id))
   
def get_avg_profile(recovered_output, distances, mov_mean):

    ## Take moving average of data
    dr = distances[1] - distances[0]
    N = int(mov_mean/dr)
    #print(recovered_output.shape, )
    recovered_output_avg = np.convolve(recovered_output[:], np.ones(N)/N, mode='valid')
    recovered_output_avg = np.r_[recovered_output_avg, recovered_output_avg[-1]*np.ones((N-1,))]
    #recovered_output_avg = recovered_output_avg[None,:]
    
    return recovered_output_avg
   
def compute_TL_error_with_ids(model, mov_mean, nb_ranges, nb_ceffs, ids, get_range_uncertainty=False, cols_ceff=['veff-1-15', 'veff-35-60', 'veff-80-120']):

    """
    Compute true and predicted TL outputs for a list of ids in the testing dataset
    DOES NOT WORK WITH POOL BECAUSE KERAS NOT MULTIPROC COMPATIBLE
    """
    
    ## Get ML predictions
    TL_pred_normalized = model.model.predict([model.dataset.encoded_input[model.test_ids,:], model.dataset.f0s[model.test_ids]])
    TL_pred = model.scaler_TL.inverse_transform(TL_pred_normalized)
    
    ## Retrieve true outputs
    TL_true = model.dataset.TL_stack[model.test_ids, :]
    dr = model.dataset.distances[1] - model.dataset.distances[0]
    N = int(mov_mean/dr)
    TL_true_avg = np.array([np.convolve(TL, np.ones(N)/N, mode='valid') for TL in TL_true])
    TL_true_avg = np.c_[TL_true_avg, np.repeat(TL_true_avg[:,-1:], N-1, axis=1)*np.ones((TL_true_avg.shape[0],N-1))]
    
    if not get_range_uncertainty:
    
        ## Global MSE
        global_mse_err = np.mean((TL_pred-TL_true_avg)**2, axis=1)
        
        ## Range dependent MSE
        ranges = np.linspace(model.dataset.distances.min(), model.dataset.distances.max(), nb_ranges+1)
        idx_ranges = [(np.argmin(abs(model.dataset.distances-ranges[irange])), np.argmin(abs(model.dataset.distances-ranges[irange+1]))) for irange in range(ranges.size-1)]
        range_mse_err = []
        for ir, irp1 in idx_ranges: range_mse_err.append( np.mean((TL_pred[:,ir:irp1+1]-TL_true_avg[:,ir:irp1+1])**2, axis=1) )
        
        ## Ceff dependent MSE
        for col_ceff in cols_ceff:
            model.dataset.properties[col_ceff+'-cat'] = pd.cut(model.dataset.properties[col_ceff], nb_ceffs)
            model.dataset.properties[col_ceff+'-mean'] = model.dataset.properties.groupby(col_ceff+'-cat')[col_ceff].transform('mean')
            
        ## Store in dataframe
        mse_errs = pd.DataFrame()
        mse_errs['simu_id'] = model.test_ids
        mse_errs['nb_slice'] = model.dataset.slice_nos[model.test_ids]
        mse_errs['f0s'] = model.dataset.f0s[model.test_ids]
        mse_errs['mse_err'] = global_mse_err
        for col_ceff in cols_ceff:
            mse_errs[col_ceff+'-mean'] = model.dataset.properties.loc[model.dataset.properties.simu_id.isin(model.test_ids)].groupby('simu_id').first().reset_index()[col_ceff+'-mean']
            mse_errs[col_ceff] = model.dataset.properties.loc[model.dataset.properties.simu_id.isin(model.test_ids)].groupby('simu_id').first().reset_index()[col_ceff]
        range_labels = ['mse_err_{r:.2f}'.format(r=0.5*(ranges[irange]+ranges[irange+1])) for irange in range(ranges.size-1)]
        for ilabel, label in enumerate(range_labels): mse_errs[label] = range_mse_err[ilabel]
    
    else:
    
        ## Compute error vs range
        mse_errs = TL_pred-TL_true_avg
        mse_errs = pd.DataFrame(mse_errs)
        mse_errs['simu_id'] = model.test_ids
        mse_errs['nb_slice'] = model.dataset.slice_nos[model.test_ids]
        mse_errs['f0'] = model.dataset.f0s[model.test_ids]
        
    mse_errs.reset_index(drop=True, inplace=True)
        
    return mse_errs

def compute_all_errors(model, nb_ranges=5, nb_ceffs=5, mov_mean=20., nb_CPU=20,
                       range_unknowns = [['mse_err', 20], ['ranges', 5], ['veff-5-20', 6], ['std-5-20', 6], ['veff-20-50', 6], ['std-20-50', 6], ['veff-50-100', 6], ['std-50-100', 6], ['f0s', 5]]):

    """
    Compute errors across the test dataset
    """

    from tensorflow.keras import models

    ids_test = model.test_ids
    all_errors = model.dataset.properties.loc[model.dataset.properties.simu_id.isin(ids_test)]
    r = model.dataset.properties.ranges.unique()
    
    ids = all_errors.simu_id.unique()
    nb_simulations = len(ids)
    compute_TL_error_with_ids_partial = \
        partial(compute_TL_error_with_ids, model, mov_mean, nb_ranges, nb_ceffs)

    N = min(nb_CPU, nb_simulations)
    ## If one CPU requested, no need for deployment
    if N == 1:
        mse_errs = compute_TL_error_with_ids_partial(ids)

    ## Otherwise, we pool the processes
    else:

        step_idx =  nb_simulations//N
        list_of_lists = []
        for i in range(N):
            idx = np.arange(i*step_idx, (i+1)*step_idx)
            if i == N-1:
                idx = np.arange(i*step_idx, nb_simulations)
            list_of_lists.append( (models.clone_model(model.model), ids[idx]) )
            
        with get_context("spawn").Pool(processes = N) as p:
            results = p.map(compute_TL_error_with_ids_partial, list_of_lists)

        mse_errs = pd.DataFrame()
        for result in results:
            mse_errs = mse_errs.append( result )

        mse_errs.reset_index(drop=True, inplace=True)
    
    all_errors = all_errors.loc[all_errors.simu_id.isin(ids)].sort_values(by=['simu_id','ranges'])
    for col in mse_errs.loc[:, ~mse_errs.columns.isin(['simu_id'])].columns:
        all_errors[col] = np.repeat(mse_errs[col].values, r.size)
    
    ## Finally, compute lateral and frequency dependent errors
    compute_errors_range_f0_stdlateral_veff(all_errors, range_unknowns)
    
    ## Get RMSE from MSE and transform range float to strings
    all_errors['q-rmse_err'] = np.sqrt(all_errors['q-mse_err'])
    all_errors['rmse_err'] = np.sqrt(all_errors['mse_err'])
    all_errors['ranges-str'] = all_errors['q-ranges'].round(1).astype(str)
    
    return all_errors
        
def compute_errors_range_f0_stdlateral_veff(all_errors, range_unknowns):

    """
    Bin specific unknowns in a certain number of quantiles
    """

    for name, nb_range_values in range_unknowns:
        range_values = np.linspace(all_errors[name].min(), all_errors[name].max(), nb_range_values)
        s = pd.cut(all_errors[name], bins=nb_range_values)#, labels=range_values)
        all_errors['q-' + name] = [(a.left + a.right)/2 for a in s]
        #compute_error_for_one_unknown(all_errors, range_values, name, nb_ranges)
    
def plot_errors(all_errors, columns_to_plot):

    fig, axs = plt.subplots(nrows=len(columns_to_plot), ncols=len(columns_to_plot), sharey=True)
    for icol, col in enumerate(columns_to_plot):
        for irow, row in enumerate(columns_to_plot):
            if col == row:
                sns.displot(data=all_errors, x="q-mse_err", hue=col, fill=True, ax=axs[irow, icol])
            else:
                sns.displot(data=all_errors, x=col, y=row, kind='kde', fill=True, ax=axs[irow, icol])
    
    plt.show()
    
def plot_distribution_winds(one_dataset, column_name='std', col_labels={}, xlabel='Average lateral std (m/s)'):

    """
    Plot the distribution of average lateral wind std in different atmospheric layers
    """
    
    df_to_plot = pd.DataFrame()
    for col in one_dataset.properties.columns:
        if not column_name in col:
            continue
        loc_df = pd.DataFrame()
        loc_df['value'] = one_dataset.properties[col]
        col_label = col
        if col in col_labels:
            col_label = col_labels[col]
        loc_df['layer'] = col_label
        df_to_plot = df_to_plot.append( loc_df )
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.kdeplot(data=df_to_plot, x="value", hue="layer", fill=True, common_norm=True, palette="rocket", alpha=.5, linewidth=0, ax=ax)
    ax.set_xlabel(xlabel)
    plt.show()
    
##########################
if __name__ == '__main__':
    
    bp()
