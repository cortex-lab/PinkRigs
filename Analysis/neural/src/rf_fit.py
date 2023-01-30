import numpy as np
import scipy.optimize as opt
import utils.spike_dat as su

def get_start_params(myresp):
    max_amplitude=np.max(myresp) 
    x0=np.argmax(np.mean(myresp,axis=0))
    y0=np.argmax(np.mean(myresp,axis=1))
    sigmaX=2#get_sigma(np.arange(0,myresp.shape[1],1),np.mean(myresp,axis=0))
    sigmaY=2#get_sigma(np.arange(0,myresp.shape[0],1),np.mean(myresp,axis=1))
    
    return [max_amplitude,x0,sigmaX,y0,sigmaY]

def f_2D_gaussian(x,y,a,x0,sigmaX,y0,sigmaY):
    return a*np.exp(-((x-x0)**2/(2*sigmaX**2)+(y-y0)**2/(2*sigmaY**2)))

# This is the callable that is passed to curve_fit. M is a (2,N) array
# where N is the total number of data points in Z, which will be ravelled
# to one dimension.
def _gaussian(M, *args):
    x, y = M
    arr = np.zeros(x.shape)
    arr += f_2D_gaussian(x, y, *args)
    return arr

def get_VE(actual,predicted):
    actual_=np.ravel(actual)
    predicted_ = np.ravel(predicted)

    VE = 1-(np.var(actual_-predicted_)/np.var(actual_))

    return VE

class rf_fit(): 
    def __init__(self): # maybe sparseNoise needs to go from init
        self.azimuth_idx=np.arange(0,36)
        self.elevation_idx=np.arange(0,10)
        self.azi_loc_screen,self.elevation_loc_screen=np.meshgrid(self.azimuth_idx,self.elevation_idx)
        self.screenpos = np.vstack((self.azi_loc_screen.ravel(), self.elevation_loc_screen.ravel())) 
        

    def bin_spikes_per_loc(self,spikes,depth_corr_window_spacing=40,t_bin=0.01):
        self.t_bin=t_bin
        self.binned_spikes_depths = su.bin_spikes_pos_and_time(spikes,depth_corr_window_spacing=depth_corr_window_spacing,spike_binning_t=self.t_bin)
    
    
    def add_sparseNoise_info(self,sparseNoise):
        self.azimuths = np.unique(sparseNoise.squareAzimuth)
        self.elevations = np.unique(sparseNoise.squareElevation)
        self.xy_pos = np.array([sparseNoise.squareAzimuth,sparseNoise.squareElevation]).T
        self.xy_times = sparseNoise.squareOnTimes

    def fit_predict(self,response):

        # if the response is also 0/inf/nan the fit will error and that cannot be fitted anyway.
        # 
        # 
        if np.isnan(response).any():
            fitted_params=np.nan
        else:
            p0 = get_start_params(response)
            try:
                popt, _ = opt.curve_fit(_gaussian, self.screenpos, response.ravel(), p0)
                fitted_params=popt
                pred=f_2D_gaussian(self.azi_loc_screen,self.elevation_loc_screen,*fitted_params)
                ve = get_VE(response,pred)
                # probably not a real fit
                if ve<.3:
                    fitted_params=np.nan
            except RuntimeError: # this is when the curve fit could not manage to fit a gaussian after 1200 iterations
                    fitted_params=np.nan
                
        return fitted_params

    def get_response_binned(self,sel_depth_binned,t_before=0.2,t_after=0.06,t_delay=0.02):

        before_ix=int(t_before/self.t_bin)
        after_ix=int(t_after/self.t_bin)
        delay_ix=int(t_delay/self.t_bin)
        
        sNstartix=np.ceil(self.xy_times*(1/self.t_bin)).astype("int")
        response=np.zeros((self.azimuths.size,self.elevations.size))

        for xct,my_x in enumerate(self.azimuths): 
            for yct,my_y in enumerate(self.elevations): 
                sq=[my_x,my_y]
                ix=np.where((self.xy_pos==sq).all(axis=1))[0]
                
                nrlix=np.ravel(sNstartix[ix])

                resp_sq=np.zeros((nrlix.shape[0],before_ix+after_ix))
                for i in range(before_ix+after_ix):
                    ixs=nrlix-before_ix+i
                    resp_sq[:,i]=sel_depth_binned[ixs]

                blmean=(resp_sq[:,:before_ix].sum(axis=1))/t_before
                respmean=(resp_sq[:,(before_ix+delay_ix):].sum(axis=1))/(t_after-t_delay)

                response[xct,yct]=np.abs((respmean-blmean).mean(axis=0)/blmean.std(axis=0))
                #mypred,ve,fitted_params=self.fit_predict(response.T)

        return response.T