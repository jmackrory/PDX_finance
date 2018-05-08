def take_log10(X,log_ind):
    """take_log10(X)
    Takes base10-log of X.  Sets NaN to zero, and shifts zero values by 1E-16.
    """
    #set missing values to zero.  (to avoid issues in network with NaNs)
    y=X.copy()
    y[np.isnan(y)]=0    
    #take logs. (shift zero to avoid NANs)
    y[:,log_ind] = np.log10(y[:,log_ind]+1E-16)
    return y

def scale_maxmin(X,log_ind):
    """scale_maxmin

    Scales each price column by taking log, and max/min scaling.
    Takes log of open/high/low/volumes/etf.
    Does not take log of indicators.
    Max-min scales all of them.
    """
    y=take_log10(X,log_ind)
    y_max = np.nanmax(y,axis=0)
    y_min = np.nanmin(y,axis=0)
    #compute middle and differences of the max/min.
    avg = 0.5*(y_max+y_min)
    rng = 0.5*(y_max-y_min)
    #scale to [-1,+1]
    yscaled= (y-avg)/rng
    return yscaled,rng,avg

def rescale_maxmin_log(Xscaled,avg,rng,log_ind):
    """rescale_maxmin

    Undoes log-max/min scaling. 
    First undoes max/min scaling.  
    Then exponentiates all variables which had log taken.

    Input: Xscaled (nrow, ncol) np.array of scaled values
           avg (ncol) np.array of average/mean values to shift by
           rng (ncol) np.array of standard devations to scale by
           
    """
    #set missing values to zero.  (to avoid issues in network with NaNs)
    X= avg + rng*Xscaled
    #take logs. (shift zero to avoid NANs)
    X[:,log_ind] = 10**X[:,log_ind]
    return X

def scale_diff_var(X,log_ind,diff_ind):
    """scale_diff_var
    Scales each column by taking log, then differencing.
    Then scales to have zero mean, and unit standard deviation..
    Takes log of all data.
    """
    y=take_log10(X,log_ind)
    #take differences logs. (shift zero to avoid NANs)    
    y[:,diff_ind] = np.diff(y[:,diff_ind],axis=0)
    y_std = np.nanstd(y,axis=0)
    y_mean = np.nanmean(y,axis=0)
    yscaled= (y-y_mean)/y_std
    return yscaled,y_mean,y_std

def scale_diff_iqr(X,diff_ind,log_ind):
    """scale_diff_var
    Scales each column by taking log, then differencing.
    Then scales to have zero median, and so 95% inter-quartile range is unity.
    Should handle the long-tail distributions we will encounter
    with some grace.  
    Only take the log for the stock/etf data.
    """

    y=take_log10(X,log_ind)
    #take differences logs. (shift zero to avoid NANs)
    nrow,ncol=y.shape
    y2=np.zeros((nrow-1,ncol))
    y2[:,diff_ind] = np.diff(y[:,diff_ind],axis=0)
    y2[:,~diff_ind] = y[1:,~diff_ind]

    keep_msk=(y2==y2)
    print('Num of NaN',np.sum(keep_msk))
    # print(keep_msk.shape,y2[keep_msk].shape)
    X_025 = np.nanpercentile(y2,q=2.5,axis=0)
    X_975 = np.nanpercentile(y2,q=97.5,axis=0)
    X_range = X_975-X_025
    X_median = np.nanmedian(y2,axis=0)
    Xscaled=np.where(keep_msk, (y2-X_median)/X_range,0)
    return Xscaled,X_median,X_range

def rescale_diff_iqr(Xscaled,mu,sd,x0,log_ind,diff_ind):
    """rescale_diffvar

    Undoes log-max/min scaling. 
    Takes log of open/high/low/volumes/etf.
    Does not take log of indicators.
    Max-min scales all of them.
    Only take the log for the stock/etf data.
    """
    #set missing values to zero.  (to avoid issues in network with NaNs)
    X= mu + sd*Xscaled
    X= np.insert(X,0,x0,axis=0)
    X[:,diff_ind] = np.cumsum(X[:,diff_ind],axis=0)
    #take logs. (shift zero to avoid NANs)
    X[:,log_ind] = 10**X[:,log_ind]
    return X