###########################################################################
# GAUSSIAN PLUME MODEL FOR TEACHING PURPOSES                              #
# PAUL CONNOLLY (UNIVERSITY OF MANCHESTER, 2017)                          #
# THIS CODE IS PROVIDED `AS IS' WITH NO GUARANTEE OF ACCURACY             #
# IT IS USED TO DEMONSTRATE THE EFFECTS OF ATMOSPHERIC STABILITY,         #
# WINDSPEED AND DIRECTION AND MULTIPLE STACKS ON THE DISPERSION OF        #
# POLLUTANTS FROM POINT SOURCES                                           #
###########################################################################

import numpy as np
import sys,os
from scipy import special
from scipy.special import erfcinv as erfcinv
import tqdm as tqdm
import ast

from os import path
from gauss_func import gauss_func

import matplotlib.pyplot as plt
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=False)



def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth




def generate_data(xLoc, yLoc):

    ###########################################################################
    # Do not change these variables                                           #
    ###########################################################################


    # SECTION 0: Definitions (normally don't modify this section)
    # view
    PLAN_VIEW=1;
    HEIGHT_SLICE=2;
    SURFACE_TIME=3;
    NO_PLOT=4;

    # wind field
    CONSTANT_WIND=1;
    FLUCTUATING_WIND=2;
    PREVAILING_WIND=3;
    RANDOM_WIND=4;

    # number of stacks
    ONE_STACK=1;
    TWO_STACKS=2;
    THREE_STACKS=3;

    # stability of the atmosphere
    CONSTANT_STABILITY=1;
    ANNUAL_CYCLE=2;
    stability_str=['Very unstable','Moderately unstable','Slightly unstable', \
        'Neutral','Moderately stable','Very stable'];
    # Aerosol properties
    HUMIDIFY=2;
    DRY_AEROSOL=1;

    SODIUM_CHLORIDE=1;
    SULPHURIC_ACID=2;
    ORGANIC_ACID=3;
    AMMONIUM_NITRATE=4;
    nu=[2., 2.5, 1., 2.];
    rho_s=[2160., 1840., 1500., 1725.]; #TODO:MODIFY THIS FOR METHANE PROPERTIES
    Ms=[58.44e-3, 98e-3, 200e-3, 80e-3];
    Mw=18e-3;


    dxy=500; #50;          # resolution of the model in both x and y directions
    dz=10;
    x=np.mgrid[-2500:2500+dxy:dxy]; # solve on a 5 km domain
    y=x;              # x-grid is same as y-grid
    ###########################################################################



    # SECTION 1: Configuration
    # Variables can be changed by the user+++++++++++++++++++++++++++++++++++++
    RH=0.90;
    aerosol_type=ORGANIC_ACID;

    dry_size=60e-9;
    humidify=DRY_AEROSOL;

    stab1=1; # set from 1-6
    stability_used=CONSTANT_STABILITY;


    output=NO_PLOT;
    x_slice=26; # position (1-50) to take the slice in the x-direction
    y_slice=1;  # position (1-50) to plot concentrations vs time

    wind=RANDOM_WIND;
    stacks=ONE_STACK;

    stack_x=[xLoc, 1000., -200.];
    stack_y=[yLoc, 250., -500.];

    Q=[40., 40., 40.]; # mass emitted per unit time
    H=[50., 50., 50.]; # stack height, m
    days=50;          # run the model for 365 days
    #--------------------------------------------------------------------------
    times=np.mgrid[1:(days)*24+1:1]/24.;

    Dy=10.;
    Dz=10.;

    # SECTION 2: Act on the configuration information

    # Decide which stability profile to use
    if stability_used == CONSTANT_STABILITY:

       stability=stab1*np.ones((days*24,1));
       stability_str=stability_str[stab1-1];
    elif stability_used == ANNUAL_CYCLE:

       stability=np.round(2.5*np.cos(times*2.*np.pi/(365.))+3.5);
       stability_str='Annual cycle';
    else:
       sys.exit()


    # decide what kind of run to do, plan view or y-z slice, or time series
    if output == PLAN_VIEW or output == SURFACE_TIME or output == NO_PLOT:

       C1=np.zeros((len(x),len(y),days*24)); # array to store data, initialised to be zero

       [x,y]=np.meshgrid(x,y); # x and y defined at all positions on the grid
       z=np.zeros(np.shape(x));    # z is defined to be at ground level.
    elif output == HEIGHT_SLICE:
       z=np.mgrid[0:500+dz:dz];       # z-grid

       C1=np.zeros((len(y),len(z),days*24)); # array to store data, initialised to be zero

       [y,z]=np.meshgrid(y,z); # y and z defined at all positions on the grid
       x=x[x_slice]*np.ones(np.shape(y));    # x is defined to be x at x_slice
    else:
       sys.exit()



    # Set the wind based on input flags++++++++++++++++++++++++++++++++++++++++
    wind_speed=5.*np.ones((days*24,1)); # m/s
    if wind == CONSTANT_WIND:
       wind_dir=0.*np.ones((days*24,1));
       wind_dir_str='Constant wind';
    elif wind == FLUCTUATING_WIND:
       wind_dir=360.*np.random.rand(days*24,1);
       wind_dir_str='Random wind';
    elif wind == PREVAILING_WIND:
       wind_dir=-np.sqrt(2.)*erfcinv(2.*np.random.rand(24*days,1))*40.; #norminv(rand(days.*24,1),0,40);
       # note at this point you can add on the prevailing wind direction, i.e.
       # wind_dir=wind_dir+200;
       wind_dir[np.where(wind_dir>=360.)]= \
            np.mod(wind_dir[np.where(wind_dir>=360)],360);
       wind_dir_str='Prevailing wind';
    elif wind == RANDOM_WIND: 
        direction = 360*np.random.rand();
        wind_dir = direction*np.ones((days*24,1));
        wind_dir_str='Random wind direction'
    else:
       sys.exit()
    #--------------------------------------------------------------------------



    # SECTION 3: Main loop
    # For all times...
    C1=np.zeros((len(x),len(y),len(wind_dir)))
    for i in tqdm.tqdm(range(0,len(wind_dir))):
       for j in range(0,stacks):
            C=np.ones((len(x),len(y)))
            C=gauss_func(Q[j],wind_speed[i],wind_dir[i],x,y,z,
                stack_x[j],stack_y[j],H[j],Dy,Dz,stability[i]);
            C1[:,:,i]=C1[:,:,i]+C;




    # SECTION 4: Post process / output

    # decide whether to humidify the aerosol and hence increase the mass
    if humidify == DRY_AEROSOL:
       #rint('do not humidify');
        2+2
    elif humidify == HUMIDIFY:
       mass=np.pi/6.*rho_s[aerosol_type]*dry_size**3.;
       moles=mass/Ms[aerosol_type];

       nw=RH*nu[aerosol_type]*moles/(1.-RH);
       mass2=nw*Mw+moles*Ms[aerosol_type];
       C1=C1*mass2/mass;
    else:
       sys.exit()

    NUM_SAMPLES = 20

#    for i in range(NUM_SAMPLES):
#        random_sample_x = np.random.randint(len(C1))
#        random_sample_y = np.random.randint(len(C1[0]))
#        total_concentration = sum(C1[random_sample_x,random_sample_y])
#        if(total_concentration != 0):
#            print([random_sample_x*dxy-2500, random_sample_y*dxy-2500, total_concentration])

#    print('Actual leak position and concentration.')
#    print('Concentration found may be slightly off by resolution')

#    for i in range(0,1):
#        for j in range(0,1):
#            print([random_x+dxy*i, random_y+dxy*j,sum(C1[(random_x+dxy*i+2500)/dxy,(random_y+dxy*j+2500)/dxy])])

#    concentrations = []

#    for i in range(len(C1)):
#        for j in range(len(C1)):
#            concentrations.append(sum(C1[i][j]))
#    print('Max concentration:')
#    max_concentration = max(concentrations)
#    print(max_concentration)

#    for i in range(len(C1)):
#        for j in range(len(C1)):
#            if(sum(C1[i][j]) == max_concentration):
#                print(str(i*dxy-2500) + ", " + str(j*dxy-2500))

    # output the plots
    if output == PLAN_VIEW:
       plt.figure()
       plt.ion()

       plt.pcolor(x,y,np.mean(C1,axis=2)*1e6, cmap='jet');
       plt.clim((0, 1e2));

       plt.title(stability_str + '\n' + wind_dir_str);
       plt.xlabel('x (metres)');
       plt.ylabel('y (metres)');
       cb1=plt.colorbar();
       #cb1.set_label('$\mu$ g m$^{-3}$');

       plt.show()
       #plt.ion()
       #plt.draw()
       plt.pause(25)

    elif output == HEIGHT_SLICE:
       plt.figure();
       plt.ion()

       plt.pcolor(y,z,np.mean(C1,axis=2)*1e6, cmap='jet')
       plt.clim((0,1e2));
       plt.xlabel('y (metres)');
       plt.ylabel('z (metres)');
       plt.title(stability_str + '\n' + wind_dir_str);
       cb1=plt.colorbar();
       cb1.set_label('$\mu$ g m$^{-3}$');
       plt.show()

    elif output == SURFACE_TIME:
       f,(ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
       ax1.plot(times,1e6*np.squeeze(C1[y_slice,x_slice,:]));
       try:
          ax1.plot(times,smooth(1e6*np.squeeze(C1[y_slice,x_slice,:]),24),'r');
          ax1.legend(('Hourly mean','Daily mean'))
       except:
          sys.exit()

       ax1.set_xlabel('time (days)');
       ax1.set_ylabel('Mass loading ($\mu$ g m$^{-3}$)');
       ax1.set_title(stability_str +'\n' + wind_dir_str);

       ax2.plot(times,stability);
       ax2.set_xlabel('time (days)');
       ax2.set_ylabel('Stability parameter');
       f.show()

    elif output == NO_PLOT:
        #print('don''t plot');
        concentrations = []
        for j in range(len(C1)):
            newrow = []
            for i in range(len(C1)):
                newrow.append(sum(C1[i][j]))
            concentrations.append(newrow)
        return [concentrations, direction, [(xLoc+2500)/dxy,(yLoc+2500)/dxy]]
    else:
       sys.exit()

###########
#ANN Stuff#
###########

def write_data(n):
    f = open('data.txt','w+')

    for i in range(n):
        print('Generation: ' + str(i) + '/' + str(n))
        #If you don't want random positions for the stack replace
        #random_x and random_y in stack_x and stack_y with 0.
        random_x = np.random.randint(-1000,1001)
        random_y = np.random.randint(-1000,1001)
        a,b,c = generate_data(random_x,random_y)
        #print(str(a))
        f.write(str(a) + ';' + str(b) + ';' + str(c)+ '\n')

    f.close()

def read_data(fileName):

    #Read the file
    f = open(fileName,'r+')
    content = f.readlines()
    f.close()

    #Parse the data
    entry_grid =[] #Concentration levels per data point
    entry_direction = [] #Wind direction per data point
    entry_leak = [] #Leak location per data point

    #Update the above lists for each data point.
    for i in range(len(content)):
        entry = content[i].split(';')
        entry_grid.append( ast.literal_eval(entry[0]))
        entry_direction.append(float(entry[1]))
        entry_leak.append(ast.literal_eval(entry[2]))

    return entry_grid, entry_direction, entry_leak


    
