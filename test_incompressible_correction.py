from parcels import FieldSet, ParticleSet, ScipyParticle, JITParticle, Variable
from parcels import AdvectionAnalytical, AdvectionRK4, plotTrajectoriesFile
import numpy as np
from datetime import timedelta as delta
import matplotlib.pyplot as plt
import matplotlib as mp

# set path
mainpath = '/Users/lrousselet/LOUISE/PARCELS/parcels/'

#dimensions
xdim = 103
ydim = 103
(Nx, Ny) = (xdim - 2, ydim - 2)
xi = np.linspace(0,Nx+1,xdim,dtype=np.float32)
yi = np.linspace(0,Ny+1,ydim,dtype=np.float32)
u = np.zeros((ydim,xdim),dtype=np.float32)
v = np.zeros((ydim,xdim),dtype=np.float32)
ue = np.zeros((ydim,xdim),dtype=np.float32)
ve = np.zeros((ydim,xdim),dtype=np.float32)
U = np.zeros((ydim,xdim),dtype=np.float32)
V = np.zeros((ydim,xdim),dtype=np.float32)
psi = np.zeros((ydim,xdim),dtype=np.float32)
x1 = np.linspace(-1,Nx-1,Nx+1,dtype=np.float32)
y1 = np.linspace(-1,Ny-1,Ny+1,dtype=np.float32)
xe = np.amax(x1)
L = np.max(y1)
dxG = np.zeros((xdim,),dtype=np.float32)
dyG = np.zeros((ydim,),dtype=np.float32)
dxG[1:Nx+1] = np.diff(x1)
dxG = np.tile(dxG,(ydim,1))
dyG[1:Ny+1] = np.diff(y1)
dyG = np.tile(dyG,(xdim,1))

#params
a = 0.8
b = 1.2
U0 = 0.5

#psi for figure
for i in range(1,Nx):
    for j in range(1,Ny):
        psi[j,i] = np.sin(np.pi*(a*(x1[i]/xe)-b*(y1[j]/L)))*(x1[i]*(x1[i]-xe)*y1[j]*(y1[j]-L))**2

#u and v exact
for i in range(1,Nx):
    for j in range(1,Ny):
        X = x1[i]/xe; Y = y1[j]/L
        ue[j,i] = -2*(X*(X-1))**2*(Y*(Y-1)**2+Y**2*(Y-1))*np.sin(np.pi*(a*X-b*Y)) + b*np.pi*np.cos(np.pi*(a*X-b*Y))*(X*(X-1)*Y*(Y-1))**2
        ue[j,i] = ue[j,i]*U0
        ve[j,i] = 2*(Y*(Y-1))**2*(X*(X-1)**2+X**2*(X-1))*np.sin(np.pi*(a*X-b*Y)) + a*np.pi*np.cos(np.pi*(a*X-b*Y))*(X*(X-1)*Y*(Y-1))**2
        ve[j,i] = ve[j,i]*U0

#u and v incompressible
for i in range(1,Nx):
    for j in range(1,Ny):
        X = x1[i]/xe; Y = (y1[j]-0.5)/L
        u[j,i] = -2*(X*(X-1))**2*(Y*(Y-1)**2+Y**2*(Y-1))*np.sin(np.pi*(a*X-b*Y)) + b*np.pi*np.cos(np.pi*(a*X-b*Y))*(X*(X-1)*Y*(Y-1))**2
        u[j,i] = u[j,i]*U0
        U[j,i] = u[j,i]*dyG[j,i]
        V[j,i] = V[j-1,i] - U[j,i] + U[j,i-1]
        v[j,i] = V[j,i]/dxG[j,i]


data = {'U': u, 'V': v, 'dxG': dxG, 'dyG': dyG}
dimensions = {'U':{'lon': xi, 'lat': yi}, 'V':{'lon': xi, 'lat': yi}, 'dxG':{'lon': xi, 'lat': yi}, 'dyG':{'lon': xi, 'lat': yi}}
allow_time_extrapolation = True
#classical
fieldset = FieldSet.from_data(data, dimensions, mesh='flat', allow_time_extrapolation=allow_time_extrapolation)
fieldset.U.interp_method = 'cgrid_velocity' #classical
fieldset.V.interp_method = 'cgrid_velocity'
#corrected
fieldsetCOR = FieldSet.from_data(data, dimensions, mesh='flat', allow_time_extrapolation=allow_time_extrapolation)
fieldsetCOR.U.interp_method = 'cgrid_velocity_incompressible'
fieldsetCOR.V.interp_method = 'cgrid_velocity_incompressible'

#test for N-E triangle
RvaluesX = np.random.uniform(1,100,200)
RvaluesY = np.random.uniform(1,102,200)
count = 0
partX, partY = [], []
for i in range(RvaluesX.shape[0]):
    if RvaluesY[i] > 100-RvaluesX[i]:
            partX += [RvaluesX[i]]
            partY += [RvaluesY[i]]

partX, partY = np.array(partX), np.array(partY)

# RK4 exact solution with no interpolation
def Advection2Dcorr_exact(particle, fieldset, time):
    """Advection of particles using fourth-order Runge-Kutta integration.

    Function needs to be converted to Kernel object before execution
    2D Test case with exact solution and no interpolation"""
    #param
    a = 0.8
    b = 1.2
    U0 = 0.5
    xe = 100
    L = 100
    
    #step 1 
    (xx, yy) = (particle.lon, particle.lat)
    (X, Y) = (xx/xe, yy/L)
    u1 = -2*(X*(X-1))**2*(Y*(Y-1)**2+Y**2*(Y-1))*np.sin(np.pi*(a*X-b*Y)) + b*np.pi*np.cos(np.pi*(a*X-b*Y))*(X*(X-1)*Y*(Y-1))**2
    u1 = u1*U0
    v1 = 2*(Y*(Y-1))**2*(X*(X-1)**2+X**2*(X-1))*np.sin(np.pi*(a*X-b*Y)) + a*np.pi*np.cos(np.pi*(a*X-b*Y))*(X*(X-1)*Y*(Y-1))**2
    v1 = v1*U0
    lon1, lat1 = (particle.lon + u1*.5*particle.dt, particle.lat + v1*.5*particle.dt)
    #step 2 
    (xx, yy) = (lon1, lat1)
    (X, Y) = (xx/xe, yy/L)
    u2 = -2*(X*(X-1))**2*(Y*(Y-1)**2+Y**2*(Y-1))*np.sin(np.pi*(a*X-b*Y)) + b*np.pi*np.cos(np.pi*(a*X-b*Y))*(X*(X-1)*Y*(Y-1))**2
    u2 = u2*U0
    v2 = 2*(Y*(Y-1))**2*(X*(X-1)**2+X**2*(X-1))*np.sin(np.pi*(a*X-b*Y)) + a*np.pi*np.cos(np.pi*(a*X-b*Y))*(X*(X-1)*Y*(Y-1))**2
    v2 = v2*U0
    lon2, lat2 = (particle.lon + u2*.5*particle.dt, particle.lat + v2*.5*particle.dt)
    #step 3
    (xx, yy) = (lon2, lat2)
    (X, Y) = (xx/xe, yy/L)
    u3 = -2*(X*(X-1))**2*(Y*(Y-1)**2+Y**2*(Y-1))*np.sin(np.pi*(a*X-b*Y)) + b*np.pi*np.cos(np.pi*(a*X-b*Y))*(X*(X-1)*Y*(Y-1))**2
    u3 = u3*U0
    v3 = 2*(Y*(Y-1))**2*(X*(X-1)**2+X**2*(X-1))*np.sin(np.pi*(a*X-b*Y)) + a*np.pi*np.cos(np.pi*(a*X-b*Y))*(X*(X-1)*Y*(Y-1))**2
    v3 = v3*U0
    lon3, lat3 = (particle.lon + u3*particle.dt, particle.lat + v3*particle.dt)
    #step 4 
    (xx, yy) = (lon3, lat3)
    (X, Y) = (xx/xe, yy/L)
    u4 = -2*(X*(X-1))**2*(Y*(Y-1)**2+Y**2*(Y-1))*np.sin(np.pi*(a*X-b*Y)) + b*np.pi*np.cos(np.pi*(a*X-b*Y))*(X*(X-1)*Y*(Y-1))**2
    u4 = u4*U0
    v4 = 2*(Y*(Y-1))**2*(X*(X-1)**2+X**2*(X-1))*np.sin(np.pi*(a*X-b*Y)) + a*np.pi*np.cos(np.pi*(a*X-b*Y))*(X*(X-1)*Y*(Y-1))**2
    v4 = v4*U0
    particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
    particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt


psetRK4_0 = ParticleSet(fieldset, pclass=ScipyParticle, lon=partX, lat=partY)
output = psetRK4_0.ParticleFile(name=mainpath+'tests/bilinear_interp_correction/2D_case_cgrid_RK4_exact.nc', outputdt=1)
psetRK4_0.execute(Advection2Dcorr_exact, dt=0.1, runtime=1000,output_file=output)
output.close()

#RK4 velocities + not corrected
psetRK4_1 = ParticleSet(fieldset, pclass=ScipyParticle, lon=partX, lat=partY)
output = psetRK4_1.ParticleFile(name=mainpath+'tests/bilinear_interp_correction/2D_case_cgrid_RK4_VNC.nc', outputdt=1)
psetRK4_1.execute(AdvectionRK4, dt=0.1, runtime=1000,output_file=output)
output.close()

#RK4 velocities +  corrected
psetRK4_2 = ParticleSet(fieldsetCOR, pclass=ScipyParticle, lon=partX, lat=partY)
output = psetRK4_2.ParticleFile(name=mainpath+'tests/bilinear_interp_correction/2D_case_cgrid_RK4_VC.nc', outputdt=1)
fieldsetCOR.UV.dxG = fieldsetCOR.dxG
fieldsetCOR.UV.dyG = fieldsetCOR.dyG
psetRK4_2.execute(AdvectionRK4, dt=0.1, runtime=1000,output_file=output)
output.close()

#do some plots
import xarray as xr

data_xarray_0 = xr.open_dataset(mainpath+'tests/bilinear_interp_correction/2D_case_cgrid_RK4_exact.nc')
xp_0 = data_xarray_0['lon'].values
yp_0 = data_xarray_0['lat'].values

data_xarray_1 = xr.open_dataset(mainpath+'tests/bilinear_interp_correction/2D_case_cgrid_RK4_VNC.nc')
xp_1 = data_xarray_1['lon'].values
yp_1 = data_xarray_1['lat'].values

data_xarray_2 = xr.open_dataset(mainpath+'tests/bilinear_interp_correction/2D_case_cgrid_RK4_VC.nc')
xp_2 = data_xarray_2['lon'].values
yp_2 = data_xarray_2['lat'].values

fig, ax = plt.subplots(figsize=(15,10),facecolor='w')
#cs = ax.pcolor(xi,y[1:Ny],psi[1:Ny,1:Nx])
plt.streamplot(xi,yi,u,v,color='k')
#fig.colorbar(cs)
for ii in range(xp_0.shape[0]):
       plt.plot(xp_0[ii,],yp_0[ii,])
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 30}
mp.rc('font', **font)
plt.title('2D bilinear case')
plt.savefig(mainpath+'tests/bilinear_interp_correction/FIGURES/2D_cgrid_RK4_VC.png', dpi=100)
plt.show()

fig, ax = plt.subplots(figsize=(15,10),facecolor='w')
cs = ax.pcolor(x[1:Nx],y[1:Ny],psi[1:Nx,1:Ny])
fig.colorbar(cs)
plt.plot(xp_0[:,0],yp_0[:,0],'kx')
plt.plot(xp_1[:,0],yp_1[:,0],'rx')#initial position
plt.plot(xp_2[:,0],yp_2[:,0],'bx')#initial position
mp.rc('font', **font)
plt.title('2D cgrid: initial positions')
plt.savefig(mainpath+'tests/bilinear_interp_correction/FIGURES/2D_cgrid_init.png', dpi=100)
plt.show()

fig, ax = plt.subplots(figsize=(15,10),facecolor='w')
cs = ax.pcolor(x[1:Nx],y[1:Ny],psi[1:Nx,1:Ny])
fig.colorbar(cs)
plota, = plt.plot(xp_0[:,-1],yp_0[:,-1],'kx',label='exact')
plotb, = plt.plot(xp_1[:,-1],yp_1[:,-1],'rx',label='no correction')
plotc, = plt.plot(xp_2[:,-1],yp_2[:,-1],'bx',label='corrected')#final position
plt.title('2D cgrid: final positions')
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 30}
mp.rc('font', **font)
plt.legend(handles=[plota,plotb,plotc],loc='lower left',fontsize='x-small')
plt.savefig(mainpath+'tests/bilinear_interp_correction/FIGURES/2D_cgrid_finalpos.png', dpi=100)
plt.show()
