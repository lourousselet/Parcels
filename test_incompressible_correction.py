from parcels import FieldSet, ParticleSet, ScipyParticle, JITParticle, Variable
from parcels import AdvectionAnalytical, AdvectionRK4, plotTrajectoriesFile
import numpy as np
from datetime import timedelta as delta
import matplotlib.pyplot as plt
import matplotlib as mp

# domain size
xdim=103
ydim=103
pa = 0.8
pb = 1.2
Nx, Ny, Nz = xdim-1, ydim-1, 1
Lx, Ly = 1e6, 1e6
H = 1e3
(U0, V0) = (0.5, 0.5)
x = np.zeros(xdim, dtype=np.float32)
y = np.zeros(ydim, dtype=np.float32)
#x[1:Nx] = np.linspace(0, Nx, Nx-1, dtype=np.float32)
#y[1:Ny] = np.linspace(0, Ny, Ny-1, dtype=np.float32)
for i in range(1,Nx):
    x[i] = 1/(Nx+1)*(i+1)

for j in range(1,Ny):
    y[j] = 1/(Ny+1)*(j+1+0.5)


xe = np.amax(x)
L = np.amax(y)
#dx, dy, dz = Lx/(Nx-1), Ly/(Ny-1), H/(Nz)
u = np.zeros((ydim, xdim), dtype=np.float32)
v = np.zeros((ydim, xdim), dtype=np.float32)
ue = np.zeros((ydim, xdim), dtype=np.float32)
ve = np.zeros((ydim, xdim), dtype=np.float32)
psi = np.zeros((ydim, xdim), dtype=np.float32)
x1 = x #- dx/2
y1 = y #- dy/2

for i in range(1,Nx):
   for j in range(1,Ny): 
       psi[j,i] = np.sin(np.pi*(pa*(x[i]/xe) - pb*(y[j]/L)))*(x[i]*(x[i]-xe)*y[j]*(y[j]-L))**2;


for i in range(1,Nx):
    for j in range(1,Ny):
        c = np.pi * (pa*(x1[i]/xe) - pb*(y1[j]/L))
        u[j, i] = ((-2*(((x1[i]/xe)*((x1[i]/xe)-1))**2)*(((y1[j]/L)*(((y1[j]/L)-1)**2))+(((y1[j]/L)**2)*((y1[j]/L)-1)))*np.sin(c))+(pb*(np.pi/L)*np.cos(c)*((x1[i]/xe)*((x1[i]/xe)-1)*(y1[j]/L)*((y1[j]/L)-1))**2))/L
        u[j, i] = U0*(u[j,i])
#exact solution
        ue[j, i] = u[j,i]
        ve[j, i] = ((2*(((x1[i]/xe)*((x1[i]/xe)-1)**2)+(((x1[i]/xe)-1)*(x1[i]/xe)**2))*(((y1[j]/L)*((y1[j]/L)-1))**2)*np.sin(c))+(pa*(np.pi/xe)*np.cos(c)*(((x1[i]/xe)*((x1[i]/xe)-1)*(y1[j]/L)*((y1[j]/L)-1))**2)))/xe
        ve[j, i] = V0*(ve[j,i])


#incompressible field
for i in range(1,Nx):
    for j in range(1,Ny):
        v[j,i] = v[j-1,i] - u[j,i] + u[j,i-1]

data = {'U': u, 'V': v}
dimensions = {'lon': x, 'lat': y}
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
RvaluesX = np.random.uniform(0.1,0.95,200)
RvaluesY = np.random.uniform(0.1,0.95,200)
count = 0
X, Y = [], []
for i in range(RvaluesX.shape[0]):
    if RvaluesY[i] > 1-RvaluesX[i]:
            X += [RvaluesX[i]]
            Y += [RvaluesY[i]]

X, Y = np.array(X), np.array(Y)

# RK4 exact solution with no interpolation
def Advection2Dcorr_exact(particle, fieldset, time):
    """Advection of particles using fourth-order Runge-Kutta integration.

    Function needs to be converted to Kernel object before execution
    2D Test case with exact solution and no interpolation"""
    #param
    pa = 0.8
    pb = 1.2
    (U0, V0) = (0.5, 0.5)
    xe = 1.0
    L = 1.0
    
    #step 1 
    (xx, yy) = (particle.lon, particle.lat)
    c = np.pi * (pa*(xx/xe) - pb*(yy/L))
    u1 = (((-2*(((xx/xe)*((xx/xe)-1))**2)*(((yy/L)*(((yy/L)-1)**2))+(((yy/L)**2)*((yy/L)-1)))*np.sin(c))+(pb*(np.pi/L)*np.cos(c)*((xx/xe)*((xx/xe)-1)*(yy/L)*((yy/L)-1))**2))/L)*U0
    v1 = (((2*(((xx/xe)*((xx/xe)-1)**2)+(((xx/xe)-1)*(xx/xe)**2))*(((yy/L)*((yy/L)-1))**2)*np.sin(c))+(pa*(np.pi/xe)*np.cos(c)*(((xx/xe)*((xx/xe)-1)*(yy/L)*((yy/L)-1))**2)))/xe)*V0    
    lon1, lat1 = (particle.lon + u1*.5*particle.dt, particle.lat + v1*.5*particle.dt)
    #step 2 
    (xx, yy) = (lon1, lat1)
    c = np.pi * (pa*(xx/xe) - pb*(yy/L))
    u2 = (((-2*(((xx/xe)*((xx/xe)-1))**2)*(((yy/L)*(((yy/L)-1)**2))+(((yy/L)**2)*((yy/L)-1)))*np.sin(c))+(pb*(np.pi/L)*np.cos(c)*((xx/xe)*((xx/xe)-1)*(yy/L)*((yy/L)-1))**2))/L)*U0
    v2 = (((2*(((xx/xe)*((xx/xe)-1)**2)+(((xx/xe)-1)*(xx/xe)**2))*(((yy/L)*((yy/L)-1))**2)*np.sin(c))+(pa*(np.pi/xe)*np.cos(c)*(((xx/xe)*((xx/xe)-1)*(yy/L)*((yy/L)-1))**2)))/xe)*V0
    lon2, lat2 = (particle.lon + u2*.5*particle.dt, particle.lat + v2*.5*particle.dt)
    #step 3
    (xx, yy) = (lon2, lat2)
    c = np.pi * (pa*(xx/xe) - pb*(yy/L))
    u3 = (((-2*(((xx/xe)*((xx/xe)-1))**2)*(((yy/L)*(((yy/L)-1)**2))+(((yy/L)**2)*((yy/L)-1)))*np.sin(c))+(pb*(np.pi/L)*np.cos(c)*((xx/xe)*((xx/xe)-1)*(yy/L)*((yy/L)-1))**2))/L)*U0
    v3 = (((2*(((xx/xe)*((xx/xe)-1)**2)+(((xx/xe)-1)*(xx/xe)**2))*(((yy/L)*((yy/L)-1))**2)*np.sin(c))+(pa*(np.pi/xe)*np.cos(c)*(((xx/xe)*((xx/xe)-1)*(yy/L)*((yy/L)-1))**2)))/xe)*V0
    lon3, lat3 = (particle.lon + u3*particle.dt, particle.lat + v3*particle.dt)
    #step 4 
    (xx, yy) = (lon3, lat3)
    c = np.pi * (pa*(xx/xe) - pb*(yy/L))
    u4 = (((-2*(((xx/xe)*((xx/xe)-1))**2)*(((yy/L)*(((yy/L)-1)**2))+(((yy/L)**2)*((yy/L)-1)))*np.sin(c))+(pb*(np.pi/L)*np.cos(c)*((xx/xe)*((xx/xe)-1)*(yy/L)*((yy/L)-1))**2))/L)*U0
    v4 = (((2*(((xx/xe)*((xx/xe)-1)**2)+(((xx/xe)-1)*(xx/xe)**2))*(((yy/L)*((yy/L)-1))**2)*np.sin(c))+(pa*(np.pi/xe)*np.cos(c)*(((xx/xe)*((xx/xe)-1)*(yy/L)*((yy/L)-1))**2)))/xe)*V0    
    particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
    particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt


psetRK4_0 = ParticleSet(fieldset, pclass=ScipyParticle, lon=X, lat=Y)
output = psetRK4_0.ParticleFile(name='tests/bilinear_interp_correction/2D_case_cgrid_RK4_exact.nc', outputdt=1)
psetRK4_0.execute(Advection2Dcorr_exact, dt=0.1, runtime=500,output_file=output)
output.close()

#RK4 velocities + not corrected
psetRK4_1 = ParticleSet(fieldset, pclass=ScipyParticle, lon=X, lat=Y)
output = psetRK4_1.ParticleFile(name='tests/bilinear_interp_correction/2D_case_cgrid_RK4_VNC.nc', outputdt=1)
psetRK4_1.execute(AdvectionRK4, dt=0.1, runtime=500,output_file=output)
output.close()

#RK4 velocities +  corrected
psetRK4_2 = ParticleSet(fieldsetCOR, pclass=ScipyParticle, lon=X, lat=Y)
output = psetRK4_2.ParticleFile(name='tests/bilinear_interp_correction/2D_case_cgrid_RK4_VC2.nc', outputdt=1)
psetRK4_2.execute(AdvectionRK4, dt=0.1, runtime=500,output_file=output)
output.close()

#do some plots
import xarray as xr

data_xarray_0 = xr.open_dataset('tests/bilinear_interp_correction/2D_case_cgrid_RK4_exact.nc')
xp_0 = data_xarray_0['lon'].values
yp_0 = data_xarray_0['lat'].values

data_xarray_1 = xr.open_dataset('tests/bilinear_interp_correction/2D_case_cgrid_RK4_VNC.nc')
xp_1 = data_xarray_1['lon'].values
yp_1 = data_xarray_1['lat'].values

data_xarray_2 = xr.open_dataset('tests/bilinear_interp_correction/2D_case_cgrid_RK4_VC.nc')
xp_2 = data_xarray_2['lon'].values
yp_2 = data_xarray_2['lat'].values

fig, ax = plt.subplots(figsize=(15,10),facecolor='w')
cs = ax.pcolor(x[1:Nx],y[1:Ny],psi[1:Ny,1:Nx])
plt.streamplot(x[1:Nx],y[1:Ny],ue[1:Ny,1:Nx],ve[1:Ny,1:Nx],color='k')
fig.colorbar(cs)
for ii in range(xp_2.shape[0]):
       plt.plot(xp_2[ii,],yp_2[ii,])
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 30}
mp.rc('font', **font)
plt.title('2D bilinear case')
plt.savefig('tests/bilinear_interp_correction/FIGURES/2D_cgrid_RK4_VC.png', dpi=100)
plt.show()

fig, ax = plt.subplots(figsize=(15,10),facecolor='w')
cs = ax.pcolor(x[1:Nx],y[1:Ny],psi[1:Nx,1:Ny])
fig.colorbar(cs)
plt.plot(xp_0[:,0],yp_0[:,0],'kx')
plt.plot(xp_1[:,0],yp_1[:,0],'rx')#initial position
plt.plot(xp_2[:,0],yp_2[:,0],'bx')#initial position
mp.rc('font', **font)
plt.title('2D cgrid: initial positions')
plt.savefig('tests/bilinear_interp_correction/FIGURES/2D_cgrid_init.png', dpi=100)
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
plt.savefig('tests/bilinear_interp_correction/FIGURES/2D_cgrid_finalpos.png', dpi=100)
plt.show()
