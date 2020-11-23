from parcels import FieldSet, ParticleSet, ScipyParticle, JITParticle, Variable
from parcels import AdvectionAnalytical, AdvectionRK4, plotTrajectoriesFile
import numpy as np
from datetime import timedelta as delta
import matplotlib.pyplot as plt

#def incompressible_fieldset(times, xdim=103, ydim=103):
#    """Implemented """
    # domain size
xdim=103
ydim=103
pa = 0.8
pb = 1.2
Nx, Ny = xdim-1, ydim-1
x = np.zeros(xdim, dtype=np.float32)
y = np.zeros(ydim, dtype=np.float32)
x[1:Nx] = np.linspace(0, 1, Nx-1, dtype=np.float32)
y[1:Ny] = np.linspace(0, 1, Ny-1, dtype=np.float32)
xe = np.amax(x)
L = np.amax(y)
dx, dy = x[2]-x[1], y[2]-y[1]
u = np.zeros((ydim, xdim), dtype=np.float32)
v = np.zeros((ydim, xdim), dtype=np.float32)
U = np.zeros((y.size, x.size), dtype=np.float32)
V = np.zeros((y.size, x.size), dtype=np.float32)
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
        u[j, i] = u[j,i]/xdim

#exact solution
#            v[j, i] = ((2*x1[i]*(a[i]**2)+2*a[i]*(x1[i]**2))*((y1[j]*b[j])**2)*np.sin(c)) + ((np.cos(c)*(x1[i]*a[i]*y1[j]*b[j])**2)*(np.pi/xe))
#

#incompressible field
for i in range(1,Nx):
    for j in range(1,Ny):
        v[j,i] = v[j-1,i] - u[j,i] + u[j,i-1]

##zeros on the boundary
#u[0,], u[-1,] = 0, 0
#u[:,0], u[:,-1]  = 0, 0
#v[0,], v[-1,] = 0, 0
#v[:,0], v[:,-1]  = 0, 0


fig, ax = plt.subplots(figsize=(15,10),facecolor='w')
cs = ax.pcolor(x[1:Nx],y[1:Ny],psi[1:Nx,1:Ny])
plt.streamplot(x[1:Nx],y[1:Ny],u[1:Nx,1:Ny],v[1:Nx,1:Ny],color='k')
fig.colorbar(cs)
plt.savefig('2D_case.png', dpi=100)
plt.show()

#    fig, ax = plt.subplots(figsize=(15,10),facecolor='w')
#    cs = ax.pcolor(x,y,psi)
#    plt.quiver(x,y,u,v)
#    fig.colorbar(cs)
#    plt.savefig('2D_case_uv.png', dpi=100)
#    plt.show()

data = {'U': u, 'V': v}
dimensions = {'lon': x, 'lat': y}
allow_time_extrapolation = True
fieldset = FieldSet.from_data(data, dimensions, mesh='flat', allow_time_extrapolation=allow_time_extrapolation)
fieldset.U.interp_method = 'cgrid_velocity'
fieldset.V.interp_method = 'cgrid_velocity'
#    return fieldset
#
#fieldsetINC = incompressible_fieldset(times=1)

#test for N-E triangle
RvaluesX = np.random.rand(20,20)
RvaluesY = np.random.rand(20,20)
count = 0
X, Y = [], []
for i in range(RvaluesX.shape[0]):
    for j in range(RvaluesY.shape[1]):
    	if RvaluesY[i,j] > 1-RvaluesX[i,j]:
            X += [RvaluesX[i,j]]
            Y += [RvaluesY[i,j]]

X, Y = np.array(X), np.array(Y)
psetAA = ParticleSet(fieldset, pclass=ScipyParticle, lon=X, lat=Y)

#analytical
output = psetAA.ParticleFile(name='test_2D_case.nc', outputdt=0.5)
psetAA.execute(AdvectionAnalytical,
               dt=np.inf,  # needs to be set to np.inf for Analytical Advection
               runtime=1000,
               output_file=output)

output.close()
#plotTrajectoriesFile('test_2D_case.nc')

#RK4
psetRK4 = ParticleSet(fieldset, pclass=JITParticle, lon=X, lat=Y)
output = psetRK4.ParticleFile(name='test_2D_case_RK4.nc', outputdt=0.5)
psetRK4.execute(AdvectionRK4, dt=0.1, runtime=10000,output_file=output)
output.close()

#try to do some plots
#import xarray as xr
#
#data_xarray = xr.open_dataset('test_2D_case.nc')
#xp = data_xarray['lon'].values
#yp = data_xarray['lat'].values
#
#fig, ax = plt.subplots(figsize=(15,10),facecolor='w')
#cs = ax.pcolor(x[1:Nx],y[1:Ny],psi[1:Nx,1:Ny])
#plt.streamplot(x[1:Nx],y[1:Ny],u[1:Nx,1:Ny],v[1:Nx,1:Ny],color='k')
#fig.colorbar(cs)
#for ii in range(xp.shape[0]):
#       plt.plot(xp[ii,],yp[ii,])
#plt.title('2D case Analytical')
#plt.savefig('2D_case.png', dpi=100)
#plt.show()
#
#fig, ax = plt.subplots(figsize=(15,10),facecolor='w')
#cs = ax.pcolor(x,y,psi)
#fig.colorbar(cs)
#plt.plot(xp[:,0],yp[:,0],'kx')#initial position
#plt.title('2D case Analytical: initial position')
#plt.savefig('2D_case_init.png', dpi=100)
#plt.show()
#
#fig, ax = plt.subplots(figsize=(15,10),facecolor='w')
#cs = ax.pcolor(x,y,psi)
#fig.colorbar(cs)
#plt.plot(xp[:,-1],yp[:,-1],'rx')#initial position
#plt.title('2D case Analytical: final position')
#plt.savefig('2D_case_final.png', dpi=100)
#plt.show()
