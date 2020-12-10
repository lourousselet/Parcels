%dimensions
xdim = 103;
ydim = 103;
Nx = xdim - 2; Ny = ydim - 2;
xi = linspace(0,Nx+1,xdim);
yi = linspace(0,Ny+1,ydim);
u = zeros(ydim,xdim);
v = zeros(ydim,xdim);
ue = zeros(ydim,xdim);
ve = zeros(ydim,xdim);
U = zeros(ydim,xdim);
V = zeros(ydim,xdim);
psi = zeros(ydim,xdim);
x1 = linspace(-1,Nx-1,Nx+1);
y1 = linspace(-1,Ny-1,Ny+1);
xe = max(x1);
L = max(y1);
dxG = diff(x1);
dyG = diff(y1);

%params
a = 0.8;
b = 1.2;
U0 = 0.5;

%psi for figure
for i = 2:Nx
    for j = 2:Ny
        psi(j,i) = sin(pi*(a*(x1(i)/xe)-b*(y1(j)/L)))*(x1(i)*(x1(i)-xe)*y1(j)*(y1(j)-L))^2;
    end
end

%u and v exact
for i = 2:Nx
    for j = 2:Ny
        X = x1(i)/xe; Y = y1(j)/L;
        ue(j,i) = -2*(X*(X-1))^2*(Y*(Y-1)^2+Y^2*(Y-1))*sin(pi*(a*X-b*Y)) + b*pi*cos(pi*(a*X-b*Y))*(X*(X-1)*Y*(Y-1))^2;
        ve(j,i) = 2*(Y*(Y-1))^2*(X*(X-1)^2+X^2*(X-1))*sin(pi*(a*X-b*Y)) + a*pi*cos(pi*(a*X-b*Y))*(X*(X-1)*Y*(Y-1))^2;
    end
end

%u and v incompressible
u = ue;
for i = 2:Nx
    for j = 2:Ny
        U(j,i) = u(j,i)*dyG(j);
        V(j,i) = V(j-1,i) - U(j,i) + U(j,i-1);
        v(j,i) = V(j,i)/dxG(i);
    end
end

