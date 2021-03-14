from fenics import *
from mshr import *


T = 5.0            # final time
num_steps = 5000  # number of time steps
dt = T / num_steps # time step size
mu = 0.001         # dynamic viscosity
rho = 1            # density



channel = Rectangle(Point(0, 0), Point(2.2, 0.45))

cylinder1 = Circle(Point(0.2, 0.15), 0.03)
cylinder2 = Circle(Point(0.2, 0.3), 0.03)

cylinder3 = Circle(Point(0.35, 0.1125), 0.03)
cylinder4 = Circle(Point(0.35, 0.225), 0.03)
cylinder5 = Circle(Point(0.35, 0.3375), 0.03)

cylinder6 = Circle(Point(0.5, 0.15), 0.03)
cylinder7 = Circle(Point(0.5, 0.3), 0.03)



domain = channel - cylinder1 - cylinder2 - cylinder3 - cylinder4 - cylinder5 - cylinder6 - cylinder7
mesh = generate_mesh(domain, 64)

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define boundaries
inflow   = 'near(x[0], 0)'
outflow  = 'near(x[0], 2.2)'
walls    = 'near(x[1], 0) || near(x[1], 0.45)'


cylinder1 = 'on_boundary && x[0]>0.14 && x[0]<0.26 && x[1]>0.12 && x[1]<0.18'
cylinder2 = 'on_boundary && x[0]>0.14 && x[0]<0.26 && x[1]>0.27 && x[1]<0.33'

cylinder3 = 'on_boundary && x[0]>0.29 && x[0]<0.41 && x[1]>0.0825 && x[1]<0.1425'
cylinder4 = 'on_boundary && x[0]>0.29 && x[0]<0.41 && x[1]>0.1950 && x[1]<0.255'
cylinder5 = 'on_boundary && x[0]>0.29 && x[0]<0.41 && x[1]>0.3075 && x[1]<0.3675'

cylinder6 = 'on_boundary && x[0]>0.44 && x[0]<0.56 && x[1]>0.12 && x[1]<0.18'
cylinder7 = 'on_boundary && x[0]>0.44 && x[0]<0.56 && x[1]>0.27 && x[1]<0.33'

# Define inflow profile
inflow_profile = ('1', '0')

# Define boundary conditions
bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
bcu_walls = DirichletBC(V, Constant((0, 0)), walls)

bcu_cylinder1 = DirichletBC(V, Constant((0, 0)), cylinder1)
bcu_cylinder2 = DirichletBC(V, Constant((0, 0)), cylinder2)

bcu_cylinder3 = DirichletBC(V, Constant((0, 0)), cylinder3)
bcu_cylinder4 = DirichletBC(V, Constant((0, 0)), cylinder4)
bcu_cylinder5 = DirichletBC(V, Constant((0, 0)), cylinder5)

bcu_cylinder6 = DirichletBC(V, Constant((0, 0)), cylinder6)
bcu_cylinder7 = DirichletBC(V, Constant((0, 0)), cylinder7)

bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bcu = [bcu_inflow, bcu_walls, bcu_cylinder1, bcu_cylinder2, bcu_cylinder3, bcu_cylinder4, bcu_cylinder5, bcu_cylinder6, bcu_cylinder7]
bcp = [bcp_outflow]

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Define functions for solutions at previous and current time steps
u_n = Function(V)
u_  = Function(V)
p_n = Function(Q)
p_  = Function(Q)

# Define expressions used in variational forms
U  = 0.5*(u_n + u)
n  = FacetNormal(mesh)
f  = Constant((0, 0))
k  = Constant(dt)
mu = Constant(mu)
rho = Constant(rho)

# Define symmetric gradient
def epsilon(u):
    return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))

# Define variational problem for step 1
F1 = rho*dot((u - u_n) / k, v)*dx \
   + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
   + inner(sigma(U, p_n), epsilon(v))*dx \
   + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
   - dot(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

# Define variational problem for step 3
a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

# VTK files for visualization
file_u = File('navier_stokes_cylinder/velocity.pvd')
file_p = File('navier_stokes_cylinder/pressure.pvd')

# Time-stepping
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1, 'bicgstab', 'petsc_amg')

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2, 'bicgstab', 'petsc_amg')

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3, 'cg', 'sor')

    # Save solution to file (XDMF/HDF5)
    file_u << u_
    file_p << p_

    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)

    print("Current time: %f / %f" % (t, T))
