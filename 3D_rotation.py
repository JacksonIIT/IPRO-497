# Enter an initial height from the legs to the joint where rotation starts: 3
# Enter the value of phi that you want to test: 45

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rotate(θ, ρ, Φ, R, xZero, yZero, zZero, λ, ζ, ε, α, μ, κ, Ψ, H0, δ, χ, Σ, Π, Ξ, ψ):
    Ax = ρ * np.sin(np.radians(Φ)) * np.cos(np.radians(θ))
    Ay = ρ * np.sin(np.radians(Φ)) * np.sin(np.radians(θ))
    Az = ρ * np.cos(np.radians(Φ)) + H0

    Bx = -R * np.cos(np.radians(Φ)) * np.cos(np.radians(θ)) + Ax
    By = -R * np.cos(np.radians(Φ)) * np.sin(np.radians(θ)) + Ay
    Bz = R * np.sin(np.radians(Φ)) + Az

    Cx = λ * np.sin(np.radians(Φ)) * np.cos(np.radians(ζ)) + Bx
    Cy = λ * np.sin(np.radians(Φ)) * np.sin(np.radians(ζ)) + By
    Cz = λ * np.cos(np.radians(Φ)) * np.cos(np.radians(ζ)) + Bz

    CxN = -ε * np.sin(np.radians(Φ)) * np.cos(np.radians(ζ)) + Bx
    CyN = -ε * np.sin(np.radians(Φ)) * np.sin(np.radians(ζ)) + By
    CzN = -ε * np.cos(np.radians(Φ)) * np.cos(np.radians(ζ)) + Bz

    Dx = -ψ * np.cos(np.radians(Φ)) * np.cos(np.radians(α)) + Cx
    Dy = -ψ * np.cos(np.radians(Φ)) * np.sin(np.radians(α)) + Cy
    Dz = ψ * np.sin(np.radians(Φ)) * np.cos(np.radians(α)) + Cz

    DxN = κ * np.cos(np.radians(Φ)) * np.cos(np.radians(α)) + Cx
    DyN = κ * np.cos(np.radians(Φ)) * np.sin(np.radians(α)) + Cy
    DzN = -κ * np.sin(np.radians(Φ)) * np.cos(np.radians(α)) + Cz

    Ex = Cx +(δ /ψ)*Dx + χ * np.sin(np.radians(Φ)) * np.cos(np.radians(Σ))
    Ey = Cy +(δ /ψ)*Dy + χ* np.sin(np.radians(Φ)) * np.sin(np.radians(Σ))
    Ez = Cz +(δ /ψ)*Dz + χ* np.cos(np.radians(Φ)) *np.cos(np.radians(Σ))

    Fx = Bx+(Π/ε)*CxN + χ * np.cos(np.radians(Φ)) * np.cos(np.radians(Ξ))
    Fy = By+(Π/ε)*CyN + χ * np.cos(np.radians(Φ)) * np.sin(np.radians(Ξ))
    Fz = Bz+(Π/ε)*CzN + χ * np.sin(np.radians(Φ)) * np.cos(np.radians(Ξ))

    return Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, CxN, CyN, CzN, Dx, Dy, Dz, DxN, DyN, DzN, Ex, Ey, Ez, Fx, Fy, Fz

# Parameters
ρ = 5
R = 6
λ = 8
ε = 6
μ = 7
κ = 5
ψ = 7
Ξ = 5  # Another Fixed Thickness
χ = 2  # Fixed thickness
θ=0
Ψ = 10

H0 = float(input("Enter an initial height from the legs to the joint where rotation starts: "))
Φ = 30

# Loop through the angles
theta_step = 15
alpha_step = 15
zeta_step = 15
Sigma_step =15
E_step= 15

angle1 = 0
angle2 = 120
angle3 = 240

xZero, yZero, zZero = 0, 0, 0

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Lists to store points for each vector
A_points = []
B_points = []
C_points = []
C_neg_points = []
D_points = []
D_neg_points = []
E_points = []
F_points = []

plt.ion()
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Loop through all the parameter combinations
for ζ in range(0, 360, zeta_step):
    for α in range(0, 360, alpha_step):
        for Σ in range(0, 360, Sigma_step):
            for Ξ in range(0, 360, E_step):
                for δ in np.arange(-3, 0, 1.0):
                    for Π in np.arange(6, 7, .2):

                        Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, CxN, CyN, CzN, Dx, Dy, Dz, DxN, DyN, DzN, Ex, Ey, Ez, Fx, Fy, Fz = rotate(
                            θ, ρ, Φ, R, xZero, yZero, zZero, λ, ζ, ε, α, μ, κ, Ψ, H0, δ, χ, Σ, Π, Ξ, ψ
                        )

                            # Store points in their respective lists
                        A_points.append([Ax, Ay, Az])
                        B_points.append([Bx, By, Bz])
                        C_points.append([Cx, Cy, Cz])
                        C_neg_points.append([CxN, CyN, CzN])
                        D_points.append([Dx, Dy, Dz])
                        D_neg_points.append([DxN, DyN, DzN])
                        E_points.append([Ex, Ey, Ez])
                        F_points.append([Fx, Fy, Fz])

# Convert points into numpy arrays for easier slicing
A_points = np.array(A_points)
B_points = np.array(B_points)
C_points = np.array(C_points)
C_neg_points = np.array(C_neg_points)
D_points = np.array(D_points)
D_neg_points = np.array(D_neg_points)
E_points = np.array(E_points)
F_points = np.array(F_points)

# Define vectors in terms of Ψ, with a Z-component
N = np.array([Ψ * np.cos(np.radians(angle1)), Ψ * np.sin(np.radians(angle1)), -Ψ * 0.5])
omega = np.array([Ψ * np.cos(np.radians(angle2)), Ψ * np.sin(np.radians(angle2)), -Ψ * 0.5])
β = np.array([Ψ * np.cos(np.radians(angle3)), Ψ * np.sin(np.radians(angle3)), -Ψ * 0.5])

# Plot each vector as a line in 3D space
ax.plot(A_points[:, 0], A_points[:, 1], A_points[:, 2], label='Spherical Vector A', color='r', alpha=0.5)
ax.plot(B_points[:, 0], B_points[:, 1], B_points[:, 2], label='Spherical Vector B', color='g', alpha=0.5)
ax.plot(C_points[:, 0], C_points[:, 1], C_points[:, 2], label='Spherical Vector C', color='b', alpha=0.5)
ax.plot(C_neg_points[:, 0], C_neg_points[:, 1], C_neg_points[:, 2], label='Spherical Vector -C', color='c', alpha=0.5)
ax.plot(D_points[:, 0], D_points[:, 1], D_points[:, 2], label='Spherical Vector D', color='m', alpha=0.5)
ax.plot(D_neg_points[:, 0], D_neg_points[:, 1], D_neg_points[:, 2], label='Spherical Vector -D', color='y', alpha=0.5)
ax.plot(E_points[:, 0], E_points[:, 1], E_points[:, 2], label='Spherical Vector E', color='brown', alpha=0.5)
ax.plot(F_points[:, 0], F_points[:, 1], F_points[:, 2], label='Spherical Vector F', color='orange', alpha=0.5)

# Plot vector arrows for N, omega, and β
ax.quiver(0, 0, 0, N[0], N[1], N[2], color='orange', label='Leg 3', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, omega[0], omega[1], omega[2], color='purple', label='Leg 2', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, β[0], β[1], β[2], color='purple', label='Leg 1', arrow_length_ratio=0.1)

# Set labels and limits
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_xlim([-15, 15])
ax.set_ylim([-15, 15])
ax.set_zlim([-15, 15])

# Set title
ax.set_title('3D Graph with Rotating Vectors and Vectors N, Omega on -Z axis')
ax.legend()
ax.set_box_aspect([1, 1, 1])
plt.show()