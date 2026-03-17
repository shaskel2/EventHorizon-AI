import torch

def get_gradients(outputs, inputs):
    """
    Computes the partial derivative of outputs with respect to inputs 
    using PyTorch's autograd engine.
    """
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True
    )[0]

def nlse_physics_loss(model, x, t):
    """
    Calculates the physics residual for the 1D Nonlinear Schrödinger Equation.
    This simulates the laser pulse propagation in the optical fiber.
    """
    # 1. Ensure our spatial and temporal inputs are tracking gradients
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    # 2. Query the neural network for the real (u) and imaginary (v) fields
    u, v = model(x, t)
    
    # 3. Calculate First-Order Derivatives (Time evolution and spatial gradients)
    u_t = get_gradients(u, t)
    v_t = get_gradients(v, t)
    
    u_x = get_gradients(u, x)
    v_x = get_gradients(v, x)
    
    # 4. Calculate Second-Order Spatial Derivatives (Dispersion/Curvature)
    u_xx = get_gradients(u_x, x)
    v_xx = get_gradients(v_x, x)
    
    # 5. The NLSE Residuals
    # The standard NLSE is: i * psi_t + 0.5 * psi_xx + |psi|^2 * psi = 0
    # Because psi = u + iv, we split this into Real (f_u) and Imaginary (f_v) parts:
    
    intensity = u**2 + v**2
    
    f_u = -v_t + 0.5 * u_xx + intensity * u
    f_v =  u_t + 0.5 * v_xx + intensity * v
    
    # 6. The loss is the Mean Squared Error of these residuals (we want them driven to 0)
    loss_physics = torch.mean(f_u**2) + torch.mean(f_v**2)
    
    return loss_physics
