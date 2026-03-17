import torch
import torch.optim as optim
import numpy as np

# Import the architecture and physics rules we just built
from models.pinn import OpticalEventHorizonPINN
from physics.wave_eq import nlse_physics_loss

def generate_training_data(num_collocation=10000, num_boundary=2000, L=10.0, T=5.0):
    """Generates spacetime points using Adaptive Importance Sampling."""
    
    # --- ADAPTIVE SAMPLING FOR SPACE (x) ---
    # 70% of points clustered around the high-error center (x=5)
    num_hard_points = int(num_collocation * 0.7)
    num_easy_points = num_collocation - num_hard_points
    
    # Hard points: Normal distribution centered at L/2 (x=5), with a tight standard deviation
    x_hard = torch.normal(mean=L/2, std=1.0, size=(num_hard_points, 1))
    # Clip the points so they don't fall outside the fiber [0, 10]
    x_hard = torch.clamp(x_hard, min=0.0, max=L) 
    
    # Easy points: Uniform distribution everywhere else
    x_easy = torch.rand(num_easy_points, 1) * L
    
    # Combine and shuffle
    x_col = torch.cat([x_hard, x_easy], dim=0)
    x_col = x_col[torch.randperm(num_collocation)] 
    
    # Time (t) remains uniformly distributed
    t_col = torch.rand(num_collocation, 1) * T
    
    # 2. Initial Condition Points (t = 0, random x)
    x_ic = torch.rand(num_boundary, 1) * L
    t_ic = torch.zeros(num_boundary, 1)
    
    u_ic_exact = torch.exp(-0.5 * ((x_ic - L/2) / 0.5)**2)
    v_ic_exact = torch.zeros_like(x_ic)
    
    # 3. Boundary Condition Points (x = 0 and x = L, random t)
    t_bc = torch.rand(num_boundary, 1) * T
    x_bc_left = torch.zeros(num_boundary, 1)
    x_bc_right = torch.ones(num_boundary, 1) * L
    
    return (x_col, t_col), (x_ic, t_ic, u_ic_exact, v_ic_exact), (x_bc_left, x_bc_right, t_bc)

def train_model():
    # Initialize the model and optimizer
    model = OpticalEventHorizonPINN(num_layers=6, hidden_dim=100)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Get the training data
    col_data, ic_data, bc_data = generate_training_data()
    x_col, t_col = col_data
    x_ic, t_ic, u_ic_exact, v_ic_exact = ic_data
    x_bc_left, x_bc_right, t_bc = bc_data
    
    epochs = 5000
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # --- 1. Physics Loss (The NLSE inside the fiber) ---
        loss_physics = nlse_physics_loss(model, x_col, t_col)
        
        # --- 2. Initial Condition Loss (Matching the starting pulse) ---
        u_ic_pred, v_ic_pred = model(x_ic, t_ic)
        loss_ic = torch.mean((u_ic_pred - u_ic_exact)**2) + torch.mean((v_ic_pred - v_ic_exact)**2)
        
        # --- 3. Boundary Condition Loss (Rigid walls at the ends) ---
        u_left, v_left = model(x_bc_left, t_bc)
        u_right, v_right = model(x_bc_right, t_bc)
        loss_bc = torch.mean(u_left**2 + v_left**2) + torch.mean(u_right**2 + v_right**2)
        
        # --- Total Loss & Backpropagation ---
        total_loss = loss_physics + loss_ic + loss_bc
        total_loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:04d} | Total Loss: {total_loss.item():.6f} | "
                  f"Physics: {loss_physics.item():.6f} | IC: {loss_ic.item():.6f}")

    # Save the model artifact for the analytics dashboard
    torch.save(model.state_dict(), "data/processed/event_horizon_pinn.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train_model()
