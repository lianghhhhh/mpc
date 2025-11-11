import os
import csv
import json
import torch
import joblib
import numpy as np
import casadi as ca
import l4casadi as l4c
from sklearn.preprocessing import StandardScaler
from car_control_pkg.car_predictor import CarPredictor

def loadConfig():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def computeTarget(u_tensor, x_data, A, B, dt=0.1):
    x_dot = torch.bmm(A, x_data) + torch.bmm(B, u_tensor)  # x' = Ax + Bu
    next_x = x_data + x_dot * dt  # Euler integration: x_next = x + x' * dt
    return next_x

def normalize(data, name, scaler=None):
    if scaler is None:
        scaler_path = f'/workspaces/model_6/{name}_scaler.save'
        scaler = joblib.load(scaler_path)
    if name == "u":
        data = scaler.transform(data)
    elif name == "state": # x, z positions only
        data = np.array(data)
        data[:2] = scaler.transform(data[:2].reshape(-1, 2)).reshape(data[:2].shape)
    elif name == "path":
        data = scaler.transform(data)
    return data

def denormalize(data, name, scaler=None):
    if scaler is None:
        scaler_path = f'/workspaces/model_6/{name}_scaler.save'
        scaler = joblib.load(scaler_path)
    if name == "u":
        data = scaler.inverse_transform(data.reshape(-1, 4)).reshape(data.shape)
    elif name == "x":
        data[:2] = scaler.inverse_transform(data[:2])
    return data

def angleToDegree(data):
    sin_component = data[:, 2, :]
    cos_component = data[:, 3, :]
    angles = np.arctan2(sin_component, cos_component)
    angles_degrees = np.degrees(angles)
    angles_degrees = (angles_degrees + 360) % 360  # Normalize to [0, 360)
    data[:, 2, :] = angles_degrees
    data[:, 3, :] = 0  # set the cosine component to zero
    return data

def loadModelFunc():
    model = CarPredictor()
    model.load_state_dict(torch.load('/workspaces/model_6/model_6.pth'))
    device = 'cpu' # if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    l4c_model = l4c.L4CasADi(model, device=device)
    u_sym = ca.SX.sym('u', 4)
    x_sym = ca.SX.sym('x', 4)
    input_sym = ca.vertcat(u_sym, x_sym)  # Shape (8, 1)
    output_sym = l4c_model(input_sym)
    A_sym = output_sym[:, :16].reshape((4, 4))  # A: shape (4,4)
    B_sym = output_sym[:, 16:].reshape((4, 4))  # B: shape (4,4)
    x_next_sym = x_sym + (ca.mtimes(A_sym, x_sym) + ca.mtimes(B_sym, u_sym)) * 0.1
    nn_model_func = ca.Function('nn_model_func', [input_sym], [x_next_sym], ['input'], ['x_next'])
    return nn_model_func

def createMpcSolver(nn_model_func, N=10):
    opti = ca.Opti()

    u_pred = opti.variable(N, 4)  # Control inputs over the horizon
    next_x_pred = opti.variable(N + 1, 4)  # Predicted states over the horizon

    current_x = opti.parameter(4) # Current state parameter
    target_path = opti.parameter(N+1, 2) # Target path parameter

    cost = 0
    state_cost_weight = ca.diag(ca.DM([10, 10, 1, 1]))
    control_cost_weight = ca.diag(ca.DM([0.01, 0.01, 0.01, 0.01]))
    avg_velocity = 0.01 # desired average velocity
    for t in range(N+1):
        position_error = next_x_pred[t, :2] - target_path[t, :]
        cost += ca.mtimes([position_error, state_cost_weight[:2, :2], position_error.T]) # position error cost
    
    for t in range(N):
        control_effort = u_pred[t, :]
        cost += ca.mtimes([control_effort, control_cost_weight, control_effort.T])

    opti.minimize(cost)

    for t in range(N):
        u_t = u_pred[t, :]
        x_t = next_x_pred[t, :]
        input_t = ca.vertcat(u_t.T, x_t.T) # Shape (8, 1)
        x_next_pred_t = nn_model_func(input_t)
        opti.subject_to(next_x_pred[t+1, :] == x_next_pred_t.T)
    opti.subject_to(next_x_pred[0, :] == current_x.T)
    opti.subject_to(opti.bounded(-1.0, u_pred, 1.0))
    opts = {"ipopt.print_level":0, "print_time":0}
    opti.solver('ipopt', opts)
    return opti, u_pred, next_x_pred, current_x, target_path