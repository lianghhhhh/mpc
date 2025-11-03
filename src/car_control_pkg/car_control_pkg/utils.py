import os
import csv
import json
import torch
import numpy as np
import casadi as ca
import l4casadi as l4c
from sklearn.preprocessing import StandardScaler
from car_control_pkg.car_predictor import CarPredictor

def getInputData(data_path):
    u_val = []
    x_val = []

    with open(data_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            u_val.append([[float(row[1])], [float(row[2])], [float(row[3])], [float(row[4])]])
            angle = np.radians(float(row[8]))
            x_val.append([[float(row[5])], [float(row[7])], [np.sin(angle)], [np.cos(angle)]])

    u_val, u_scaler = normalize(np.array(u_val), "u")
    x_val, x_scaler = normalize(np.array(x_val), "x")

    u_val = u_val[:-1]
    x_next_val = x_val[1:]
    x_val = x_val[:-1]

    train_size = int(0.8 * len(u_val))
    train_u = u_val[:train_size]
    train_x = x_val[:train_size]
    train_x_next = x_next_val[:train_size]

    test_u = u_val[train_size:]
    test_x = x_val[train_size:]
    test_x_next = x_next_val[train_size:]

    return train_u, train_x, train_x_next, test_u, test_x, test_x_next, u_scaler, x_scaler

def loadConfig():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def computeTarget(u_tensor, x_data, A, B, dt=0.01):
    x_dot = torch.bmm(A, x_data) + torch.bmm(B, u_tensor)  # x' = Ax + Bu
    next_x = x_data + x_dot * dt  # Euler integration: x_next = x + x' * dt

    return next_x

def normalize(data, name="data"):
    if name == "x":
        # normalize on first 2 features only (x,z positions)
        scaler = StandardScaler()
        data[:, :, :2] = scaler.fit_transform(data[:, :, :2].reshape(-1, 2)).reshape(data[:, :, :2].shape)
    else:
        scaler = StandardScaler()
        original_shape = data.shape
        data_reshaped = data.reshape(-1, original_shape[-1])
        data_normalized = scaler.fit_transform(data_reshaped)
        data = data_normalized.reshape(original_shape)
    return data, scaler

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
    model.load_state_dict(torch.load('/home/selena/mpc/model.pth'))
    l4c_model = l4c.L4CasADi(model, device='cuda')
    x_sym = ca.SX.sym('x', 4)
    u_sym = ca.SX.sym('u', 4)
    A_sym, B_sym = l4c_model(u_sym, x_sym)
    x_next_sym = x_sym + (ca.mtimes(A_sym, x_sym) + ca.mtimes(B_sym, u_sym)) * 0.01
    nn_model_func = ca.Function('nn_model_func', [x_sym, u_sym], [x_next_sym], ['x', 'u'], ['x_next'])
    return nn_model_func

def createMpcSolver(nn_model_func, N=10):
    opti = ca.Opti()

    u_pred = opti.variable(N, 4)  # Control inputs over the horizon
    next_x_pred = opti.variable(N + 1, 4)  # Predicted states over the horizon

    current_x = opti.parameter(4) # Current state parameter
    target_path = opti.parameter(N+1, 2) # Target path parameter

    cost = 0
    state_cost_weight = ca.diag(ca.DM([10, 10, 1, 1]))
    control_cost_weight = ca.diag(ca.DM([1, 1, 0.1, 0.1]))
    for t in range(N):
        position_error = next_x_pred[t, :2] - target_path[t, :]
        cost += ca.mtimes([position_error.T, state_cost_weight[:2, :2], position_error])
        control_effort = u_pred[t, :]
        cost += ca.mtimes([control_effort.T, control_cost_weight, control_effort])

    opti.minimize(cost)

    for t in range(N):
        x_t = next_x_pred[t, :]
        u_t = u_pred[t, :]
        x_next_pred_t = nn_model_func(x_t, u_t)
        opti.subject_to(next_x_pred[t+1, :] == x_next_pred_t)
    opti.subject_to(next_x_pred[0, :] == current_x)
    opts = {"ipopt.print_level":0, "print_time":0}
    opti.solver('ipopt', opts)
    return opti, u_pred, next_x_pred, current_x, target_path