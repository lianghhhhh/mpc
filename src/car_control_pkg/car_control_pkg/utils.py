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
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

def loadConfig():
    with open('/workspaces/config.json', 'r') as f:
        config = json.load(f)
    return config

def computeTarget(u_tensor, x_data, A, B, dt=0.1):
    x_dot = torch.bmm(A, x_data) + torch.bmm(B, u_tensor)  # x' = Ax + Bu
    next_x = x_data + x_dot * dt  # Euler integration: x_next = x + x' * dt
    return next_x

def normalize(data, name, scaler):
    if name == "u":
        data = scaler.transform(data)
    elif name == "state": # x, z positions only
        data = np.array(data)
        data[:2] = scaler.transform(data[:2].reshape(-1, 2)).reshape(data[:2].shape)
    elif name == "path":
        data = scaler.transform(data)
    return data

def denormalize(data, name, scaler):
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

def loadModelFunc(model_path):
    model = CarPredictor()
    model.load_state_dict(torch.load(f'{model_path}/model_6.pth'))
    device = 'cuda' # if torch.cuda.is_available() else 'cpu'
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

    return nn_model_func, l4c_model.shared_lib_dir, l4c_model.name

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

def createAcadosSolver(nn_model_func, lib_dir, lib_name, N=10):
    ocp = AcadosOcp()
    model = AcadosModel()
    model.name = 'car_model'

    x = ca.SX.sym('x', 4)
    u = ca.SX.sym('u', 4)
    p = ca.SX.sym('p', 2)
    input_sym = ca.vertcat(u, x)
    x_next = nn_model_func(input_sym)

    model.x = x
    model.u = u
    model.p = p
    model.disc_dyn_expr = x_next
    ocp.model = model

    ocp.solver_options.N_horizon = N
    Tf = N * 0.1  # (N steps) * (0.1 s/step)
    ocp.solver_options.tf = Tf

    Q_pos = np.diag([1000.0, 1000.0]) # Position cost
    R_ctrl = np.diag([0.0001, 0.0001, 0.0001, 0.0001]) # Control effort cost
    W_speed = 0.01

    position_error = x[:2] - p

    stage_cost_expr = ca.mtimes([position_error.T, ca.DM(Q_pos), position_error]) \
                      + ca.mtimes([u.T, ca.DM(R_ctrl), u])
    terminal_cost_expr = ca.mtimes([position_error.T, ca.DM(Q_pos), position_error])

    # --- Set cost for STAGE 0 ---
    ocp.cost.cost_type_0 = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_0 = stage_cost_expr

    # --- Set cost for STAGES 1 to N-1 ---
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost = stage_cost_expr

    # --- Set cost for STAGE N (Terminal) ---
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_e = terminal_cost_expr   

    ocp.constraints.x0 = np.zeros(4)
    ocp.parameter_values = np.zeros(2)
    ocp.constraints.lbu = np.array([-1.0, -1.0, -1.0, -1.0])
    ocp.constraints.ubu = np.array([1.0, 1.0, 1.0, 1.0])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.model_external_shared_lib_dir = lib_dir
    ocp.solver_options.model_external_shared_lib_name = lib_name

    acados_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    return acados_solver