import time
import pickle

import jax
import jax.numpy as jnp
from jax.nn import softmax
from jax.nn.initializers import normal, glorot_normal
from jax import jit, vmap, value_and_grad, lax, random
from jax.example_libraries import stax, optimizers

rng = random.PRNGKey(20)
weight_initializer = glorot_normal()
bias_initializer = normal()
depth = 2
time_discretization_points = 49
Nr = 5 # dimension of r
lipschitz_constants_init = jnp.ones(Nr)
dofs = 3
ODE_dim = 1 # original problem dimension
parameter_dim = 3 # dimensionality of parameters
enc_i_input_size = 3
enc_s_input_size = 3
enc_l_input_size = 4
hidden_dim = 20
tspan = [0,1]
times = jnp.linspace(tspan[0],tspan[1],time_discretization_points)
dt = (times[1] - times[0]) / tspan[1]
interp = 0

## Load Data
Train_Data = jnp.log10(jnp.load("data/train_sol_data.npy"))[:, 1:, :]
Train_odeParameters = jnp.load("data/train_rates_data.npy")

Validation_Data = jnp.log10(jnp.load("data/validation_sol_data.npy"))[:, 1:, :]
Validation_odeParameters = jnp.load("data/validation_rates_data.npy")

print("Data:", Train_Data.shape, Validation_Data.shape)

maxes = jnp.max(Train_odeParameters, axis=0)
mins = jnp.min(Train_odeParameters, axis=0)

Train_odeParameters = 2 * (Train_odeParameters - mins) / (maxes - mins) - 1
Validation_odeParameters = 2 * (Validation_odeParameters - mins) / (maxes - mins) - 1

# Training Settings
act = 'tanh'
num_samples = Train_Data.shape[0] + Validation_Data.shape[0]
num_train = Train_Data.shape[0]
num_test = Validation_Data.shape[0]
batch_size = num_train // 20
num_epochs = 500
learning_rate = 1e-5
alpha = 1
plot = 1
interval = 100000000
PRINT_EVERY = 10

# Shuffle Data
perm = random.permutation(rng, Train_Data.shape[0])
Train_Data = Train_Data[perm]
Train_rates = Train_odeParameters[perm]

Sol_Data = jnp.vstack((Train_Data[:num_train], Validation_Data))
rates = jnp.vstack((Train_odeParameters[:num_train], Validation_odeParameters))
num_samples = Sol_Data.shape[0]

## Calculate Mean

@jit
def param_vector(params):
    return jnp.concatenate(tuple([p.flatten() for p in jax.tree_util.tree_flatten(params)[0]]))

## Networks

ei_param_list = []
es_param_list = []
el_param_list = []
dec_param_list = []
if act == 'tanh':
    _init, Intercept_Encoder = stax.serial(
        stax.Dense(hidden_dim),
        stax.Tanh,
        stax.Dense(Nr),
    )
    for dof in range(dofs):
        _, init_enc_params_intercept = _init(rng, (enc_i_input_size,))
        ei_param_list.append(param_vector(init_enc_params_intercept))
    ei_def = jax.tree_util.tree_flatten(init_enc_params_intercept)[1]

    _init, Slope_Encoder = stax.serial(
        stax.Dense(hidden_dim),
        stax.Tanh,
        stax.Dense(Nr),
    )
    for dof in range(dofs):
        _, init_enc_params_slope = _init(rng, (enc_s_input_size,))
        es_param_list.append(param_vector(init_enc_params_slope))
    es_def = jax.tree_util.tree_flatten(init_enc_params_slope)[1]

    _init, Time_Encoder = stax.serial(
        stax.Dense(hidden_dim),
        stax.Tanh,
        stax.Dense(Nr),
    )
    for dof in range(dofs):
        _, init_enc_params_lip = _init(rng, (enc_l_input_size,))
        el_param_list.append(param_vector(init_enc_params_lip))
    el_def = jax.tree_util.tree_flatten(init_enc_params_lip)[1]

    _init, Decoder = stax.serial(
        stax.Dense(hidden_dim),
        stax.Tanh,
        stax.Dense(hidden_dim),
        stax.Tanh,
        stax.Dense(ODE_dim),
    )
    for dof in range(dofs):
        _, init_dec_params = _init(rng, (Nr,))
        dec_param_list.append(param_vector(init_dec_params))
    d_def = jax.tree_util.tree_flatten(init_dec_params)[1]

A = weight_initializer(rng, (dofs, Nr))

print("PARAMS:", jnp.asarray(ei_param_list).size + jnp.asarray(es_param_list).size + jnp.asarray(el_param_list).size + A.size + jnp.asarray(dec_param_list).size)

ei_params = jnp.asarray(ei_param_list)
es_params = jnp.asarray(es_param_list)
el_params = jnp.asarray(el_param_list)
d_params = jnp.asarray(dec_param_list)

init_params = (ei_params, es_params, el_params, A, d_params)

@jit
def rebuild_encoder(param_vec):
    # Checkpoints for Network
    c1 = enc_i_input_size * hidden_dim
    c2 = c1 + hidden_dim
    c3 = c2 + hidden_dim * Nr
    c4 = c3 + Nr
    
    W1_e = param_vec[:c1].reshape((enc_i_input_size,hidden_dim))
    b1_e = param_vec[c1:c2]
    W2_e = param_vec[c2:c3].reshape((hidden_dim,Nr))
    b2_e = param_vec[c3:c4]
    e_list = [W1_e, b1_e, W2_e, b2_e]

    e_params = jax.tree_util.tree_unflatten(ei_def, e_list)

    return e_params

@jit
def rebuild_t_encoder(param_vec):
    # Checkpoints for Network
    c1 = enc_l_input_size * hidden_dim
    c2 = c1 + hidden_dim
    c3 = c2 + hidden_dim * Nr
    c4 = c3 + Nr
    
    W1_e = param_vec[:c1].reshape((enc_l_input_size,hidden_dim))
    b1_e = param_vec[c1:c2]
    W2_e = param_vec[c2:c3].reshape((hidden_dim,Nr))
    b2_e = param_vec[c3:c4]
    e_list = [W1_e, b1_e, W2_e, b2_e]

    e_params = jax.tree_util.tree_unflatten(ei_def, e_list)

    return e_params

@jit
def rebuild_decoder(param_vec):
    # Checkpoints for Network
    c8 = 0
    c9 = c8 + Nr * hidden_dim
    c10 = c9 + hidden_dim
    c11 = c10 + hidden_dim * hidden_dim
    c12 = c11 + hidden_dim
    c13 = c12 + hidden_dim * ODE_dim
    c14 = c13 + ODE_dim

    W1_d = param_vec[c8:c9].reshape((Nr,hidden_dim))
    b1_d = param_vec[c9:c10]
    W2_d = param_vec[c10:c11].reshape((hidden_dim,hidden_dim))
    b2_d = param_vec[c11:c12]
    W3_d = param_vec[c12:c13].reshape((hidden_dim,ODE_dim))
    b3_d = param_vec[c13:c14]
    d_list = [W1_d, b1_d, W2_d, b2_d, W3_d, b3_d]

    d_params = jax.tree_util.tree_unflatten(d_def, d_list)

    return d_params

def decode(params, x):
    return Decoder(params, x)

vmap_decode = vmap(decode, in_axes=(None, 0))

def time_net(t_p, y0, t, cons):
    inputs = jnp.concatenate((y0, jnp.array([t])))
    return (jnp.abs(cons) + 1e-3) * jnp.exp(Time_Encoder(t_p, inputs))

vmap_time = vmap(time_net, in_axes=(None, None, 0, None)) 

def reservoir(r, dr, t):
    return r + dr * t

vmap_r = vmap(reservoir, in_axes=(None, None, 0))

def dof_loop(i, args):
    pred, y0, params = args
    i_enc_p, s_enc_p, l_enc_p, A_, d_p = params

    dof_i_enc_p = rebuild_encoder(i_enc_p[i])
    dof_s_enc_p = rebuild_encoder(s_enc_p[i])
    dof_l_enc_p = rebuild_t_encoder(l_enc_p[i])
    dof_d_p = rebuild_decoder(d_p[i])

    r0 = Intercept_Encoder(dof_i_enc_p, y0)
    m = Slope_Encoder(dof_s_enc_p, y0)
    ts = vmap_time(dof_l_enc_p, y0, times, A_[i])

    dof_res = vmap_r(r0, m, ts)

    dof_pred = jnp.squeeze(vmap_decode(dof_d_p, dof_res[1:]))
    pred = pred.at[:,i].set(dof_pred)

    return pred, y0, params 

vmap_softmax = vmap(softmax)

def predict(params, y0, constraint):
    pred = jnp.empty((time_discretization_points - 1, dofs))
    pred, _, _ = lax.fori_loop(0, dofs, dof_loop, (pred, y0, params))
    pred = jnp.log10(constraint * vmap_softmax(pred))
    return pred

# Loss
def Loss(params, data, ic):
    conservation_constraint = 1
    pred = predict(params, ic, conservation_constraint)
    return 10**jnp.mean(jnp.abs(data[1:] - pred))

vmap_loss = vmap(Loss, in_axes=(None, 0, 0))

@jit
def Loss_Batch(params, data_batch, ic_batch):
    return jnp.mean(vmap_loss(params, data_batch, ic_batch))

#@jit
def body_fun(i, args):
    loss, opt_state = args

    data_batch = lax.dynamic_slice_in_dim(Train_Data, i * batch_size, batch_size)
    ic_batch = lax.dynamic_slice_in_dim(Train_rates, i * batch_size, batch_size)

    loss, gradients = value_and_grad(Loss_Batch)(
        opt_get_params(opt_state), data_batch, ic_batch)

    opt_state = opt_update(i, gradients, opt_state)

    return loss, opt_state

@jit
def run_epoch(opt_state):
    loss = 0
    return lax.fori_loop(0, num_batches, body_fun, (loss, opt_state))

import matplotlib.pyplot as plt

def Train_Model(opt_state):

    test_accuracy_min = 1e300
    train_accuracy_min = 1e300
    epoch_min = 1
    epoch_min_train = 1

    test_accuracy = 0

    for epoch in range(1, num_epochs+1):
        
        t1 = time.time()
        train_loss, opt_state = run_epoch(opt_state)
        test_accuracy = Loss_Batch(opt_get_params(opt_state), Validation_Data, Validation_odeParameters)
        t2 = time.time()

        if train_accuracy_min >= train_loss:
            train_accuracy_min = train_loss
            epoch_min_train = epoch

        if test_accuracy_min >= test_accuracy:
            test_accuracy_min = test_accuracy
            epoch_min = epoch
            optimal_opt_state = opt_state

        if epoch % PRINT_EVERY == 0:
            print("Data_d {:d} batch {:d} time {:.3e}s loss {:.3e} loss_min {:.3e} EPminT {:d} TE {:.3e} TE_min {:.3e} EPmin {:d} EP {} ".format(
                num_train, batch_size, t2 - t1, train_loss, train_accuracy_min, epoch_min_train, test_accuracy, test_accuracy_min, epoch_min, epoch))

    return optimal_opt_state

num_complete_batches, leftover = divmod(num_train, batch_size)
num_batches = num_complete_batches + bool(leftover)

## Train from new initialization
opt_int, opt_update, opt_get_params = optimizers.rmsprop_momentum(learning_rate)
opt_state = opt_int(init_params)

best_opt_state = Train_Model(opt_state)

optimum_params = opt_get_params(best_opt_state)

filename = "best_params"

trained_params = optimizers.unpack_optimizer_state(best_opt_state)
pickle.dump((trained_params, mins, maxes), open('Networks/' + filename, "wb"))
