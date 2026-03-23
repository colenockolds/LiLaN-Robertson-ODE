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

Test_Data = jnp.log10(jnp.load("data/test_sol_data.npy"))[:, 1:, :]
Test_odeParameters = jnp.load("data/test_rates_data.npy")

print("Data:", Train_Data.shape, Validation_Data.shape, Test_Data.shape)

maxes = jnp.max(Train_odeParameters, axis=0)
mins = jnp.min(Train_odeParameters, axis=0)

Train_odeParameters = 2 * (Train_odeParameters - mins) / (maxes - mins) - 1
Validation_odeParameters = 2 * (Validation_odeParameters - mins) / (maxes - mins) - 1
Test_odeParameters = 2 * (Test_odeParameters - mins) / (maxes - mins) - 1

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
Train_odeParameters = Train_odeParameters[perm]

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

vmap_predict = vmap(predict, in_axes=(None, 0, None))

# Loss
def Loss(params, data, ic):
    conservation_constraint = 1
    pred = predict(params, ic, conservation_constraint)
    return 10**jnp.mean(jnp.abs(data[1:] - pred))

vmap_loss = vmap(Loss, in_axes=(None, 0, 0))

@jit
def Loss_Batch(params, data_batch, ic_batch):
    return jnp.mean(vmap_loss(params, data_batch, ic_batch))

# Load trained network

best_params, _, _ = pickle.load(open("Networks/best_params", "rb"))
opt_state = optimizers.pack_optimizer_state(best_params)
opt_int, opt_update, opt_get_params = optimizers.rmsprop_momentum(learning_rate)
all_params = opt_get_params(opt_state)

# Check Losses

print(Loss_Batch(all_params, Train_Data, Train_odeParameters))
print(Loss_Batch(all_params, Validation_Data, Validation_odeParameters))
print(Loss_Batch(all_params, Test_Data, Test_odeParameters))

# Prediction

predictions = vmap_predict(all_params, Test_odeParameters[:], 1.0)
print("Predictions", predictions.shape)

# Plot one sample

sample = 0
sample_predictions = 10**predictions[sample]
sample_truth = 10**Test_Data[sample]
physical_times = jnp.logspace(-5,5,50)[1:]

print(sample_predictions)
print(sample_truth)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(6, 8))

for i in range(dofs):
    ax = axes[i]
    ax.plot(physical_times[1:], sample_predictions[:,i], label=f'Species {i+1}, ML Solution')
    ax.plot(physical_times[1:], sample_truth[1:,i], 'k--', label=f'Species {i+1}, Numerical Solution')
    ax.set_xlim([physical_times[1], physical_times[-1]])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f'Species {i+1}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Concentration (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
