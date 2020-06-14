#%%
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
style.use('seaborn')

#%% Helpers
def sample_swiss_roll(n, low=np.pi, high=5 * np.pi, mean=0, std=0.7, plot=False):
    psi = np.random.uniform(low, high, n)
    v1 = mean + std * np.random.randn(n)
    v2 = mean + std * np.random.randn(n)
    x1 = (psi + v1) * np.cos(psi + v2)
    x2 = (psi + v1) * np.sin(psi + v2)

    # Dataset Plot
    if plot:
        plt.figure(figsize=(5, 5))
        plt.figure()
        plt.scatter(x1, x2, c=psi / np.pi, cmap=ListedColormap(sns.color_palette("gist_rainbow", 256)))
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        plt.xlim(-17, 17)
        plt.ylim(-17, 17)
        cbar = plt.colorbar(label=r"target values t in multiples of $\pi$")
        plt.savefig(plot, dpi=400)

    return np.vstack((np.ones(len(x1)), x1, x2, np.sqrt(x1**2 + x2**2))), psi


def plot_precision(y, t, method, filename):
    plt.figure(figsize=(5,5))
    plt.scatter(y, t)
    plt.xlabel(r"True $\psi$")
    plt.ylabel(r"Estimated $\psi$")
    plt.title(f"Precision with {method}")
    plt.plot([1, 19], [1, 19], color="green")
    plt.xlim(1, 19)
    plt.ylim(1, 19)
    plt.axes().set_aspect('equal')
    plt.grid
    plt.savefig(filename, dpi=400)

def kernel(phi, phi_test, sigma_K=False):
    if not sigma_K:
        K = phi.T @ phi
        K_test = phi.T @ phi_test
    else:
        raise NotImplementedError

    return K, K_test


#%% Optimizers
def GD(phi, t, phi_test, t_test, C=1, eps=0.1, maxit=10000, tol=1e-4, lr=1e-5):
    cost = [np.inf]
    w = np.zeros(phi.shape[0])
    for i in range(maxit):
        e = w.T@phi - t
        grad = phi * np.sign(e)
        grad[:, np.abs(e) < eps] = 0
        w -= lr * (np.sum(grad, axis=1) + w)

        cost.append((np.sum(np.abs(e[np.abs(e) > eps]) - eps) + w@w/2))
        if np.abs(cost[-2] - cost[-1]) < tol:
            break

    return (w@phi, w@phi_test), cost


def AGD(phi, t, phi_test, t_test, C=1, eps=0.1, maxit=1000, tol=1e-4, lr=1e-5):
    tk = 0
    w = np.zeros(phi.shape[0])
    w_prev = w.copy()
    cost = [np.inf]
    for k in range(maxit):
        t_next = (1 + np.sqrt(1 + 4 * tk**2)) / 2
        v = w + (tk - 1) / (t_next) * (w - w_prev)
        e = v.T@phi - t
        grad = phi * np.sign(e)
        grad[:, np.abs(e) < eps] = 0

        w_prev = w.copy()
        w = v - lr * (np.sum(grad, axis=1) + v)
        e_w = w.T@phi - t
        cost.append((np.sum(np.abs(e_w[np.abs(e_w) > eps]) - eps) + w@w / 2))
        tk = t_next
        if np.abs(cost[-2] - cost[-1]) < tol:
            break

    return (w@phi, w@phi_test), cost


def PG(phi, t, phi_test, t_test, sigma_K=False, C=1, eps=0.1, maxit=10000, tol=1e-10, lr=1e-5):
    K, K_test = kernel(phi, phi_test, sigma_K)
    a = np.zeros(len(t))
    b = np.zeros(len(t))
    cost = [np.inf]

    for k in range(maxit):
        v_a = a - lr * (K @ (a-b) + eps - t)
        v_b = b - lr * (K @ (b-a) + eps + t)
        a = np.clip(v_a, 0, 1)
        b = np.clip(v_b, 0, 1)

        cost.append((-1/2 * (a-b) @ K @ (a-b) - eps*np.sum(a+b) + np.sum(t*(a-b))))
        if np.abs(cost[-2] - cost[-1]) < tol:
            break

    return ((a - b) @ K, (a - b) @ K_test), cost


def FISTA(phi, t, phi_test, t_test, sigma_K=False, C=1, eps=0.1, maxit=10000, tol=1e-10, lr=1e-5):
    K, K_test = kernel(phi, phi_test, sigma_K)
    a = np.zeros(len(t))
    b = np.zeros(len(t))
    a_prev = a.copy()
    b_prev = a.copy()

    cost = [np.inf]
    tk = 0
    for k in range(maxit):
        t_next = (1 + np.sqrt(1+4*tk**2)) / 2
        s = (tk-1)/(t_next)
        v_a = a + s * (a-a_prev)
        v_b = b + s * (b-b_prev)

        a_prev = a.copy()
        b_prev = b.copy()
        a = np.clip(v_a - lr * (K@(v_a - v_b) + eps - t), 0, 1)
        b = np.clip(v_b - lr * (K@(v_b - v_a) + eps + t), 0, 1)

        cost.append((-1/2 * (a-b) @ K @ (a-b) - eps*np.sum(a+b) + np.sum(t*(a-b))))
        tk = t_next
        if np.abs(cost[-2] - cost[-1]) < tol:
            break

    return ((a-b) @ K, (a-b) @ K_test), cost


#%% 1.1 Sampling
phi, t = sample_swiss_roll(n=100, plot="1_1_dataset.png")
phi_test, t_test = sample_swiss_roll(n=100)

#%% 1.1+1.2
# GD
(y_train, y_test), cost_GD = GD(phi, t, phi_test, t_test)
plot_precision(y_test, t_test, "Gradient Descent", "1_1_precision_GD.png")

# AGD
(y_train, y_test), cost_AGD = AGD(phi, t, phi_test, t_test)
plot_precision(y_test, t_test, "Accelerated Gradient Descent", "1_1_precision_AGD.png")
#%%
# Comparison GD/AGD
plt.figure(figsize=(7,5))
plt.plot(cost_GD, label="GD")
plt.plot(cost_AGD, label="AGD")
plt.xlabel(r"Iterations")
plt.ylabel(r"Cost")
plt.yscale("symlog")
plt.legend()
plt.savefig("1_2_cost.png", dpi=400)


#%% 2.1+2.2
# PG

(y_train, y_test), cost_PG = PG(phi, t, phi_test, t_test, maxit=10000)
plot_precision(y_test, t_test, "Projected Gradient Descent", "2_1_precision_PG.png")

# FISTA
(y_train, y_test), cost_FISTA = FISTA(phi, t, phi_test, t_test, maxit=10000, tol=1e-15)
plot_precision(y_test, t_test, "FISTA", "2_1_precision_FISTA.png")

#%%
# Comparison PG/FISTA
plt.figure(figsize=(7,5))
plt.plot(cost_PG[:10000], label="PG")
plt.plot(cost_FISTA[:10000], label="FISTA")
plt.xlabel(r"Iterations")
plt.ylabel(r"Cost")
plt.yscale("symlog")
plt.legend()
plt.savefig("2_2_cost_short.png", dpi=400)

plt.figure(figsize=(7,5))
plt.plot(cost_PG, label="PG")
plt.plot(cost_FISTA, label="FISTA")
plt.xlabel(r"Iterations")
plt.ylabel(r"Cost")
plt.yscale("symlog")
plt.legend()
plt.savefig("2_2_cost.png", dpi=400)



