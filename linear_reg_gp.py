import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import autograd.numpy as np
from scipy.optimize import minimize
import celerite
from celerite import terms
import corner


def generate_sinusoidal_data(num_points, amplitude, frequency, noise_std_dev):
    x = np.linspace(0, 10, num_points)
    noise = np.random.normal(0, noise_std_dev, num_points)
    y = sinus_eq(amplitude, frequency, x) + noise
    return x, y


def sinus_eq(amplitdue, frequency, x):
    phase_shift = np.random.uniform(0, 2 * np.pi)  # Random phase shift
    return amplitude * np.sin(frequency * x * np.pi + phase_shift)


#make x y data
def generate_linear_data(num_points, gradient, intercept, noise_std_dev):
    rand_nums = np.random.rand(num_points)
    x = 10 * rand_nums
    noise = np.random.normal(0, noise_std_dev, num_points)
    y = linear_eq(x, gradient, intercept) + noise
    return x, y

def linear_eq(X, m, c):
    return m*X + c


# Example usage
num_points = 100  # Number of data points to generate
gradient = 1 # User-defined gradient
intercept = 0  # User-defined intercept
noise_std_dev = 0.1  # Standard deviation of the noise
frequency = 1/2
amplitude = 2


X, Ylin = generate_linear_data(num_points, gradient, intercept, noise_std_dev)
X, Y = generate_sinusoidal_data(num_points, amplitude, frequency, noise_std_dev)

mag = Y + linear_eq(X, 1,0)
magerr = Y * (0.1 * np.random.normal(0,1))
time = X

class CustomTerm(terms.Term):
	parameter_names = ("log_b", "log_c", "log_l", "log_P")

	def get_real_coefficients(self, params):
		log_b, log_c, log_l, log_P = params
		c = np.exp(log_c)
		return (
			np.exp(log_c) * (1.0 + c) / (2.0 + c), np.exp(log_l),
		)

	def get_complex_coefficients(self, params):
		log_b, log_c, log_l, log_P = params
		c = np.exp(log_c)
		return (
			np.exp(log_b) / (2.0 + c), 0.0,	np.exp(log_l), 2*np.pi*np.exp(-log_P),
		)


def nll(p, y, gp):
    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(y)
    return -ll if np.isfinite(ll) else 1e25


def grad_nll(p, y, gp):
    gp.set_parameter_vector(p)
    gll = gp.grad_log_likelihood(y)[1]
    return -gll


def lnprior(p):
    return gp.log_prior()


def lnprob(p, x, y):
    lp = lnprior(p)
    return lp + lnlike(p, x, y) if np.isfinite(lp) else -np.inf


def lnlike(p, x, y):
    ln_a, ln_b, ln_c, ln_p = p
    p0 = np.array([ln_a, ln_b, ln_c, ln_p])
    gp.set_parameter_vector(p0)
    try:
        ll = gp.log_likelihood(y)
    except:
        ll = 1e25
    return ll if np.isfinite(ll) else 1e25


def plot_psd(gp):
    plt.loglog(GP_periods, gp.kernel.get_psd(GP_omega), ":k", label="model")
    plt.xlim(GP_periods.min(), GP_periods.max())
    plt.legend()
    plt.xlabel("Period [day]")
    plt.ylabel("Power [day ppt$^2$]")


# Setup the data
sort = np.argsort(time)
y = (np.array(mag) - min(mag)) / (max(mag) - min(mag))
yerr = np.array(magerr)[sort]
t = np.array(time)[sort]
t_full = np.linspace(np.min(time), np.max(time), 1000)  # For plotting

# Define priors
log_b = 0.0
log_c = 0.0
log_l = 0.0
log_P = np.log(10)
log_slope = 0.0

# Define bounds
period_sigma = 2.0
freqs = np.linspace(0.01,10,100000)
GP_periods = 1 / freqs
GP_omega = (2 * np.pi) / GP_periods
bnds = ((np.log(0.01), np.log(2.0)),
        (np.log(0.1), np.log(100.0)),
        (np.log(0.0001), np.log(10.0)),
        (np.log(1.0 / 10), np.log(1.0 / 0.001)))


# Setup the GP class
kernel = CustomTerm(log_b, log_c, log_l, log_P)
#global gp
gp = celerite.GP(kernel, mean=0.0)

gp.compute(t,yerr)

# Define initial params for optimization
p0 = gp.get_parameter_vector()

# Run optimization
results = minimize(nll, p0, method='L-BFGS-B', jac=grad_nll, args=(y, gp), bounds=bnds)

# Set optimized parameters to the GP
gp.set_parameter_vector(results.x)


# Get the samples array
samples = np.vstack((np.exp(results.x[0]), np.exp(results.x[1]), -np.exp(results.x[2]), np.exp(results.x[3]))).T


# Get the median values
found_b_post = np.median(np.exp(results.x[0]))
found_c_post = np.median(np.exp(results.x[1]))
found_l_post = -1.0 * np.median(np.exp(results.x[2]))
found_period_post = np.median(np.exp(results.x[3]))

error_b_post = np.std(np.exp(results.x[0]))
error_c_post = np.std(np.exp(results.x[1]))
error_l_post = np.std(np.exp(results.x[2]))
error_period_post = np.std(np.exp(results.x[3]))

# Print the results
print(f"b: {found_b_post} +/- {error_b_post}")
print(f"c: {found_c_post} +/- {error_c_post}")
print(f"l: {found_l_post} +/- {error_l_post}")
print(f"Period: {found_period_post} +/- {error_period_post}")








# Assuming you have the optimized parameters stored in 'results.x'
optimized_params = results.x

# Setup the GP class with optimized parameters
kernel = CustomTerm(*optimized_params[:4])
gp = celerite.GP(kernel, mean=0.0)
gp.compute(t, yerr)





# Generate samples from the posterior using the optimized parameters
samples = gp.sample_conditional(y, t_full, size=1000)




# Plot light curve with GP prediction
plt.errorbar(t, y, yerr=yerr, fmt=".k", label="Data")
plt.plot(t_full, np.median(samples, axis=0), label="GP Prediction")
plt.fill_between(t_full, np.percentile(samples, 16, axis=0), np.percentile(samples, 84, axis=0), alpha=0.3)
plt.xlabel("Time")
plt.ylabel("Normalized Magnitude")
plt.legend()
plt.show()
plt.clf()

# Get the optimized parameter values
found_b_post = np.exp(results.x[0])
found_c_post = np.exp(results.x[1])
found_l_post = -1.0 * np.exp(results.x[2])
found_period_post = np.exp(results.x[3])

# Reshape the samples array
reshaped_samples = np.exp(samples).reshape((-1, 4))

# Create labels for the corner plot
labels = ["b", "c", "l", "Period"]

# Create the corner plot
figure = corner.corner(reshaped_samples, labels=labels, truths=[found_b_post, found_c_post, found_l_post, found_period_post])

# Display the plot
plt.show()
