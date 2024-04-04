import numpy as np
import matplotlib.pyplot as plt

def pdm(t, y, periods, phase_bins=10, mag_bins=10):
    results = []
    min_mag, max_mag = np.min(y), np.max(y)
    mag_bin_edges = np.linspace(min_mag, max_mag, mag_bins + 1)
    
    for period in periods:
        phases = np.mod(t, period) / period
        indices = np.argsort(phases)
        phase_sorted = phases[indices]
        y_sorted = y[indices]
        
        phase_bin_edges = np.linspace(0, 1, phase_bins + 1)
        overall_variance = np.var(y_sorted)
        if overall_variance == 0:
            results.append(np.nan)
            continue
        
        total_binned_variance = 0
        for i in range(phase_bins):
            phase_in_bin = (phase_sorted >= phase_bin_edges[i]) & (phase_sorted < phase_bin_edges[i+1])
            for j in range(mag_bins):
                mag_in_bin = (y_sorted >= mag_bin_edges[j]) & (y_sorted < mag_bin_edges[j+1])
                bin_indices = phase_in_bin & mag_in_bin
                
                if np.sum(bin_indices) > 1:  # Ensure there's more than one point in the bin
                    bin_variance = np.var(y_sorted[bin_indices])
                    total_binned_variance += bin_variance
        
        ratio = total_binned_variance / (overall_variance * phase_bins * mag_bins)
        results.append(ratio)
    
    return periods, np.array(results)


cat_type = 'RR'

N = 1000
t = np.linspace(0, 10, N)
P = 3.
Amplitude = 5.
phase_bins = 10
mag_bins = 10


if cat_type == 'EB':
    sub_class = np.random.choice([0,1,2])
    if sub_class == 0:
        cat_type = 'EB_0'
        A2 = np.random.uniform(0.2*Amplitude, 0.9*Amplitude)
        A1 = Amplitude
    elif sub_class == 1:
        cat_type = 'EB_1'
        A2 = np.random.uniform(2.5*Amplitude, 3.5*Amplitude)
        A1 = Amplitude
    elif sub_class == 2:
        cat_type = 'EB_2'
        A1 = np.random.uniform(0.01, Amplitude - 0.01)
        A2 = Amplitude - A1
    m = 1 - (A1*np.sin((2*np.pi*t)/P)**2 + A2*np.sin((np.pi*t)/P)**2)

elif cat_type == 'CV':
    A1 = np.random.uniform(0.2, 0.6) * Amplitude
    A2 = Amplitude - A1
    m = A1*np.sin((2*np.pi*t)/P)**2 + A2*np.sin((np.pi*t)/P)**2

elif cat_type == 'Ceph':
    sub_class = np.random.choice([0,1,2,3])
    if sub_class == 0:
        cat_type = 'Ceph_0'
        m = ((0.5*np.sin((2*np.pi*t)/P) - 0.15*np.random.uniform(0.5,1.5)*np.sin((2*2*np.pi*t)/P) - 0.05*np.random.uniform(0.5,1.5)*np.sin((3*2*np.pi*t)/P)))*Amplitude
    elif sub_class == 1:
        cat_type = 'Ceph_1'
        m = ((0.5*np.sin((2*np.pi*t)/P) + 0.15*np.random.uniform(0.5,1.5)*np.sin((2*2*np.pi*t)/P) + 0.05*np.random.uniform(0.5,1.5)*np.sin((3*2*np.pi*t)/P)))*Amplitude
    elif sub_class == 2:
        cat_type = 'Ceph_2'
        m = ((0.5*np.sin((2*np.pi*t)/P) + 0.15*np.random.uniform(0.5,1.5)*np.sin((2*2*np.pi*t)/P) - 0.05*np.random.uniform(0.5,1.5)*np.sin((3*2*np.pi*t)/P)))*Amplitude
    elif sub_class == 3:
        cat_type = 'Ceph_3'
        m = ((0.5*np.sin((2*np.pi*t)/P) - 0.15*np.random.uniform(0.5,1.5)*np.sin((2*2*np.pi*t)/P) + 0.05*np.random.uniform(0.5,1.5)*np.sin((3*2*np.pi*t)/P)))*Amplitude

elif cat_type == 'RR':
    sub_class = np.random.choice([0,1])
    if sub_class == 0:
        cat_type = 'RR_0'
        m = abs(np.sin((np.pi*t)/P)) * Amplitude
    elif sub_class == 1:
        cat_type = 'RR_1'
        m = abs(np.cos((np.pi*t)/P)) * Amplitude

#normalise and add noise
y = (np.asarray([(xi - min(m)) / (max(m) - min(m)) for xi in m]) * Amplitude) + np.random.normal(0, 0.1, N)  # Signal + noise


periods = np.linspace(0.1, 10, 1000)  # Trial periods
periods, pdm_power = pdm(t, y, periods, phase_bins=phase_bins, mag_bins=mag_bins)

min_pdm_index = np.argmin(pdm_power)
best_period = periods[min_pdm_index]

phase_folded_time = np.mod(t, best_period) / best_period
sorted_indices = np.argsort(phase_folded_time)
phase_folded_y = y[sorted_indices]
phase_folded_time_sorted = phase_folded_time[sorted_indices]
phase_folded_y_sorted = y[sorted_indices]



plt.figure(figsize=(12, 18))  # Adjust the figure size to accommodate all subplots

plt.subplot(3, 1, 1)  # First subplot
plt.plot(t, y, 'k.', alpha=0.5, label='Original Time Series')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.gca().invert_yaxis()

plt.subplot(3, 1, 2)  # Second subplot
plt.plot(periods, pdm_power, label='PDM Periodogram')
plt.axvline(best_period, color='r', linestyle='--', label=f'Best Period: {best_period:.2f}')
plt.axvline(P, color='g', linestyle='--', label=f'True Period: {P:.2f}')
plt.xlabel('Period')
plt.ylabel('Theta Statistic')
plt.legend()


ax = plt.subplot(3, 1, 3)
bin_phase_edges = np.linspace(0, 1, phase_bins + 1)
min_mag, max_mag = np.min(phase_folded_y_sorted), np.max(phase_folded_y_sorted)
bin_mag_edges = np.linspace(min_mag, max_mag, mag_bins + 1)

bin_variance = np.zeros((phase_bins, mag_bins))

for i in range(phase_bins):
    for j in range(mag_bins):
        in_bin = (phase_folded_time_sorted >= bin_phase_edges[i]) & (phase_folded_time_sorted < bin_phase_edges[i+1]) & (phase_folded_y_sorted >= bin_mag_edges[j]) & (phase_folded_y_sorted < bin_mag_edges[j+1])
        if np.any(in_bin):
            bin_variance[i, j] = np.var(phase_folded_y_sorted[in_bin])

norm = plt.Normalize(vmin=bin_variance.min(), vmax=bin_variance.max())
cmap = plt.get_cmap('viridis')

for i in range(phase_bins):
    for j in range(mag_bins):
        rect = plt.Rectangle((bin_phase_edges[i], bin_mag_edges[j]), width=bin_phase_edges[i+1]-bin_phase_edges[i], height=bin_mag_edges[j+1]-bin_mag_edges[j], facecolor=cmap(norm(bin_variance[i, j])))
        ax.add_patch(rect)

plt.plot(phase_folded_time_sorted, phase_folded_y_sorted, 'wx', alpha=0.5, label='Original Time Series')
plt.plot(phase_folded_time_sorted, phase_folded_y_sorted, 'k.', alpha=0.5, label='Original Time Series')
plt.xlim(0,1)
plt.ylim(min_mag, max_mag)
ax.set_xlabel('Phase')
ax.set_ylabel('Signal')
ax.invert_yaxis()

plt.tight_layout()
plt.show()

