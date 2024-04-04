import numpy as np
import matplotlib.pyplot as plt


def lomb_scargle(t, y, frequency):
    y = y - np.mean(y)
    cos_term = np.cos(2 * np.pi * frequency * t)
    sin_term = np.sin(2 * np.pi * frequency * t)
    Syy = np.sum(y**2)
    Sy_cos = np.sum(y * cos_term)
    Sy_sin = np.sum(y * sin_term)
    S_cos_cos = np.sum(cos_term**2)
    S_sin_sin = np.sum(sin_term**2)
    S_cos_sin = np.sum(cos_term * sin_term)
    power = (Sy_cos**2 / S_cos_cos + Sy_sin**2 / S_sin_sin) / Syy
    
    return power



cat_type = 'EB'

N = 1000
t = np.linspace(0, 10, N)
P = 3.
Amplitude = 5.




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


frequency = np.fft.fftfreq(N, d=(t[1] - t[0]))[1:N//2]  # Positive frequencies
frequency = 1./np.linspace(0.001, 10, 1000)
power = np.array([lomb_scargle(t, y, f) for f in frequency])

peak_frequency_index = np.argmax(power)
peak_frequency = frequency[peak_frequency_index]

y_mean = np.mean(y)
y_detrended = y - y_mean

cos_term = np.cos(2 * np.pi * peak_frequency * t)
sin_term = np.sin(2 * np.pi * peak_frequency * t)
A = 2 * np.sum(y_detrended * cos_term) / len(t)
B = 2 * np.sum(y_detrended * sin_term) / len(t)

amplitude = np.sqrt(A**2 + B**2)
phase_shift = np.arctan2(B, A)

y_fit = y_mean + amplitude * np.sin(2 * np.pi * peak_frequency * t + phase_shift)

phase_folded_time = np.mod(t, 1/peak_frequency) * peak_frequency
sorted_indices = np.argsort(phase_folded_time)
phase_folded_y = y[sorted_indices]
phase_folded_time_sorted = phase_folded_time[sorted_indices]



# Plotting
plt.figure(figsize=(12, 9))

# Original Time Series with Lomb-Scargle Fit
plt.subplot(3, 1, 1)
plt.scatter(t, y, label='Original Time Series', marker='x')
plt.plot(t, y_fit, label=f'Lomb-Scargle Fit\nTrue Period: {P}, Guessed Period: {1/peak_frequency:.2f}', color='red')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.xlim(t[0], t[-1])
plt.legend()
plt.gca().invert_yaxis()  # Flip y-axis

# Lomb-Scargle Periodogram in Log Space
plt.subplot(3, 1, 2)
plt.semilogx(frequency, power, label='Lomb-Scargle Periodogram')  # Plot frequency in log space
plt.axvline(1/P, color='r', linestyle='--', label='True Period')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.xlim(frequency[0], frequency[-1])
plt.legend()

# Phase Folded Signal
plt.subplot(3, 1, 3)
plt.plot(phase_folded_time_sorted, phase_folded_y, 'k.', alpha=0.5)
plt.plot(phase_folded_time_sorted+1, phase_folded_y, 'k.', alpha=0.5)
plt.xlabel(f'Phase (Period = {1/peak_frequency:.2f})')
plt.ylabel('Signal')
#plt.xlim(0, 1/peak_frequency * peak_frequency)
plt.gca().invert_yaxis()  # Flip y-axis

plt.tight_layout()
plt.show()


