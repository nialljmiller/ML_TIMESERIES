import random
import numpy as np
from math import floor, log10

def normalize(x):
    """
    Normalize the given list of numbers.

    Args:
        x: List of numbers to be normalized.

    Returns:
        List: Normalized list of numbers.
    """
    min_val = min(x)
    max_val = max(x)
    x_normalized = [(xi - min_val) / (max_val - min_val) for xi in x]
    return x_normalized


def round_sig(x, sig=3):
    """
    Round the given number to the specified significant figures.

    Args:
        x: Number to be rounded.
        sig (int): Number of significant figures.

    Returns:
        float: Rounded number.
    """
    return round(x, sig - int(floor(log10(np.abs(x)))) - 1)


def average_separation(x):
    """
    Calculate the average separation between adjacent elements in the given list.

    Args:
        x: List of numbers.

    Returns:
        List: Differences between adjacent numbers.
    """
    x_sorted = sorted(x)
    diff_list = [x_sorted[i+1] - xi for i, xi in enumerate(x_sorted[:-1])]
    return diff_list


def source_lc_gen(mag, magerr, time, N, median_mag, time_range):
    """
    Generate a synthetic light curve based on the provided parameters.

    Args:
        mag: Magnitude values (optional).
        magerr: Magnitude error values (optional).
        time: Time values (optional).
        N: Number of data points (optional).
        median_mag: Median magnitude value (optional).
        time_range: Range of time values (optional).

    Returns:
        Tuple: mag, magerr, time arrays representing the synthetic light curve.
    """
    def seed_gen(N, median_mag):
        N = int(N)
        eemag = 0.1 * np.random.rand(N)
        if median_mag is None:
            median_mag = random.uniform(11, 16)
        mmag = np.random.normal(loc=median_mag, scale=eemag)
        ttime = np.array(sorted(list(np.random.uniform(time_range[0], time_range[1], N))))
        return mmag, eemag, ttime

    if not hasattr(mag, '__len__'):
        if mag is None and time is None and magerr is None:
            if N is None:
                N = 0
                while N < 40:
                    N = np.random.normal(150, 100)
            mag, magerr, time = seed_gen(N, median_mag)
    return mag, magerr, time




def synthesize(mag = None, magerr = None, time = None, N = None, period = None, amplitude = None, median_mag = None, other_pert = 1, scatter_flag = 1, contamination_flag = 1, err_rescale = 1, cat_type = None, time_range = [0.5,3000]):

    #rather shit model of VVV mag with error
    avg_mag_bin = [10.040372, 10.106127, 10.173317, 10.234861, 10.308775, 10.377685, 10.451731500000001, 10.5285015, 10.598759, 10.669487, 10.73957, 10.810725000000001, 10.881359, 10.95342, 11.027721499999998, 11.0958905, 11.1650545, 11.238672, 11.309748, 11.382094, 11.457752500000002, 11.534345, 11.604072500000001, 11.6745515, 11.7512355, 11.81738075, 11.894604, 11.963408, 12.036707, 12.109016, 12.181256, 12.252554, 12.324789, 12.3981645, 12.470251, 12.542015, 12.614317, 12.6858965, 12.756556750000001, 12.832128, 12.902282499999998, 12.9748005, 13.046791, 13.118004, 13.1920285, 13.2634495, 13.3350035, 13.407931, 13.481561, 13.552341, 13.622314, 13.6955625, 13.76782, 13.838537500000001, 13.9120025, 13.984888, 14.056742499999999, 14.128523, 14.199853000000001, 14.2713355, 14.346825, 14.417756, 14.489715, 14.5600635, 14.634105, 14.706612, 14.777646, 14.8505705, 14.9211275, 14.9943975, 15.06782825, 15.139413000000001, 15.209597, 15.2831265, 15.35553, 15.4276005, 15.499678, 15.5723, 15.6427, 15.716056, 15.787646500000001, 15.859731, 15.932091, 16.004604, 16.076054, 16.147951, 16.219805, 16.2915725, 16.3641245, 16.434887, 16.506264, 16.578470000000003, 16.6485325, 16.720625, 16.7927785, 16.8620885, 16.929991, 16.9983635, 17.0589535]
    avg_error_bin = [0.026844555638067216, 0.026197345927357674, 0.02640560921281576, 0.025627582801207, 0.026792809550292035, 0.02788254529449955, 0.029116684478593023, 0.02997786006315046, 0.031448022462427616, 0.033220841419287106, 0.035331808030605316, 0.03665425255894661, 0.036811774320255064, 0.03687429800629616, 0.03636088700046998, 0.0368590605275446, 0.03852042395514339, 0.04032818170460159, 0.037341348367940425, 0.01936853154962071, 0.01614197300046612, 0.014526145900680045, 0.01443700303573087, 0.0141100195715209, 0.013984198553913614, 0.014125574888887344, 0.01398088278231592, 0.013632951328235379, 0.013461585534912624, 0.01357920092549383, 0.014281293006155922, 0.013696851548206213, 0.013744714233774142, 0.014178681527743112, 0.013750323715893138, 0.014050657219297961, 0.01393119691627764, 0.013958681096272425, 0.013856950635459341, 0.014122184043215741, 0.014078056517752773, 0.014803738425579631, 0.014827648530303213, 0.015508792676079693, 0.015446021243079339, 0.015523268317554427, 0.015896457705680562, 0.015407678991816682, 0.01703616628799508, 0.01710015062572744, 0.017012397364019077, 0.017571954631031787, 0.018254719418201348, 0.019051763324180654, 0.019537952002466885, 0.01970697469443024, 0.021865301816455244, 0.022140924902887, 0.023460359654389876, 0.024154016896792436, 0.025848377700503862, 0.028487512041906613, 0.03197414027451289, 0.03369389764697338, 0.038291906975039994, 0.042842694, 0.045709997, 0.051625828964748696, 0.058396097070579536, 0.06763478368520737, 0.07339992728626596, 0.08075345504271889, 0.08418310757165579, 0.08857984267734197, 0.09266220667471713, 0.09619357366851675, 0.09949459806729367, 0.10244049153184318, 0.10534778376864008, 0.10852749488299651, 0.11068468101667518, 0.11390939184986243, 0.11675941442972973, 0.12006474594509783, 0.12344997832387634, 0.12693700379800404, 0.12878291722476976, 0.13236667692454124, 0.13580439516004444, 0.13670937229127034, 0.1394514041085406, 0.14087193189112926, 0.14294291225945582, 0.14091840386390686, 0.14045158783765338, 0.13852295118269414, 0.14258005947744384, 0.1501466555443668, 0.1588832772837242]
    avg_error_scatter_bin = [0.0023249415513398397, 0.0032464219803419454, 0.005014493349579674, 0.0058966845203024055, 0.005552451638542173, 0.005470780196631637, 0.006122925972573526, 0.0070763489483612, 0.007242603339618587, 0.007889815898890528, 0.00907805017351719, 0.009947439189154503, 0.010544483321065142, 0.010685587330814884, 0.011940412675263289, 0.013341752070576139, 0.015026615856788437, 0.018130276897773214, 0.020146587344894776, 0.020924469075681556, 0.018690327692789954, 0.010665227028891945, 0.005445444603574777, 0.004964782308821221, 0.004905396562693245, 0.006410713089037467, 0.005842500052720586, 0.007319586003106991, 0.007276974625352824, 0.007139729801268189, 0.007083371675306288, 0.006958359596767747, 0.00790339918368763, 0.008370351277849845, 0.0081361966337113, 0.01010693722359795, 0.009788566475071092, 0.009699964464619402, 0.010252060523274773, 0.011036837756567315, 0.010947011238695195, 0.013349340007865034, 0.012410327386829284, 0.012462621223280769, 0.013365271567050842, 0.012678883809515555, 0.014371990450106942, 0.014672523148968475, 0.015370712091303137, 0.01579883573534652, 0.016848800532741095, 0.01588645015141995, 0.01753621375505945, 0.016873456598130426, 0.01810190553603428, 0.018373409108896652, 0.018910351821588817, 0.019581111823412906, 0.020881150743108475, 0.021838265272177335, 0.022756156657350684, 0.02468432914620037, 0.025115615159347873, 0.026330841113654718, 0.027416966621945378, 0.028700866780444072, 0.030224066781072256, 0.030773710453556695, 0.03158162786647261, 0.032232609710229884, 0.03187949281352205, 0.031743990592337804, 0.031687451087010016, 0.032016074906030235, 0.03165036409770324, 0.032090026560939525, 0.03164141387876358, 0.031977030060576304, 0.03188550860336392, 0.03288817072112862, 0.033253812544322275, 0.033533595472442924, 0.03315656190416287, 0.03412086881631629, 0.03430001404230631, 0.035381259047728326, 0.034799534073335, 0.03500746772563861, 0.03567193611390633, 0.03647414742229041, 0.03612555668303998, 0.036344353740833515, 0.0374580029634055, 0.03601953195312909, 0.03564415085237351, 0.034446657020004365, 0.03453775530002765, 0.02711997913167808, 0.01726980227984657]

    if hasattr(mag, '__len__') == False:
        mag, magerr, time = source_lc_gen(mag, magerr, time, N, median_mag, time_range)

    # Set up an example seed light curve
    time_range = max(np.array(time)) - min(np.array(time))

    n_pts = int(len(mag))

    # Either use real or generated mag and mag error as seed
    # Use generated if seed is causing problems (likely source LC is variable itself)
    seed_t = time

    mag25, mag75 = np.percentile(mag, [25,75])
    seed_m = random.uniform(mag25, mag75)

    magerr25, magerr75 = np.percentile(magerr, [25,75])
    seed_em = random.uniform(magerr25, magerr75)

    cat_types = ['Ceph', 'EB', 'RR', 'YSO', 'CV']

    if cat_type == None or cat_type not in cat_types:
        cat_type = np.random.choice(cat_types)  # If no type, randomly pick any

    if isinstance(cat_type, str) == False:
        cat_type = np.random.choice(cat_type)  # If cat type has multiple types, randomly pick one

    if period == None:
        # If no period, try to pick a vaguely physical period. NB these came to me in a dream
        if cat_type == 'Ceph':
            period = random.uniform(1, 100)
        if cat_type == 'EB':
            period = random.uniform(1, 20)
        if cat_type == 'RR':
            period = random.uniform(0.1, 3)
        if cat_type == 'YSO':
            period = random.uniform(0.1, 14)
        if cat_type == 'CV':
            period = random.uniform(3, 30)

    if amplitude == None:
        # If no amplitude, pick one that's above the noise
        avg_error = np.median(seed_em)
        amplitude = random.uniform(3 * avg_error, 10 * avg_error)


def lc_model(t, P, Amplitude, cat_type):
    """
    Generate a synthetic light curve based on the provided parameters.

    Parameters:
        t (list): List of time for the given light curve.
        P (float): Period of the signal being constructed.
        Amplitude (float): Amplitude of the signal being constructed.
        cat_type (string): Type of periodic signal to use.

    Returns:
        Tuple: m (list) containing y data for the periodic signal normalized between -0.5 and 0.5,
               cat_type (string) indicating the type of periodic signal used.
    """
    def normalize(m):
        min_val = min(m)
        max_val = max(m)
        return [(xi - min_val) / (max_val - min_val) - 0.5 for xi in m]

    def perturber(sub_class, t, P, Amplitude):
        if sub_class == 0:
            return np.sin((2*np.pi*t)/P) * Amplitude
        elif sub_class == 1:
            return np.sin((2*np.pi*t)/P) * Amplitude + np.sin((2*np.pi*t)/np.random.uniform(0.01*P,0.1*P)) * np.random.uniform(0.005*Amplitude,0.05*Amplitude)
        else:
            return []

    if cat_type == 'EB':
        sub_class = np.random.choice([0,1,2])
        if sub_class == 0:
            cat_type = 'EB_0'
            A2 = random.uniform(0.2*Amplitude, 0.9*Amplitude)
            A1 = Amplitude
        elif sub_class == 1:
            cat_type = 'EB_1'
            A2 = random.uniform(2.5*Amplitude, 3.5*Amplitude)
            A1 = Amplitude
        elif sub_class == 2:
            cat_type = 'EB_2'
            A1 = random.uniform(0.01, Amplitude - 0.01)
            A2 = Amplitude - A1
        m = 1 - (A1*np.sin((2*np.pi*t)/P)**2 + A2*np.sin((np.pi*t)/P)**2)

    elif cat_type == 'CV':
        A1 = random.uniform(0.2, 0.6) * Amplitude
        A2 = Amplitude - A1
        m = A1*np.sin((2*np.pi*t)/P)**2 + A2*np.sin((np.pi*t)/P)**2

    elif cat_type == 'Ceph':
        sub_class = np.random.choice([0,1,2,3])
        if sub_class == 0:
            cat_type = 'Ceph_0'
            m = ((0.5*np.sin((2*np.pi*t)/P) - 0.15*random.uniform(0.5,1.5)*np.sin((2*2*np.pi*t)/P) - 0.05*random.uniform(0.5,1.5)*np.sin((3*2*np.pi*t)/P)))*Amplitude
        elif sub_class == 1:
            cat_type = 'Ceph_1'
            m = ((0.5*np.sin((2*np.pi*t)/P) + 0.15*random.uniform(0.5,1.5)*np.sin((2*2*np.pi*t)/P) + 0.05*random.uniform(0.5,1.5)*np.sin((3*2*np.pi*t)/P)))*Amplitude
        elif sub_class == 2:
            cat_type = 'Ceph_2'
            m = ((0.5*np.sin((2*np.pi*t)/P) + 0.15*random.uniform(0.5,1.5)*np.sin((2*2*np.pi*t)/P) - 0.05*random.uniform(0.5,1.5)*np.sin((3*2*np.pi*t)/P)))*Amplitude
        elif sub_class == 3:
            cat_type = 'Ceph_3'
            m = ((0.5*np.sin((2*np.pi*t)/P) - 0.15*random.uniform(0.5,1.5)*np.sin((2*2*np.pi*t)/P) + 0.05*random.uniform(0.5,1.5)*np.sin((3*2*np.pi*t)/P)))*Amplitude

    elif cat_type == 'RR':
        sub_class = np.random.choice([0,1])
        if sub_class == 0:
            cat_type = 'RR_0'
            m = abs(np.sin((np.pi*t)/P)) * Amplitude
        elif sub_class == 1:
            cat_type = 'RR_1'
            m = abs(np.cos((np.pi*t)/P)) * Amplitude

    elif cat_type == 'YSO':
        sub_class = 0
        if sub_class == 0:
            cat_type = 'YSO_0'
            m = perturber(sub_class, t, P, Amplitude)
        elif sub_class == 1:
            cat_type = 'YSO_1'
            m = perturber(sub_class, t, P, Amplitude)

    m = normalize(m)
    return m, cat_type

def simulate_light_curve(seed_t, period, amplitude, cat_type, seed_m, other_pert=1, err_rescale=1, scatter_flag=1, contamination_flag=1):
    """
    Simulate a light curve based on the provided parameters.

    Parameters:
        seed_t (list): List of time values for the seed light curve.
        period (float): Period of the signal being constructed.
        amplitude (float): Amplitude of the signal being constructed.
        cat_type (string): Type of periodic signal to use.
        seed_m (list): Seed magnitudes for the light curve.
        other_pert (int): Flag indicating whether to include additional perturbations (default=1).
        err_rescale (int): Flag indicating whether to rescale the errors (default=1).
        scatter_flag (int): Flag indicating whether to add scatter to the simulated magnitudes (default=1).
        contamination_flag (int): Flag indicating whether to add contamination to the simulated magnitudes (default=1).

    Returns:
        Tuple: sim_mags (list) containing the simulated magnitudes,
               seed_em (list) containing the seed magnitude errors,
               seed_t (list) containing the seed time values,
               cat_type (string) indicating the type of periodic signal used,
               period (float) indicating the period of the signal.
    """
    avg_mag_bin = []  # Define avg_mag_bin

    # Create new_err by iterating over sim_mags
    new_err = [0.1 for _ in seed_m]

    seed_em = new_err

    if err_rescale == 1:
        mag_err_scaler = ((seed_m - np.min(seed_m)) / ((np.max(seed_m) - np.min(seed_m)) * 1000)) + 0.9995
        seed_em = seed_em * mag_err_scaler

    if scatter_flag == 1:
        try:
            old_mags = seed_m
            seed_m = [np.random.normal(float(mag), float(seed_em[i])*2) for i, mag in enumerate(old_mags)]
        except Exception as e:
            print(e)
            print(old_mags)
            print(seed_m)
            print(seed_em)
            print(mag_err_scaler)

    if contamination_flag == 1:
        contaminations = int(random.uniform(1, 4))
        for _ in range(contaminations):
            cont_id = int(random.uniform(0, len(seed_m)))
            seed_m[cont_id] = float(seed_m[cont_id] + random.uniform(-2, 2) * amplitude)
            seed_em[cont_id] = float(seed_em[cont_id] * random.uniform(2, 5))

    sim_mags, _ = lc_model(seed_t, period, amplitude, cat_type)

    sim_mags = sim_mags + seed_m

    if other_pert == 1:
        sim_mags = sim_mags + (np.sin((2*np.pi*seed_t+random.uniform(0,128*np.pi))/period) * amplitude * random.uniform(0.0005, 0.01))

    return sim_mags, seed_em, seed_t, cat_type, period







