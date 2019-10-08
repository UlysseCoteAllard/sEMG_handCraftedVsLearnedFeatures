from scipy import stats
import numpy as np
import math
import matplotlib.pyplot as plt

# Util function cmp (needed as we are in Python 3)
def cmp(a, b):
    return bool(a > b) - bool(a < b)

def extract_features(vector):
    features = []
    for c in range(len(vector)):

        #plt.plot(np.asarray(vector[c]))
        #plt.show()
    #    features.append(getApEn(vector[c]))
        features.extend(getAR(vector[c], 4))
        features.extend(getCC(vector[c], 4))
        features.append(getDASDV(vector[c]))
    #    features.append(getFuzzyEn(vector))
        features.extend(getHIST(vector, 3))

    #    features.append(getHOMYOP(vector))
    #    features.append(getHOSSC(vector))
    #    features.append(getHOWAMP(vector))
    #    features.append(getHOZC(vector))

        features.append(getIEMG(vector[c]))
        features.extend(getIQR(vector[c]))
        features.append(getLD(vector[c]))
    #    features.append(getLS(vector)) # LOOKUP LMOM function to implement this <--
        features.extend(getMAVFD(vector[c]))
        features.append(getMAVFDn(vector[c]))
        features.append(getMAV(vector[c]))
        features.append(getMAVSD(vector[c]))
        features.append(getMAVSDn(vector[c]))
        features.extend(getMAVSLP(vector[c],2))
        features.append(getMDF(vector[c],1000))
        features.append(getMMAV1(vector[c]))
        features.append(getMMAV2(vector[c]))
        features.append(getMNF(vector[c],1000))
        features.append(getMNP(vector[c]))
        features.append(getMPK(vector[c]))
        features.append(getMSR(vector[c]))
        features.append(getMYOP(vector[c],1))
        features.append(getRANGE(vector[c]))
        features.append(getRMS(vector[c]))
        #features.append(getSampEn(vector[c]))
        features.append(getSKEW(vector[c]))
        features.append(getSM(vector[c],2,1000))
        features.append(getSSC(vector[c], 0.01))
        features.append(getSSI(vector[c]))
        features.append(getSTD(vector[c]))
        features.append(getTM(vector[c]))
        features.append(getTTP(vector[c]))
        features.append(getVAR(vector[c]))
        features.append(getWAMP(vector[c],0.1))
        features.append(getWL(vector[c]))
        features.append(getZC(vector[c],0.1))
    return features

# New function
def getAR(vector, order = 4):
    # Using Levinson Durbin prediction algorithm, get autoregressive coefficients
    # Square signal
    vector = np.asarray(vector)
    R = [vector.dot(vector)]
    if R[0] == 0:
        return [1] + [0]*(order-2) + [-1]
    else:
        for i in range(1, order + 1):
            r = vector[i:].dot(vector[:-i])
            R.append(r)
        R = np.array(R)
        # step 2:
        AR = np.array([1, -R[1] / R[0]])
        E = R[0] + R[1] * AR[1]
        for k in range(1, order):
            if (E == 0):
                E = 10e-17
            alpha = - AR[:k + 1].dot(R[k + 1:0:-1]) / E
            AR = np.hstack([AR, 0])
            AR = AR + alpha * AR[::-1]
            E *= (1 - alpha ** 2)
        return AR

    return AR_coeff

# New function
def getCC(vector, order =4):
    AR =  getAR(vector,order)
    cc = np.zeros(order+1)
    cc[0] = -1*AR[0]# issue with this line
    if order > 2:
        for p in range(2,order+2):
            for l in range(1, p):
                cc[p-1] = cc[p-1]+(AR[p-1] * cc[p-2] * (1-(l/p)))

    return cc


#New function
def getDASDV(vector):
    vector = np.asarray(vector)
    return np.lib.scimath.sqrt(np.mean(np.diff(vector)))


# New function
def getHIST(vector, bins=3):
    hist,bin_edges = np.histogram(vector, bins)
    return hist.tolist()

# New function
def getIEMG(vector):
    vector = np.asarray(vector)
    return np.sum(np.abs(vector))

# New function
def getIQR(vector):
    vector = np.asarray(vector)
    vector.sort()
    return [vector[int(round(vector.shape[0]/4))], vector[int(round(vector.shape[0]*3/4))]]

# New function
def getLD(vector):
    vector = np.asarray(vector)
    return np.exp(np.mean(np.log(np.abs(vector)+1)))

# New function
def getMAVFD(vector):
    vector = np.diff(np.asarray(vector))
    total_sum = 0
    for i in range(len(vector)):
        total_sum += abs(vector[i])
    return (total_sum / vector.shape).tolist()

# New function
def getMAVFDn(vector):
    vector = np.asarray(vector)
    std = np.std(vector)
    return np.mean(np.abs(np.diff(vector)))/std

# New function
def getMAVSD(vector):
    vector = np.asarray(vector)
    return np.mean(np.abs(np.diff(np.diff(vector))))

# New function
def getMAVSDn(vector):
    vector = np.asarray(vector)
    std = np.std(vector)
    return np.mean(np.abs(np.diff(np.diff(vector))))/std

def getMAVSLP(vector, segment=2):
    vector = np.asarray(vector)
    m = int(round(vector.shape[0]/segment))
    mav = []
    mavslp = []
    for i in range(0,segment):
        mav.append(np.mean(np.abs(vector[i*m:(i+1)*m])))
    for i in range (0, segment-1):
        mavslp.append(mav[i+1]- mav[i])
    return mavslp


# Ulysse's function
def getMAV(vector):
    total_sum = 0
    for i in range(len(vector)):
        total_sum += abs(vector[i])
    return total_sum/len(vector)

# New function
def getMDF(vector,fs=1000):
    vector = np.asarray(vector)
    spec = np.fft.fft(vector)
    spec = spec[0:int(round(spec.shape[0]/2))]
    #f = np.fft.fftfreq(vector.shape[-1])
    POW = spec * np.conj(spec)
    totalPOW = np.sum(POW)
    medfreq = 0
    for i in range(0, vector.shape[0]):
        if np.sum(POW[0:i]) > 0.5 *totalPOW:
            medfreq = fs*i/vector.shape[0]
            break
    return medfreq

# Ulysse function
def getMMAV1(vector):
    vector_array = np.array(vector)
    total_sum = 0.0
    for i in range(0,len(vector_array)):
        if((i+1) < 0.25*len(vector_array) or (i+1) > 0.75*len(vector_array)):
            w = 0.5
        else:
            w = 1.0
        total_sum += abs(vector_array[i]*w)
    return total_sum/len(vector_array)

def getMMAV2(vector):
    total_sum = 0.0
    vector_array = np.array(vector)
    for i in range(0, len(vector_array)):
        if ((i + 1) < 0.25 * len(vector_array)):
            w = ((4.0 * (i + 1)) / len(vector_array))
        elif ((i + 1) > 0.75 * len(vector_array)):
            w = (4.0 * ((i + 1) - len(vector_array))) / len(vector_array)
        else:
            w = 1.0
        total_sum += abs(vector_array[i] * w)
    return total_sum / len(vector_array)

# New function
def getMNF(vector, fs=1000):
    vector = np.asarray(vector)
    spec = np.fft.fft(vector)
    f = np.fft.fftfreq(vector.shape[-1])*fs
    spec = spec[0:int(round(spec.shape[0]/2))]
    f = f[0:int(round(f.shape[0]/2))]
    POW = spec * np.conj(spec)

    return np.sum(POW*f)/sum(POW)

# New function
def getMNP(vector):
    vector = np.asarray(vector)
    spec = np.fft.fft(vector)
    spec = spec[0:int(round(spec.shape[0]/2))]
    POW = spec*np.conj(spec)
    return np.sum(POW)/POW.shape[0]


# New function
def getMPK(vector):
    vector = np.asarray(vector)
    return vector.max()

# New function
def getMSR(vector):
    vector = np.asarray(vector)
    return (np.abs(np.mean(np.lib.scimath.sqrt(vector))))

# New function
def getMYOP(vector, threshold=1):
    vector = np.asarray(vector)
    return np.sum(np.abs(vector) >= threshold)/float(vector.shape[0])

# New function
def getRANGE(vector,filsize=2):
    vector = np.asarray(vector)
    return vector.max()-vector.min()

# New function
def getRMS(vector):
    vector = np.asarray(vector)
    return np.sqrt(np.mean(np.square(vector)))

# Ulysse function
def getSampEn(vector,m=2, r_multiply_by_sigma=.2):
    vector = np.asarray(vector)
    r = r_multiply_by_sigma * np.std(vector)
    results = sampen.sampen2(data=vector.tolist(), mm=m, r=r, normalize=True)
    results_SampEN = []
    for x in np.array(results)[:, 1]:
        if x is not None:
            results_SampEN.append(x)
        else:
            results_SampEN.append(-100.)
    return list(results_SampEN)

# New function
def getSKEW(vector):
    vector = np.asarray(vector)
    return stats.skew(vector)

# New function
def getSM(vector, order=2, fs=1000):
    vector = np.asarray(vector)
    spec = np.fft.fft(vector)
    spec = spec[0:int(round(spec.shape[0]/2))]
    f = np.fft.fftfreq(vector.shape[-1]) * fs
    f = f[0:int(round(f.shape[0] / 2))]
    POW = spec*np.conj(spec)
    return np.sum(POW.dot(np.power(f,order)))

# Ulysse function
def getSSC(vector, threshold=0.1):
    vector = np.asarray(vector)
    slope_change = 0
    for i in range(1,len(vector)-1):
        get_x = (vector[i]-vector[i-1])*(vector[i]-vector[i+1])
        if(get_x >= threshold):
            slope_change += 1
    return slope_change

# New function
def getSSI(vector):
    vector = np.asarray(vector)
    return np.sum(np.square(vector))

# New function
def getSTD(vector):
    vector = np.asarray(vector)
    return np.std(vector)

def getTM(vector, order=3):
    vector = np.asarray(vector)
    return np.abs(np.mean(np.power(vector, order)))

# New function
def getTTP(vector):
    vector = np.asarray(vector)
    spec = np.fft.fft(vector)
    spec = spec[0:int(round(spec.shape[0]/2))]
    POW = spec*np.conj(spec)
    return np.sum(POW)

# New function
def getVAR(vector):
    vector = np.asarray(vector)
    return np.square(np.std(vector))

# Ulysse function
def getWAMP(vector, threshold=0.1):
    vector = np.asarray(vector)
    wamp_decision = 0
    for i in range(1, len(vector)):
        get_x = abs(vector[i] - vector[i - 1])
        if (get_x >= threshold):
            wamp_decision += 1
    return wamp_decision

# New function
def getWL(vector):
    vector = np.asarray(vector)
    return np.sum(np.abs(np.diff(vector)))

# Ulysse function
def getZC(vector,threshold=0.1):
    vector = np.asarray(vector)
    number_zero_crossing = 0
    current_sign = cmp(vector[0], 0)
    for i in range(0, len(vector)):
        if current_sign == -1:
            if current_sign != cmp(vector[i], threshold):  # We give a delta to consider that the zero was crossed
                current_sign = cmp(vector[i], 0)
                number_zero_crossing += 1
        else:
            if current_sign != cmp(vector[i], -threshold):
                current_sign = cmp(vector[i], 0)
                number_zero_crossing += 1
    return number_zero_crossing