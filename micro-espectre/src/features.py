"""
Micro-ESPectre - CSI Feature Extraction (Publish-Time)

Pure Python implementation for MicroPython.
Features are calculated ONLY at publish time, using:
  - W=1 features: skewness, kurtosis (from current packet amplitudes)
  - Turbulence buffer features: variance_turb, iqr_turb, entropy_turb

This approach:
  - No separate amplitude buffer needed (saves 92% memory)
  - Features synchronized with MVS state (no lag)
  - No background thread required

Top 5 features by Fisher's Criterion (tested with SEG_WINDOW_SIZE=50):
  - iqr_turb (J=3.56): IQR approximation of turbulence buffer
  - skewness (J=2.54): Distribution asymmetry (W=1)
  - kurtosis (J=2.24): Distribution tailedness (W=1)
  - entropy_turb (J=2.08): Shannon entropy of turbulence buffer
  - variance_turb (J=1.21): Moving variance (already calculated by MVS!)

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import math
import ubinascii
import struct

def encode_sc_amps(sc_amps):
    """Pack 44 floats as uint16 (x100) → base64 string."""
    buf = bytearray(len(sc_amps) * 2)
    for i, v in enumerate(sc_amps):
        val = min(65535, int(v * 100))
        struct.pack_into('>H', buf, i * 2, val)
    return ubinascii.b2a_base64(buf).decode().strip()


# ============================================================================
# W=1 Features (Current Packet Amplitudes)
# ============================================================================

def calc_skewness(amplitudes):
    """
    Calculate Fisher's skewness (third standardized moment).
    
    Skewness measures asymmetry of the distribution:
    - γ₁ > 0: Right-skewed (tail on right)
    - γ₁ < 0: Left-skewed (tail on left)
    - γ₁ = 0: Symmetric
    
    Args:
        amplitudes: List of amplitudes from current packet
    
    Returns:
        float: Skewness coefficient
    """
    n = len(amplitudes)
    if n < 3:
        return 0.0
    
    # Calculate mean
    mean = sum(amplitudes) / n
    
    # Calculate variance and std
    variance = sum((x - mean) ** 2 for x in amplitudes) / n
    std = math.sqrt(variance) if variance > 0 else 0
    
    if std < 1e-10:
        return 0.0
    
    # Third central moment
    m3 = sum((x - mean) ** 3 for x in amplitudes) / n
    
    return m3 / (std ** 3)


def calc_kurtosis(amplitudes):
    """
    Calculate Fisher's excess kurtosis (fourth standardized moment - 3).
    
    Kurtosis measures "tailedness" of the distribution:
    - γ₂ > 0: Leptokurtic (heavy tails, sharp peak)
    - γ₂ < 0: Platykurtic (light tails, flat peak)
    - γ₂ = 0: Mesokurtic (normal distribution)
    
    Args:
        amplitudes: List of amplitudes from current packet
    
    Returns:
        float: Excess kurtosis coefficient
    """
    n = len(amplitudes)
    if n < 4:
        return 0.0
    
    # Calculate mean
    mean = sum(amplitudes) / n
    
    # Calculate variance and std
    variance = sum((x - mean) ** 2 for x in amplitudes) / n
    std = math.sqrt(variance) if variance > 0 else 0
    
    if std < 1e-10:
        return 0.0
    
    # Fourth central moment
    m4 = sum((x - mean) ** 4 for x in amplitudes) / n
    
    # Excess kurtosis (subtract 3 for normal distribution baseline)
    return (m4 / (std ** 4)) - 3.0


# ============================================================================
# Turbulence Buffer Features
# ============================================================================

def calc_iqr_turb(turbulence_buffer, buffer_count):
    """
    Calculate IQR approximation using range (max - min) * 0.5.
    
    Args:
        turbulence_buffer: Circular buffer of turbulence values
        buffer_count: Number of valid values in buffer
    
    Returns:
        float: IQR approximation
    """
    if buffer_count < 2:
        return 0.0
    
    # Find min/max in buffer
    min_val = turbulence_buffer[0]
    max_val = turbulence_buffer[0]
    
    for i in range(1, buffer_count):
        val = turbulence_buffer[i]
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val
    
    return (max_val - min_val) * 0.5


def calc_entropy_turb(turbulence_buffer, buffer_count, n_bins=10):
    """
    Calculate Shannon entropy of turbulence distribution.
    
    Args:
        turbulence_buffer: Circular buffer of turbulence values
        buffer_count: Number of valid values in buffer
        n_bins: Number of histogram bins
    
    Returns:
        float: Shannon entropy in bits
    """
    if buffer_count < 2:
        return 0.0
    
    # Find min/max
    min_val = turbulence_buffer[0]
    max_val = turbulence_buffer[0]
    
    for i in range(1, buffer_count):
        val = turbulence_buffer[i]
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val
    
    if max_val - min_val < 1e-10:
        return 0.0
    
    # Create histogram
    bin_width = (max_val - min_val) / n_bins
    bins = [0] * n_bins
    
    for i in range(buffer_count):
        val = turbulence_buffer[i]
        bin_idx = int((val - min_val) / bin_width)
        if bin_idx >= n_bins:
            bin_idx = n_bins - 1
        bins[bin_idx] += 1
    
    # Calculate entropy
    entropy = 0.0
    for count in bins:
        if count > 0:
            p = count / buffer_count
            entropy -= p * math.log(p) / math.log(2)  # log2
    
    return entropy

# ============================================================================
# Spatial Classification Features
# ============================================================================

def calc_amplitude_mean(amplitudes):
    """
    Calculates Mean amplitude

    Args:
        amplitudes (list): List of amplitudes

    Returns:
        float: Mean amplitude
    """
    if (len(amplitudes) == 0):
        return 0.0
    return sum(amplitudes) / len(amplitudes)

def calc_amplitude_range(amplitudes):
    """Calculates peak-to-peak range for spatial differences

    Args:
        amplitudes (list): List of amplitudes

    Returns:
        float: amplitude range
    """
    if (len(amplitudes) == 0):
        return 0.0
    return (max(amplitudes) - min(amplitudes))

def calc_amplitude_standard_deviation(amplitudes):
    """Calculate standard deviation of amplitudes
    
    Args:
        amplitudes (list): List of amplitudes
    
    Returns:
        float: Standard deviation
    """
    num_subcarriers = len(amplitudes)
    if num_subcarriers < 2:
        return 0.0
    
    mean = sum(amplitudes) / num_subcarriers
    variance = sum((x - mean) ** 2 for x in amplitudes) / num_subcarriers
    return math.sqrt(variance) if variance > 0 else 0.0

def calc_subband_features(amplitudes):
    """
    Calculate mean amplitude for low, mid, and high frequency subbands.

    Splits subcarriers into three equal bands and computes the mean of
    each. Different spatial positions produce distinct subband energy
    profiles due to frequency-selective multipath.

    Args:
        amplitudes: List of subcarrier amplitudes

    Returns:
        dict: Keys amp_mean_low, amp_mean_mid, amp_mean_high (floats)
    """
    num_subcarriers = len(amplitudes)
    if (num_subcarriers < 3):
        return {
            'amp_mean_low':0,
            'amp_mean_mid':0,
            'amp_mean_high':0
        }
        
    third = num_subcarriers // 3
    low_band = amplitudes[:third]
    mid_band = amplitudes[third:2*third]
    high_band = amplitudes[2*third:]
    
    return {
        'amp_mean_low': sum(low_band) / len(low_band) if low_band else 0,
        'amp_mean_mid': sum(mid_band) / len(mid_band) if mid_band else 0,
        'amp_mean_high': sum(high_band) / len(high_band) if high_band else 0,
    }

    
def calc_iqr_turb_real(turbulence_buffer, buffer_count):
    """
    Write all subcarrier amplitudes into a pre-allocated buffer.

    Reuses the same list object on every call to avoid heap allocation
    on the ESP32, which reduces GC pressure during the main loop.

    Args:
        amplitudes: List of subcarrier amplitudes (up to 64 values)
        buf: Pre-allocated list of length 64 (mutated in place)

    Returns:
        list: The same buf object, updated with current amplitudes
              rounded to 2 decimal places
    """
    buf = sorted(turbulence_buffer[:buffer_count])
    n = len(buf)
    q1 = buf[n // 4]
    q3 = buf[(3 * n) // 4]
    return q3 - q1

def calc_all_subcarrier_amps(amplitudes, buf):
    """
    Write all filtered amplitudes into pre-allocated buffer.
    """
    NULL_SC_INDICES = frozenset([0,1,2,3,4,5,27,28,29,30,31,32,33,34,35,59,60,61,62,63])
    VALID_SC_INDICES = [i for i in range(64) if i not in NULL_SC_INDICES]
    for out_idx, sc_idx in enumerate(VALID_SC_INDICES):
        if sc_idx < len(amplitudes):
            buf[out_idx] = round(amplitudes[sc_idx], 2)
    return buf[:44]

# ============================================================================
# Channel Invariant Features
# ============================================================================

def calc_phase_diff(csi_raw):
    """Differential phase between adjacent subcarriers — channel-invariant."""
    phases = []
    for i in range(0, len(csi_raw) - 1, 2):
        I = csi_raw[i] if csi_raw[i] < 128 else csi_raw[i] - 256
        Q = csi_raw[i+1] if csi_raw[i+1] < 128 else csi_raw[i+1] - 256
        if I != 0 or Q != 0:
            phases.append(math.atan2(Q, I))
    return [phases[i+1] - phases[i] for i in range(len(phases)-1)]


def calc_phase_and_diff_stats(csi_raw):
    """Single pass for both phase and differential phase stats."""
    phases = []
    for i in range(0, len(csi_raw) - 1, 2):
        I = csi_raw[i] if csi_raw[i] < 128 else csi_raw[i] - 256
        Q = csi_raw[i+1] if csi_raw[i+1] < 128 else csi_raw[i+1] - 256
        if I != 0 or Q != 0:
            phases.append(math.atan2(Q, I))

    if len(phases) < 2:
        return (
            {'phase_mean': 0.0, 'phase_std': 0.0, 'phase_range': 0.0},
            {'phase_diff_mean': 0.0, 'phase_diff_std': 0.0,
             'phase_diff_range': 0.0, 'phase_diff_skew': 0.0}
        )

    # Phase stats
    n = len(phases)
    mean = sum(phases) / n
    std = math.sqrt(sum((p - mean)**2 for p in phases) / n)
    phase_stats = {
        'phase_mean': round(mean, 4),
        'phase_std': round(std, 4),
        'phase_range': round(max(phases) - min(phases), 4)
    }

    # Diff stats (reuse phases list, no second I/Q pass)
    diffs = [phases[i+1] - phases[i] for i in range(n - 1)]
    nd = len(diffs)
    dmean = sum(diffs) / nd
    dvariance = sum((x - dmean)**2 for x in diffs) / nd
    dstd = math.sqrt(dvariance) if dvariance > 0 else 0.0
    dskew = 0.0
    if dstd > 1e-10:
        m3 = sum((x - dmean)**3 for x in diffs) / nd
        dskew = m3 / (dstd**3)
    diff_stats = {
        'phase_diff_mean':  round(dmean, 4),
        'phase_diff_std':   round(dstd, 4),
        'phase_diff_range': round(max(diffs) - min(diffs), 4),
        'phase_diff_skew':  round(dskew, 4),
    }

    return phase_stats, diff_stats


# ============================================================================
# Feature Extractor (Publish-Time)
# ============================================================================

class PublishTimeFeatureExtractor:
    """
    Feature extractor that calculates features at publish time.
    
    No internal buffer - uses turbulence buffer from SegmentationContext
    and current packet amplitudes.
    
    Features:
    - skewness (W=1): From current packet amplitudes
    - kurtosis (W=1): From current packet amplitudes  
    - variance_turb: From MVS (already calculated)
    - iqr_turb: From turbulence buffer
    - entropy_turb: From turbulence buffer
    """
    
    def __init__(self):
        """Initialize feature extractor."""
        self.last_features = None
        self._sc_buf = [0.0] * 64  # pre-allocated, reused every call
    
    def compute_features(self, amplitudes, turbulence_buffer, buffer_count, moving_variance, csi_raw=None, selected_indices=None):
        """
        Compute all features at publish time.
        
        Args:
            amplitudes: Current packet amplitudes (list)
            turbulence_buffer: Circular buffer of turbulence values
            buffer_count: Number of valid values in turbulence buffer
            moving_variance: Already calculated by MVS
            amp_mean: Mean Amplitude
        
        Returns:
            dict: All 7 features
        """
        subband_features = calc_subband_features(amplitudes)
        if csi_raw is not None:
            phase_features, phase_diff_stats = calc_phase_and_diff_stats(csi_raw)
        else:
            phase_features = {'phase_mean': 0.0, 'phase_std': 0.0, 'phase_range': 0.0}
            phase_diff_stats = {'phase_diff_mean': 0.0, 'phase_diff_std': 0.0,
                                'phase_diff_range': 0.0, 'phase_diff_skew': 0.0}
        self.last_features = {
            # W=1 features (current packet)
            'skewness': calc_skewness(amplitudes),
            'kurtosis': calc_kurtosis(amplitudes),
            
            # Turbulence buffer features
            'variance_turb': moving_variance,  # Already calculated by MVS!
            'iqr_turb': calc_iqr_turb_real(turbulence_buffer, buffer_count), # more granular
            'entropy_turb': calc_entropy_turb(turbulence_buffer, buffer_count),
            
            # Spatial classification features
            'amp_mean': calc_amplitude_mean(amplitudes),
            'amp_range': calc_amplitude_range(amplitudes),
            'amp_std': calc_amplitude_standard_deviation(amplitudes),
            
            # Subband features
            'amp_mean_low': subband_features['amp_mean_low'],
            'amp_mean_mid': subband_features['amp_mean_mid'],
            'amp_mean_high': subband_features['amp_mean_high'],
            'sc_amps': encode_sc_amps(calc_all_subcarrier_amps(amplitudes, self._sc_buf)),
                        
            # Phase features from raw I/Q
            'phase_mean':  phase_features['phase_mean'],
            'phase_std':   phase_features['phase_std'],
            'phase_range': phase_features['phase_range'],

            # Channel Invariant Features
            'phase_diff_mean':  phase_diff_stats['phase_diff_mean'],
            'phase_diff_std':   phase_diff_stats['phase_diff_std'],
            'phase_diff_range': phase_diff_stats['phase_diff_range'],
            'phase_diff_skew':  phase_diff_stats['phase_diff_skew'],
        }
        
        return self.last_features
    
    def get_features(self):
        """Get last computed features."""
        return self.last_features


# ============================================================================
# Multi-Feature Detector (Confidence-based)
# ============================================================================

class MultiFeatureDetector:
    """
    Multi-feature motion detector with confidence scoring.
    
    Uses top 5 features for robust detection.
    Returns confidence score (0-1) instead of binary state.
    
    Thresholds derived from testing (14_test_publish_time_features.py) with window=50:
    - iqr_turb: J=3.56, threshold=2.18
    - entropy_turb: J=2.08, threshold=2.94
    - variance_turb: J=1.21, threshold=0.99
    - skewness: J=2.54, threshold=0.57
    - kurtosis: J=2.24, threshold=-1.33 (below)
    """
    
    # Thresholds derived from testing with SEG_WINDOW_SIZE=50
    # Weights proportional to Fisher's Criterion
    DEFAULT_THRESHOLDS = {
        'iqr_turb': {'threshold': 2.18, 'weight': 1.0, 'direction': 'above'},
        'skewness': {'threshold': 0.57, 'weight': 0.71, 'direction': 'above'},
        'kurtosis': {'threshold': -1.33, 'weight': 0.63, 'direction': 'below'},
        'entropy_turb': {'threshold': 2.94, 'weight': 0.58, 'direction': 'above'},
        'variance_turb': {'threshold': 0.99, 'weight': 0.34, 'direction': 'above'},
    }
    
    def __init__(self, thresholds=None, min_confidence=0.5):
        """
        Initialize multi-feature detector.
        
        Args:
            thresholds: Dict of feature thresholds (or None for defaults)
            min_confidence: Minimum confidence to declare motion (0-1)
        """
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
        self.min_confidence = min_confidence
        self.total_weight = sum(t['weight'] for t in self.thresholds.values())
        
        self.last_confidence = 0.0
        self.last_triggered = []
    
    def detect(self, features):
        """
        Detect motion based on multiple features.
        
        Args:
            features: Dict of feature values
        
        Returns:
            tuple: (is_motion, confidence, triggered_features)
        """
        if features is None:
            return False, 0.0, []
        
        triggered = []
        weighted_score = 0.0
        
        for name, config in self.thresholds.items():
            if name not in features:
                continue
            
            value = features[name]
            threshold = config['threshold']
            weight = config['weight']
            direction = config['direction']
            
            # Check if feature triggers
            if direction == 'above' and value > threshold:
                triggered.append(name)
                weighted_score += weight
            elif direction == 'below' and value < threshold:
                triggered.append(name)
                weighted_score += weight
        
        # Calculate confidence (0-1)
        confidence = weighted_score / self.total_weight if self.total_weight > 0 else 0.0
        is_motion = confidence >= self.min_confidence
        
        self.last_confidence = confidence
        self.last_triggered = triggered
        
        return is_motion, confidence, triggered
    
    def get_confidence(self):
        """Get last computed confidence."""
        return self.last_confidence
    
    def get_triggered(self):
        """Get list of last triggered features."""
        return self.last_triggered
