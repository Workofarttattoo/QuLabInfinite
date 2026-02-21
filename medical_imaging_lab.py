# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
MEDICAL IMAGING LAB
Advanced medical imaging physics with CT, MRI, ultrasound, and image reconstruction.
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy import ndimage, signal, fft
from scipy.ndimage import rotate, gaussian_filter
from scipy.special import jv  # Bessel function for ultrasound
import warnings

@dataclass
class CTParameters:
    """CT imaging parameters."""
    kvp: float = 120  # Kilovoltage peak
    mas: float = 200  # Milliampere-seconds
    slice_thickness: float = 5.0  # mm
    pitch: float = 1.0  # Helical pitch
    detector_rows: int = 64
    rotation_time: float = 0.5  # seconds
    kernel: str = 'standard'  # Reconstruction kernel

@dataclass
class MRIParameters:
    """MRI imaging parameters."""
    field_strength: float = 1.5  # Tesla
    tr: float = 500  # Repetition time (ms)
    te: float = 20  # Echo time (ms)
    flip_angle: float = 90  # degrees
    fov: float = 256  # Field of view (mm)
    matrix_size: int = 256
    slice_thickness: float = 5.0  # mm
    sequence: str = 'spin_echo'  # Imaging sequence

@dataclass
class UltrasoundParameters:
    """Ultrasound imaging parameters."""
    frequency: float = 5.0  # MHz
    power: float = 0.1  # Acoustic power (W)
    depth: float = 150  # mm
    gain: float = 50  # dB
    focus_depth: float = 75  # mm
    frame_rate: int = 30  # fps
    beam_width: float = 2.0  # mm

class MedicalImagingLab:
    """
    Advanced medical imaging simulation laboratory.

    Features:
    - CT: Radon transform, filtered backprojection, beam hardening
    - MRI: Bloch equations, T1/T2 relaxation, k-space sampling
    - Ultrasound: Wave propagation, beamforming, Doppler
    - Image quality metrics: SNR, CNR, MTF, NPS
    - Artifact simulation and correction
    """

    def __init__(self):
        """Initialize medical imaging lab."""
        self.ct_params = CTParameters()
        self.mri_params = MRIParameters()
        self.us_params = UltrasoundParameters()

    def generate_phantom(self, size: int = 256,
                        phantom_type: str = 'shepp_logan') -> np.ndarray:
        """
        Generate standard imaging phantom for testing.

        Args:
            size: Image size
            phantom_type: Type of phantom

        Returns:
            2D phantom array
        """
        if phantom_type == 'shepp_logan':
            return self._shepp_logan_phantom(size)
        elif phantom_type == 'resolution':
            return self._resolution_phantom(size)
        elif phantom_type == 'contrast':
            return self._contrast_phantom(size)
        else:
            return np.zeros((size, size))

    def _shepp_logan_phantom(self, size: int) -> np.ndarray:
        """
        Generate Shepp-Logan head phantom.

        Classic phantom for CT/MRI testing.
        """
        phantom = np.zeros((size, size))

        # Ellipse parameters: (intensity, x0, y0, a, b, theta)
        ellipses = [
            (1.0, 0, 0, 0.69, 0.92, 0),  # Skull
            (-0.8, 0, -0.0184, 0.6624, 0.874, 0),  # Brain
            (-0.2, 0.22, 0, 0.11, 0.31, -18),  # Right ventricle
            (-0.2, -0.22, 0, 0.16, 0.41, 18),  # Left ventricle
            (0.1, 0, 0.35, 0.21, 0.25, 0),  # Blood vessel
            (0.1, 0, 0.1, 0.046, 0.023, 0),  # Small feature
            (0.1, 0, -0.1, 0.046, 0.023, 0),  # Small feature
            (0.1, -0.08, -0.605, 0.046, 0.023, 0),  # Small feature
            (0.1, 0, -0.605, 0.023, 0.023, 0),  # Small feature
            (0.1, 0.06, -0.605, 0.023, 0.046, 0)  # Small feature
        ]

        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)

        for intensity, x0, y0, a, b, theta in ellipses:
            theta_rad = np.deg2rad(theta)
            cos_t = np.cos(theta_rad)
            sin_t = np.sin(theta_rad)

            # Rotate coordinates
            X_rot = cos_t * (X - x0) + sin_t * (Y - y0)
            Y_rot = -sin_t * (X - x0) + cos_t * (Y - y0)

            # Check if inside ellipse
            mask = (X_rot/a)**2 + (Y_rot/b)**2 <= 1
            phantom[mask] += intensity

        return phantom

    def _resolution_phantom(self, size: int) -> np.ndarray:
        """Generate resolution test phantom with line pairs."""
        phantom = np.zeros((size, size))

        # Add line pairs of decreasing spacing
        spacings = [32, 16, 8, 4, 2]
        y_offset = 0

        for spacing in spacings:
            height = size // len(spacings)
            for i in range(0, size, spacing*2):
                phantom[y_offset:y_offset+height, i:i+spacing] = 1

            y_offset += height

        return phantom

    def _contrast_phantom(self, size: int) -> np.ndarray:
        """Generate contrast resolution phantom."""
        phantom = np.ones((size, size)) * 0.5

        # Add circles of varying contrast
        contrasts = [0.01, 0.02, 0.05, 0.1, 0.2]
        radius = size // 20

        for i, contrast in enumerate(contrasts):
            cx = (i + 1) * size // (len(contrasts) + 1)
            cy = size // 2

            y, x = np.ogrid[:size, :size]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            phantom[mask] = 0.5 + contrast

        return phantom

    def ct_forward_projection(self, image: np.ndarray,
                            angles: np.ndarray) -> np.ndarray:
        """
        Compute Radon transform (CT forward projection).

        Based on: Kak & Slaney (1988) "Principles of Computerized Tomographic Imaging"

        Args:
            image: 2D image
            angles: Projection angles in degrees

        Returns:
            Sinogram (projections x detectors)
        """
        sinogram = np.zeros((len(angles), image.shape[1]))

        for i, angle in enumerate(angles):
            # Rotate image
            rotated = rotate(image, angle, reshape=False, order=1)
            # Sum along rays (parallel beam)
            sinogram[i, :] = np.sum(rotated, axis=0)

        return sinogram

    def ct_filtered_backprojection(self, sinogram: np.ndarray,
                                  angles: np.ndarray,
                                  filter_type: str = 'ram_lak') -> np.ndarray:
        """
        Filtered backprojection reconstruction.

        Standard CT reconstruction algorithm.

        Args:
            sinogram: Projection data
            angles: Projection angles
            filter_type: Reconstruction filter

        Returns:
            Reconstructed image
        """
        num_angles, num_detectors = sinogram.shape
        img_size = num_detectors

        # Create filter
        freq = np.fft.fftfreq(num_detectors)
        if filter_type == 'ram_lak':
            # Ram-Lak (ramp) filter
            filter_freq = np.abs(freq)
        elif filter_type == 'shepp_logan':
            # Shepp-Logan filter
            filter_freq = np.abs(freq) * np.sinc(freq)
        elif filter_type == 'cosine':
            # Cosine filter
            filter_freq = np.abs(freq) * np.cos(np.pi * freq / 2)
        else:
            filter_freq = np.abs(freq)

        # Apply filter to projections
        filtered_sinogram = np.zeros_like(sinogram)
        for i in range(num_angles):
            # FFT of projection
            proj_fft = np.fft.fft(sinogram[i, :])
            # Apply filter
            filtered_proj_fft = proj_fft * filter_freq
            # Inverse FFT
            filtered_sinogram[i, :] = np.real(np.fft.ifft(filtered_proj_fft))

        # Backprojection
        reconstructed = np.zeros((img_size, img_size))

        for i, angle in enumerate(angles):
            # Create projection image
            proj_img = np.tile(filtered_sinogram[i, :], (img_size, 1))
            # Rotate back
            rotated_back = rotate(proj_img, -angle, reshape=False, order=1)
            # Add to reconstruction
            reconstructed += rotated_back

        # Normalize
        reconstructed *= np.pi / (2 * len(angles))

        return reconstructed

    def ct_simulate_scan(self, phantom: np.ndarray,
                        params: Optional[CTParameters] = None) -> Dict[str, np.ndarray]:
        """
        Simulate complete CT scan with artifacts.

        Args:
            phantom: Input phantom
            params: CT parameters

        Returns:
            Dictionary with sinogram and reconstructed image
        """
        if params is None:
            params = self.ct_params

        # Generate projection angles
        num_angles = int(360 / params.pitch)
        angles = np.linspace(0, 180, num_angles, endpoint=False)

        # Forward projection
        sinogram = self.ct_forward_projection(phantom, angles)

        # Add noise (Poisson noise based on photon count)
        photon_count = params.kvp * params.mas * 1000
        sinogram_noisy = np.random.poisson(photon_count * np.exp(-sinogram/1000)) / photon_count
        sinogram_noisy = -np.log(sinogram_noisy + 1e-6) * 1000

        # Beam hardening artifact
        if params.kvp < 140:
            beam_hardening = 0.1 * sinogram**2 / np.max(sinogram)
            sinogram_noisy += beam_hardening

        # Reconstruction
        reconstructed = self.ct_filtered_backprojection(sinogram_noisy, angles, filter_type=params.kernel)

        # Apply windowing (soft tissue window)
        window_center = 40  # HU
        window_width = 400  # HU
        reconstructed = np.clip(reconstructed, window_center - window_width/2,
                              window_center + window_width/2)

        return {
            'sinogram': sinogram,
            'sinogram_noisy': sinogram_noisy,
            'reconstructed': reconstructed,
            'angles': angles
        }

    def mri_bloch_equation_solver(self, M0: np.ndarray,
                                 T1: float, T2: float,
                                 time: np.ndarray,
                                 B1: float = 0) -> np.ndarray:
        """
        Solve Bloch equations for MRI signal evolution.

        Bloch equations describe nuclear magnetization dynamics.

        Args:
            M0: Equilibrium magnetization
            T1: Longitudinal relaxation time (ms)
            T2: Transverse relaxation time (ms)
            time: Time points (ms)
            B1: RF pulse amplitude (μT)

        Returns:
            Magnetization vector over time
        """
        # Simplified Bloch equation solution
        # Assume 90-degree pulse at t=0

        Mz = M0 * (1 - np.exp(-time/T1))  # Longitudinal recovery
        Mxy = M0 * np.exp(-time/T2)  # Transverse decay

        # Add T2* effects (field inhomogeneities)
        T2_star = T2 * 0.7  # Typical T2* < T2
        Mxy *= np.exp(-time/T2_star)

        return Mz, Mxy

    def mri_generate_kspace(self, image: np.ndarray,
                          params: Optional[MRIParameters] = None) -> np.ndarray:
        """
        Generate k-space data from image.

        k-space is the Fourier domain of the MR image.

        Args:
            image: Spatial domain image
            params: MRI parameters

        Returns:
            Complex k-space data
        """
        if params is None:
            params = self.mri_params

        # 2D Fourier transform
        kspace = np.fft.fft2(image)
        kspace = np.fft.fftshift(kspace)

        # Add phase encoding
        if params.sequence == 'spin_echo':
            # Symmetric k-space
            pass
        elif params.sequence == 'gradient_echo':
            # Asymmetric k-space (partial echo)
            kspace[:, :kspace.shape[1]//4] = 0

        return kspace

    def mri_reconstruct_from_kspace(self, kspace: np.ndarray) -> np.ndarray:
        """
        Reconstruct image from k-space data.

        Args:
            kspace: Complex k-space data

        Returns:
            Magnitude image
        """
        # Inverse Fourier transform
        kspace_shifted = np.fft.ifftshift(kspace)
        image_complex = np.fft.ifft2(kspace_shifted)

        # Take magnitude
        image = np.abs(image_complex)

        return image

    def mri_simulate_scan(self, phantom: np.ndarray,
                        tissue_params: Dict[str, Dict[str, float]],
                        params: Optional[MRIParameters] = None) -> Dict[str, np.ndarray]:
        """
        Simulate MRI acquisition with tissue contrast.

        Args:
            phantom: Tissue type map
            tissue_params: T1/T2 values for each tissue
            params: MRI parameters

        Returns:
            Dictionary with k-space and images
        """
        if params is None:
            params = self.mri_params

        # Default tissue parameters (ms)
        default_tissues = {
            'gray_matter': {'T1': 920, 'T2': 100, 'PD': 0.85},
            'white_matter': {'T1': 780, 'T2': 90, 'PD': 0.75},
            'csf': {'T1': 4000, 'T2': 2000, 'PD': 1.0},
            'fat': {'T1': 250, 'T2': 60, 'PD': 0.9},
            'muscle': {'T1': 870, 'T2': 45, 'PD': 0.8}
        }

        tissue_params = {**default_tissues, **tissue_params}

        # Calculate signal for each tissue type
        signal_map = np.zeros_like(phantom)

        for tissue_type, tissue_props in tissue_params.items():
            T1 = tissue_props['T1']
            T2 = tissue_props['T2']
            PD = tissue_props['PD']  # Proton density

            # Signal equation for spin echo
            if params.sequence == 'spin_echo':
                signal = PD * (1 - np.exp(-params.tr/T1)) * np.exp(-params.te/T2)
            elif params.sequence == 'gradient_echo':
                E1 = np.exp(-params.tr/T1)
                signal = PD * np.sin(np.deg2rad(params.flip_angle)) * (1 - E1) / (1 - E1 * np.cos(np.deg2rad(params.flip_angle)))
                signal *= np.exp(-params.te/T2)
            else:
                signal = PD

            # Apply to phantom
            tissue_mask = phantom == list(tissue_params.keys()).index(tissue_type)
            signal_map[tissue_mask] = signal

        # Generate k-space
        kspace = self.mri_generate_kspace(signal_map, params)

        # Add noise (complex Gaussian in k-space)
        noise_level = 0.01
        noise = noise_level * (np.random.randn(*kspace.shape) + 1j * np.random.randn(*kspace.shape))
        kspace_noisy = kspace + noise

        # Reconstruct
        image = self.mri_reconstruct_from_kspace(kspace_noisy)

        # Simulate different contrasts
        t1_weighted = self._simulate_t1_contrast(phantom, tissue_params)
        t2_weighted = self._simulate_t2_contrast(phantom, tissue_params)
        pd_weighted = self._simulate_pd_contrast(phantom, tissue_params)

        return {
            'kspace': kspace,
            'kspace_noisy': kspace_noisy,
            'magnitude_image': image,
            't1_weighted': t1_weighted,
            't2_weighted': t2_weighted,
            'pd_weighted': pd_weighted
        }

    def _simulate_t1_contrast(self, phantom: np.ndarray,
                             tissue_params: Dict) -> np.ndarray:
        """Simulate T1-weighted image."""
        tr_short = 500  # ms
        te_short = 10  # ms

        image = np.zeros_like(phantom)
        for i, (tissue, props) in enumerate(tissue_params.items()):
            signal = props['PD'] * (1 - np.exp(-tr_short/props['T1'])) * np.exp(-te_short/props['T2'])
            image[phantom == i] = signal

        return image

    def _simulate_t2_contrast(self, phantom: np.ndarray,
                             tissue_params: Dict) -> np.ndarray:
        """Simulate T2-weighted image."""
        tr_long = 3000  # ms
        te_long = 80  # ms

        image = np.zeros_like(phantom)
        for i, (tissue, props) in enumerate(tissue_params.items()):
            signal = props['PD'] * (1 - np.exp(-tr_long/props['T1'])) * np.exp(-te_long/props['T2'])
            image[phantom == i] = signal

        return image

    def _simulate_pd_contrast(self, phantom: np.ndarray,
                             tissue_params: Dict) -> np.ndarray:
        """Simulate proton density weighted image."""
        tr_long = 3000  # ms
        te_short = 10  # ms

        image = np.zeros_like(phantom)
        for i, (tissue, props) in enumerate(tissue_params.items()):
            signal = props['PD'] * (1 - np.exp(-tr_long/props['T1'])) * np.exp(-te_short/props['T2'])
            image[phantom == i] = signal

        return image

    def ultrasound_wave_propagation(self, medium: np.ndarray,
                                   frequency: float,
                                   speed_of_sound: float = 1540) -> np.ndarray:
        """
        Simulate ultrasound wave propagation.

        Based on: Szabo (2014) "Diagnostic Ultrasound Imaging"

        Args:
            medium: Acoustic impedance map
            frequency: Ultrasound frequency (MHz)
            speed_of_sound: m/s

        Returns:
            RF signal
        """
        # Wavelength
        wavelength = speed_of_sound / (frequency * 1e6)  # meters

        # Generate transmitted pulse
        pulse_length = 3  # cycles
        t = np.linspace(0, pulse_length/frequency, 100)
        pulse = np.sin(2 * np.pi * frequency * t) * np.exp(-t*frequency/2)

        # Simulate reflections at impedance boundaries
        rf_signal = np.zeros((medium.shape[0], medium.shape[1]))

        # Calculate reflection coefficients
        for i in range(1, medium.shape[0]):
            impedance_diff = np.diff(medium[i-1:i+1, :], axis=0)
            reflection_coeff = impedance_diff / (medium[i, :] + medium[i-1, :] + 1e-6)
            rf_signal[i, :] = reflection_coeff.flatten()

        # Apply depth-dependent attenuation
        attenuation_coeff = 0.5 * frequency  # dB/cm/MHz
        depth_cm = np.arange(medium.shape[0]) * 0.1  # Convert to cm
        attenuation = np.exp(-attenuation_coeff * depth_cm[:, np.newaxis] / 20)
        rf_signal *= attenuation

        return rf_signal

    def ultrasound_beamforming(self, rf_data: np.ndarray,
                              num_elements: int = 128,
                              focus_depth: float = 50) -> np.ndarray:
        """
        Apply delay-and-sum beamforming.

        Standard ultrasound image formation technique.

        Args:
            rf_data: Raw RF data
            num_elements: Number of transducer elements
            focus_depth: Focus depth in mm

        Returns:
            Beamformed image
        """
        # Simplified beamforming
        # Apply delays to focus at specified depth
        speed_of_sound = 1540  # m/s

        # Element positions
        element_spacing = 0.3e-3  # 0.3mm
        element_positions = np.arange(num_elements) * element_spacing

        # Calculate delays
        focus_depth_m = focus_depth * 1e-3
        delays = np.sqrt((element_positions - element_positions[num_elements//2])**2 +
                        focus_depth_m**2) / speed_of_sound

        # Apply delays (simplified - just phase shift in frequency domain)
        beamformed = np.zeros_like(rf_data)
        for i in range(num_elements):
            delay_samples = int(delays[i] * 40e6)  # 40 MHz sampling
            if i < rf_data.shape[1]:
                beamformed[:, i] = np.roll(rf_data[:, i], delay_samples)

        # Sum across elements
        image = np.sum(beamformed, axis=1)

        return image

    def ultrasound_doppler_processing(self, rf_data: np.ndarray,
                                     prf: float = 5000,
                                     angle: float = 60) -> Dict[str, np.ndarray]:
        """
        Process Doppler ultrasound for flow measurement.

        Args:
            rf_data: RF data from multiple pulses
            prf: Pulse repetition frequency (Hz)
            angle: Doppler angle (degrees)

        Returns:
            Dictionary with Doppler measurements
        """
        # Simplified color Doppler processing
        # Estimate phase shift between consecutive pulses

        # Autocorrelation method
        num_pulses = min(rf_data.shape[1], 8)
        R = np.zeros(rf_data.shape[0], dtype=complex)

        for i in range(num_pulses-1):
            R += rf_data[:, i] * np.conj(rf_data[:, i+1])

        # Calculate mean frequency (Doppler shift)
        mean_freq = np.angle(R) * prf / (2 * np.pi)

        # Convert to velocity
        c = 1540  # m/s
        f0 = 5e6  # 5 MHz carrier
        velocity = mean_freq * c / (2 * f0 * np.cos(np.deg2rad(angle)))

        # Power Doppler (flow intensity)
        power = np.abs(R)

        # Spectral Doppler (FFT of slow-time signal)
        if rf_data.shape[1] >= 64:
            spectral = np.fft.fft(rf_data[:, :64], axis=1)
            spectral_power = np.abs(spectral)**2
        else:
            spectral_power = np.zeros((rf_data.shape[0], 64))

        return {
            'velocity': velocity,
            'power': power,
            'spectral': spectral_power,
            'mean_frequency': mean_freq
        }

    def calculate_snr(self, image: np.ndarray,
                     signal_roi: Tuple[int, int, int, int],
                     noise_roi: Tuple[int, int, int, int]) -> float:
        """
        Calculate signal-to-noise ratio.

        Args:
            image: Input image
            signal_roi: Signal region (x, y, width, height)
            noise_roi: Noise region

        Returns:
            SNR in dB
        """
        x_s, y_s, w_s, h_s = signal_roi
        signal = image[y_s:y_s+h_s, x_s:x_s+w_s]
        signal_mean = np.mean(signal)

        x_n, y_n, w_n, h_n = noise_roi
        noise = image[y_n:y_n+h_n, x_n:x_n+w_n]
        noise_std = np.std(noise)

        snr = signal_mean / (noise_std + 1e-6)
        snr_db = 20 * np.log10(snr)

        return snr_db

    def calculate_cnr(self, image: np.ndarray,
                     roi1: Tuple[int, int, int, int],
                     roi2: Tuple[int, int, int, int]) -> float:
        """
        Calculate contrast-to-noise ratio.

        Args:
            image: Input image
            roi1: First region
            roi2: Second region

        Returns:
            CNR
        """
        x1, y1, w1, h1 = roi1
        region1 = image[y1:y1+h1, x1:x1+w1]
        mean1 = np.mean(region1)

        x2, y2, w2, h2 = roi2
        region2 = image[y2:y2+h2, x2:x2+w2]
        mean2 = np.mean(region2)

        noise_std = np.std(region2)

        cnr = np.abs(mean1 - mean2) / (noise_std + 1e-6)

        return cnr

    def calculate_mtf(self, edge_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate modulation transfer function from edge image.

        MTF measures spatial resolution.

        Args:
            edge_image: Image of sharp edge

        Returns:
            Spatial frequencies and MTF values
        """
        # Find edge
        edge_profile = np.mean(edge_image, axis=0)

        # Differentiate to get line spread function
        lsf = np.diff(edge_profile)

        # FFT to get MTF
        mtf = np.abs(np.fft.fft(lsf))
        mtf = mtf[:len(mtf)//2]
        mtf = mtf / mtf[0]  # Normalize

        # Frequency axis
        freq = np.fft.fftfreq(len(lsf))[:len(mtf)]

        return freq, mtf

    def simulate_artifacts(self, image: np.ndarray,
                         artifact_type: str = 'motion') -> np.ndarray:
        """
        Simulate common imaging artifacts.

        Args:
            image: Input image
            artifact_type: Type of artifact

        Returns:
            Image with artifact
        """
        if artifact_type == 'motion':
            # Motion blur
            kernel = np.ones((5, 5)) / 25
            return ndimage.convolve(image, kernel)

        elif artifact_type == 'ring':
            # Ring artifact (CT)
            center = image.shape[0] // 2
            y, x = np.ogrid[:image.shape[0], :image.shape[1]]
            r = np.sqrt((x - center)**2 + (y - center)**2)
            ring = 0.1 * np.sin(r * 0.5)
            return image + ring

        elif artifact_type == 'aliasing':
            # Aliasing (MRI)
            wrapped = np.roll(image, image.shape[1]//4, axis=1)
            return 0.7 * image + 0.3 * wrapped

        elif artifact_type == 'beam_hardening':
            # Beam hardening (CT)
            y, x = np.ogrid[:image.shape[0], :image.shape[1]]
            center = image.shape[0] // 2
            cupping = 0.2 * ((x - center)**2 + (y - center)**2) / center**2
            return image * (1 + cupping)

        else:
            return image

    def demo(self):
        """Demonstrate medical imaging simulations."""
        print("=" * 80)
        print("MEDICAL IMAGING LAB - Advanced Physics Simulation")
        print("=" * 80)

        # Generate phantom
        print("\n1. Generating Shepp-Logan phantom...")
        phantom = self.generate_phantom(256, 'shepp_logan')
        print(f"   Phantom size: {phantom.shape}")
        print(f"   Value range: [{phantom.min():.2f}, {phantom.max():.2f}]")

        # CT Imaging
        print("\n2. CT Imaging Simulation:")
        ct_result = self.ct_simulate_scan(phantom)
        print(f"   Sinogram shape: {ct_result['sinogram'].shape}")
        print(f"   Number of projections: {len(ct_result['angles'])}")
        print(f"   Reconstruction quality (PSNR): {self._calculate_psnr(phantom, ct_result['reconstructed']):.1f} dB")

        # MRI Imaging
        print("\n3. MRI Imaging Simulation:")
        # Create tissue map
        tissue_map = np.zeros_like(phantom, dtype=int)
        tissue_map[phantom > 0.5] = 0  # Gray matter
        tissue_map[(phantom > 0) & (phantom <= 0.5)] = 1  # White matter
        tissue_map[phantom <= 0] = 2  # CSF

        tissue_params = {}  # Use defaults
        mri_result = self.mri_simulate_scan(tissue_map, tissue_params)

        print(f"   k-space size: {mri_result['kspace'].shape}")
        print(f"   T1-weighted contrast: {np.std(mri_result['t1_weighted']):.3f}")
        print(f"   T2-weighted contrast: {np.std(mri_result['t2_weighted']):.3f}")

        # Ultrasound Imaging
        print("\n4. Ultrasound Imaging Simulation:")
        # Create acoustic impedance map
        impedance_map = 1.5 + 0.5 * phantom  # Rayl units

        rf_signal = self.ultrasound_wave_propagation(impedance_map, frequency=5.0)
        print(f"   RF signal shape: {rf_signal.shape}")
        print(f"   Dynamic range: {20*np.log10(np.max(np.abs(rf_signal))/np.min(np.abs(rf_signal)+1e-6)):.1f} dB")

        # Image Quality Metrics
        print("\n5. Image Quality Assessment:")

        # SNR calculation
        signal_roi = (100, 100, 50, 50)
        noise_roi = (10, 10, 30, 30)
        snr_ct = self.calculate_snr(ct_result['reconstructed'], signal_roi, noise_roi)
        print(f"   CT SNR: {snr_ct:.1f} dB")

        # CNR calculation
        roi1 = (100, 100, 30, 30)
        roi2 = (150, 150, 30, 30)
        cnr_mri = self.calculate_cnr(mri_result['magnitude_image'], roi1, roi2)
        print(f"   MRI CNR: {cnr_mri:.2f}")

        # MTF calculation
        edge_phantom = self._create_edge_phantom(256)
        ct_edge = self.ct_simulate_scan(edge_phantom)['reconstructed']
        freq, mtf = self.calculate_mtf(ct_edge)
        mtf_50 = freq[np.where(mtf < 0.5)[0][0]] if np.any(mtf < 0.5) else freq[-1]
        print(f"   CT MTF at 50%: {mtf_50:.3f} lp/mm")

        # Artifact Simulation
        print("\n6. Artifact Simulation:")
        artifacts = ['motion', 'ring', 'aliasing', 'beam_hardening']
        for artifact in artifacts:
            artifacted = self.simulate_artifacts(ct_result['reconstructed'], artifact)
            psnr = self._calculate_psnr(ct_result['reconstructed'], artifacted)
            print(f"   {artifact.title()} artifact - PSNR: {psnr:.1f} dB")

        # Advanced Sequences
        print("\n7. Advanced Imaging Sequences:")

        # Diffusion MRI parameters
        b_values = [0, 500, 1000]  # s/mm²
        adc = 0.001  # mm²/s (typical brain)
        diffusion_signal = np.exp(-np.array(b_values) * adc)
        print(f"   Diffusion signal decay: {diffusion_signal}")

        # Perfusion imaging
        time_points = np.linspace(0, 60, 100)  # seconds
        contrast_curve = 100 * (1 - np.exp(-time_points/10)) * np.exp(-time_points/30)
        peak_time = time_points[np.argmax(contrast_curve)]
        print(f"   Perfusion peak time: {peak_time:.1f} s")

        # Spectroscopy (simplified)
        metabolites = {
            'NAA': {'freq': 2.0, 'amplitude': 1.0},  # ppm
            'Creatine': {'freq': 3.0, 'amplitude': 0.8},
            'Choline': {'freq': 3.2, 'amplitude': 0.6}
        }
        print(f"   MR Spectroscopy metabolites: {list(metabolites.keys())}")

        print("\n" + "=" * 80)
        print("Medical imaging simulation complete.")
        print("=" * 80)

    def _calculate_psnr(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate peak signal-to-noise ratio."""
        mse = np.mean((original - reconstructed)**2)
        if mse == 0:
            return 100
        max_val = np.max(original)
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
        return psnr

    def _create_edge_phantom(self, size: int) -> np.ndarray:
        """Create edge phantom for MTF measurement."""
        phantom = np.zeros((size, size))
        phantom[:, size//2:] = 1
        # Slight rotation for better edge
        phantom = rotate(phantom, 2, reshape=False)
        return phantom

if __name__ == "__main__":
    lab = MedicalImagingLab()
    lab.demo()