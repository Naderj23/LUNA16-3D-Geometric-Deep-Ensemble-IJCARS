"""Modular 3D Geometric-Harmonic and Multi-Task CoordConv Architecture for LUNA16 FPRED."""
from .proposed_model import ProposedHybrid3DNoduleCAD
from .backbone import MultiWindowCoordConv3DResNeXtWithDiameterHead, FoldIsolatedTriPlanarMIPConvNet
from .attention_module import compute_analytical_bottleneck_snr_db, apply_8view_cube_equivariance
from .physical_3d_descriptors import (
    FAMILY_DIMENSIONS_1166D,
    extract_all_9families_from_3d_patches_array,
    compute_exact_3x3_symmetric_hessian_eigenvalues,
)
from .detection_head import (
    Stage0GPUMultiScaleScreener,
    recenter_3d_patches_to_local_foveal_centroid_gpu,
    apply_monotonic_two_stage_cascade_calibration,
)
from .feature_fusion import run_v106_9family_coordconv_multitask_upgrade_on_split
