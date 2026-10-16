"""Decoded-record evaluation for the ULIFlow pipeline."""

from .decoded_records import (
    Stage2DecodedSampleRecord,
    Stage2WorldMotionArrays,
    build_stage2_eval128_decoded_record,
    read_stage2_decoded_record,
    split_batched_world_arrays,
    write_stage2_decoded_record,
)
from .stage1_ceiling import compute_stage1_reconstruction_ceiling
from .windowed_stage1_ceiling import (
    build_local_stage1_window_batch,
    compute_windowed_stage1_reconstruction_ceiling,
)
from .interact_feature_evaluator import (
    InterActFeatureArtifactPaths,
    InterActFeatureEvaluatorUnavailable,
    build_interact_feature_evaluator_from_artifacts,
    default_interact_feature_artifact_paths,
    load_interact_feature_artifacts,
    run_interact_feature_parity_check,
    sha256_file,
)
from .marker_bps_audit import read_marker_bps_alignment_report, write_marker_bps_alignment_report
from .quantitative import (
    evaluate_stage2_decoded_records,
    read_stage2_decoded_records_from_dir,
    write_stage2_quantitative_outputs,
)
from .reporting import summarize_stage2_repeat_runs, write_stage2_repeat_report
from .split_audit import (
    audit_interact_official_split_no_leak,
    read_interact_split_audit_report,
    write_interact_split_audit_report,
)
