mod assembler;
mod audibility;
mod duration;
mod options;
mod slicing;
mod timeline;
mod vad;

pub use assembler::{
    LongFormAssembleStats, SegmentMergePolicy, SegmentTimeDomain, SliceTranscript,
    TranscriptAssembler,
};
pub(crate) use duration::{ExecutorWindowLimitError, executor_window_limit_samples};
pub use options::{LongFormMode, LongFormOptions, LongFormOptionsError, LongFormVadOptions};
pub use slicing::{
    AudioSlice, AudioSliceKind, LongFormBenchmarkMetadata, LongFormSliceError, LongFormSlicePlan,
    LongFormSliceStats, LongFormVadProvider, LongFormVadProviderError, LongFormVadProviderKind,
    LongFormVadSlice, plan_longform_slices,
};
pub(crate) use slicing::{
    LongFormSlicePlanningError, VAD_SLICE_DECODE_MIN_SPEECH_SAMPLES,
    plan_longform_slices_with_materialization_gate, vad_speech_spans_overlap_samples,
};
pub use timeline::{TimelineAnchor, TimelineMap};
