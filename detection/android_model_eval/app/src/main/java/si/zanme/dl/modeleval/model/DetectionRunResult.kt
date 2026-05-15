package si.zanme.dl.modeleval.model

data class DetectionRunResult(
    val modelKey: String,
    val modelLabel: String,
    val threshold: Float,
    val boxes: List<DetectionBox>,
    val inferenceLatencyMs: Float,
)
