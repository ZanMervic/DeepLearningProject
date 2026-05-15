package si.zanme.dl.modeleval.logging

data class RunRecord(
    val timestampIso: String,
    val sourceType: String,
    val sourceDisplayName: String,
    val modelKey: String,
    val inputSize: Int,
    val threshold: Float,
    val imageWidth: Int,
    val imageHeight: Int,
    val detectionCount: Int,
    val inferenceLatencyMs: Float,
)
