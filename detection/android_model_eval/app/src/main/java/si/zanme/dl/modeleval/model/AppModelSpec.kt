package si.zanme.dl.modeleval.model

data class AppModelSpec(
    val key: String,
    val label: String,
    val assetFile: String,
    val inputSize: Int,
    val defaultThreshold: Float,
)
