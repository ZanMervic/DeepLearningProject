package si.zanme.dl.modeleval.model

enum class ModelStatus {
    IDLE,
    LOADING,
    READY,
    ERROR,
}

data class ModelLoadState(
    val status: ModelStatus = ModelStatus.IDLE,
    val message: String? = null,
)
