package si.zanme.dl.modeleval.model

import si.zanme.dl.modeleval.logging.RunRecord

data class AppUiState(
    val availableModels: List<AppModelSpec> = emptyList(),
    val selectedModelKey: String? = null,
    val modelLoadState: ModelLoadState = ModelLoadState(),
    val selectedThreshold: Float = ThresholdOptions.default,
    val selectedImage: SelectedImage? = null,
    val detectionResult: DetectionRunResult? = null,
    val overlayMode: OverlayMode = OverlayMode.Annotated,
    val isBusy: Boolean = false,
    val isResultStale: Boolean = false,
    val sessionLog: List<RunRecord> = emptyList(),
    val infoMessage: String? = null,
)

val AppUiState.selectedModel: AppModelSpec?
    get() = availableModels.firstOrNull { it.key == selectedModelKey }
