package si.zanme.dl.modeleval

import android.app.Application
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import androidx.core.content.FileProvider
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import si.zanme.dl.modeleval.catalog.ModelCatalogLoader
import si.zanme.dl.modeleval.detector.TfliteDetectorEngine
import si.zanme.dl.modeleval.logging.RunLogStore
import si.zanme.dl.modeleval.logging.RunRecord
import si.zanme.dl.modeleval.model.AppUiState
import si.zanme.dl.modeleval.model.ImageSourceType
import si.zanme.dl.modeleval.model.ModelStatus
import si.zanme.dl.modeleval.model.ModelLoadState
import si.zanme.dl.modeleval.model.OverlayMode
import si.zanme.dl.modeleval.model.SelectedImage
import si.zanme.dl.modeleval.model.ThresholdOptions
import si.zanme.dl.modeleval.model.selectedModel
import si.zanme.dl.modeleval.storage.AnnotatedImageSaver
import si.zanme.dl.modeleval.storage.UriImageLoader
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

class AppViewModel(
    application: Application,
) : AndroidViewModel(application) {
    var uiState by mutableStateOf(AppUiState())
        private set

    private val modelCatalogLoader = ModelCatalogLoader()
    private val detectorEngine = TfliteDetectorEngine(application.applicationContext)
    private val runLogStore = RunLogStore(application.applicationContext)
    private val uriImageLoader = UriImageLoader()
    private val imageSaver = AnnotatedImageSaver()

    init {
        viewModelScope.launch {
            val models = withContext(Dispatchers.IO) {
                modelCatalogLoader.load(getApplication<Application>().assets)
            }
            val log = withContext(Dispatchers.IO) { runLogStore.readAll().asReversed() }
            val firstModel = models.firstOrNull()
            uiState = uiState.copy(
                availableModels = models,
                selectedModelKey = firstModel?.key,
                selectedThreshold = firstModel?.defaultThreshold ?: ThresholdOptions.default,
                sessionLog = log,
                modelLoadState = if (firstModel == null) {
                    ModelLoadState(ModelStatus.ERROR, "No models configured.")
                } else {
                    ModelLoadState(ModelStatus.IDLE, null)
                },
            )
            firstModel?.let { loadModel(it.key) }
        }
    }

    override fun onCleared() {
        detectorEngine.close()
        super.onCleared()
    }

    fun createCameraCaptureUri(context: Context): Uri {
        val cameraDir = File(context.cacheDir, "camera").apply { mkdirs() }
        val fileName = "capture_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())}.jpg"
        val file = File(cameraDir, fileName)
        return FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            file,
        )
    }

    fun loadModel(modelKey: String) {
        val model = uiState.availableModels.firstOrNull { it.key == modelKey } ?: return
        uiState = uiState.copy(
            selectedModelKey = modelKey,
            selectedThreshold = uiState.selectedThreshold.takeIf { it in ThresholdOptions.presets } ?: model.defaultThreshold,
            modelLoadState = ModelLoadState(ModelStatus.LOADING, "Loading ${model.label}"),
            infoMessage = null,
            isResultStale = uiState.selectedImage != null,
        )

        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) { detectorEngine.loadModel(model) }
            uiState = result.fold(
                onSuccess = {
                    uiState.copy(
                        modelLoadState = ModelLoadState(ModelStatus.READY, null),
                        infoMessage = "${model.label} ready",
                    )
                },
                onFailure = { error ->
                    uiState.copy(
                        modelLoadState = ModelLoadState(ModelStatus.ERROR, error.message ?: "Failed to load model"),
                        infoMessage = error.message ?: "Failed to load model",
                    )
                },
            )
        }
    }

    fun updateThreshold(value: Float) {
        uiState = uiState.copy(
            selectedThreshold = value,
            isResultStale = uiState.selectedImage != null,
            infoMessage = "Threshold set to ${String.format(Locale.US, "%.2f", value)}",
        )
    }

    fun setOverlayMode(mode: OverlayMode) {
        uiState = uiState.copy(overlayMode = mode)
    }

    fun importImage(context: Context, uri: Uri, sourceType: ImageSourceType) {
        uiState = uiState.copy(isBusy = true, infoMessage = "Loading image...")
        viewModelScope.launch {
            val loaded = withContext(Dispatchers.IO) { runCatching { uriImageLoader.load(context, uri) } }
            loaded.fold(
                onSuccess = { image ->
                    uiState = uiState.copy(
                        selectedImage = SelectedImage(
                            uri = uri,
                            displayName = image.displayName,
                            sourceType = sourceType,
                            bitmap = image.bitmap,
                        ),
                        detectionResult = null,
                        isBusy = false,
                        isResultStale = false,
                        infoMessage = "Loaded ${image.displayName}",
                    )
                    runDetection()
                },
                onFailure = { error ->
                    uiState = uiState.copy(
                        isBusy = false,
                        infoMessage = error.message ?: "Failed to load image",
                    )
                },
            )
        }
    }

    fun runDetection() {
        val image = uiState.selectedImage ?: return
        val model = uiState.selectedModel ?: return

        if (uiState.modelLoadState.status != ModelStatus.READY) {
            uiState = uiState.copy(
                isResultStale = true,
                infoMessage = "Selected model is not ready yet. Run again once it finishes loading.",
            )
            return
        }

        uiState = uiState.copy(isBusy = true, infoMessage = "Running detection...")
        viewModelScope.launch {
            val detection = withContext(Dispatchers.IO) {
                detectorEngine.detect(
                    bitmap = image.bitmap,
                    threshold = uiState.selectedThreshold,
                    iouThreshold = 0.45f,
                    maxDet = 300,
                )
            }
            detection.fold(
                onSuccess = { result ->
                    val record = RunRecord(
                        timestampIso = isoTimestamp(),
                        sourceType = image.sourceType.name.lowercase(Locale.US),
                        sourceDisplayName = image.displayName,
                        modelKey = model.key,
                        inputSize = model.inputSize,
                        threshold = uiState.selectedThreshold,
                        imageWidth = image.bitmap.width,
                        imageHeight = image.bitmap.height,
                        detectionCount = result.boxes.size,
                        inferenceLatencyMs = result.inferenceLatencyMs,
                    )
                    withContext(Dispatchers.IO) { runLogStore.append(record) }
                    uiState = uiState.copy(
                        detectionResult = result,
                        sessionLog = listOf(record) + uiState.sessionLog,
                        isBusy = false,
                        isResultStale = false,
                        infoMessage = "Detected ${result.boxes.size} people in ${String.format(Locale.US, "%.2f", result.inferenceLatencyMs)} ms",
                    )
                },
                onFailure = { error ->
                    uiState = uiState.copy(
                        isBusy = false,
                        infoMessage = error.message ?: "Detection failed",
                    )
                },
            )
        }
    }

    fun saveAnnotatedImage(context: Context) {
        val image = uiState.selectedImage ?: return
        val result = uiState.detectionResult ?: return

        uiState = uiState.copy(isBusy = true, infoMessage = "Saving annotated image...")
        viewModelScope.launch {
            val saveResult = withContext(Dispatchers.IO) {
                runCatching {
                    imageSaver.save(
                        context = context,
                        sourceBitmap = image.bitmap,
                        boxes = result.boxes,
                        sourceDisplayName = image.displayName,
                        modelKey = result.modelKey,
                        threshold = result.threshold,
                    )
                }
            }
            uiState = saveResult.fold(
                onSuccess = { uri ->
                    uiState.copy(
                        isBusy = false,
                        infoMessage = "Saved annotated image: $uri",
                    )
                },
                onFailure = { error ->
                    uiState.copy(
                        isBusy = false,
                        infoMessage = error.message ?: "Failed to save image",
                    )
                },
            )
        }
    }

    fun buildShareLogIntent(): Intent? {
        if (uiState.sessionLog.isEmpty()) {
            return null
        }
        val uri = runLogStore.shareUri()
        return Intent(Intent.ACTION_SEND).apply {
            type = "text/csv"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
    }

    fun clearSessionLog() {
        runLogStore.clear()
        uiState = uiState.copy(
            sessionLog = emptyList(),
            infoMessage = "Session log cleared",
        )
    }

    private fun isoTimestamp(): String =
        SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US).format(Date())
}
