package si.zanme.dl.modeleval.ui

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import si.zanme.dl.modeleval.AppViewModel
import si.zanme.dl.modeleval.logging.RunRecord
import si.zanme.dl.modeleval.model.AppModelSpec
import si.zanme.dl.modeleval.model.AppUiState
import si.zanme.dl.modeleval.model.DetectionBox
import si.zanme.dl.modeleval.model.ImageSourceType
import si.zanme.dl.modeleval.model.ModelStatus
import si.zanme.dl.modeleval.model.OverlayMode
import si.zanme.dl.modeleval.model.ThresholdOptions
import si.zanme.dl.modeleval.model.selectedModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppScreen(viewModel: AppViewModel) {
    val context = LocalContext.current
    val state = viewModel.uiState
    var showLogSheet by rememberSaveable { mutableStateOf(false) }
    var pendingCameraUri by remember { mutableStateOf<Uri?>(null) }

    val cameraLauncher = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { success ->
        val uri = pendingCameraUri
        pendingCameraUri = null
        if (success && uri != null) {
            viewModel.importImage(context, uri, ImageSourceType.Camera)
        }
    }
    val galleryLauncher = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) {
            viewModel.importImage(context, uri, ImageSourceType.Gallery)
        }
    }

    if (showLogSheet) {
        SessionLogSheet(
            records = state.sessionLog,
            onDismiss = { showLogSheet = false },
            onShare = {
                viewModel.buildShareLogIntent()?.let { intent ->
                    context.startActivity(Intent.createChooser(intent, "Share session log"))
                }
            },
            onClear = viewModel::clearSessionLog,
        )
    }

    Scaffold(
        topBar = {
            CenterAlignedTopAppBar(
                title = { Text("Manual Model Eval") },
                actions = {
                    TextButton(onClick = { showLogSheet = true }) {
                        Text("Session Log")
                    }
                },
            )
        },
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            ModelSelector(
                models = state.availableModels,
                selected = state.selectedModelKey,
                status = state.modelLoadState,
                onSelect = viewModel::loadModel,
            )

            ThresholdSelector(
                selected = state.selectedThreshold,
                onSelect = viewModel::updateThreshold,
            )

            ActionRow(
                hasImage = state.selectedImage != null,
                onTakePhoto = {
                    val captureUri = viewModel.createCameraCaptureUri(context)
                    pendingCameraUri = captureUri
                    cameraLauncher.launch(captureUri)
                },
                onPickGallery = {
                    galleryLauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                },
                onRunAgain = viewModel::runDetection,
            )

            ImagePreviewCard(
                state = state,
                onOverlayModeChange = viewModel::setOverlayMode,
            )

            MetadataCard(state = state)

            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(
                    onClick = { viewModel.saveAnnotatedImage(context) },
                    enabled = state.detectionResult != null && !state.isBusy,
                ) {
                    Text("Save Annotated Image")
                }

                if (state.isBusy) {
                    CircularProgressIndicator(modifier = Modifier.size(24.dp))
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ModelSelector(
    models: List<AppModelSpec>,
    selected: String?,
    status: si.zanme.dl.modeleval.model.ModelLoadState,
    onSelect: (String) -> Unit,
) {
    var expanded by rememberSaveable { mutableStateOf(false) }
    val selectedLabel = models.firstOrNull { it.key == selected }?.label ?: "Select model"

    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Model", fontWeight = FontWeight.SemiBold)
            ExposedDropdownMenuBox(
                expanded = expanded,
                onExpandedChange = { expanded = !expanded },
            ) {
                OutlinedTextField(
                    modifier = Modifier
                        .menuAnchor()
                        .fillMaxWidth(),
                    readOnly = true,
                    value = selectedLabel,
                    onValueChange = {},
                    label = { Text("Active model") },
                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
                )
                DropdownMenu(
                    expanded = expanded,
                    onDismissRequest = { expanded = false },
                ) {
                    models.forEach { model ->
                        DropdownMenuItem(
                            text = { Text(model.label) },
                            onClick = {
                                expanded = false
                                onSelect(model.key)
                            },
                        )
                    }
                }
            }
            Text(
                when (status.status) {
                    ModelStatus.IDLE -> "Model idle"
                    ModelStatus.LOADING -> status.message ?: "Loading model..."
                    ModelStatus.READY -> "Model ready"
                    ModelStatus.ERROR -> status.message ?: "Model error"
                },
                color = when (status.status) {
                    ModelStatus.ERROR -> MaterialTheme.colorScheme.error
                    else -> MaterialTheme.colorScheme.onSurfaceVariant
                },
            )
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ThresholdSelector(
    selected: Float,
    onSelect: (Float) -> Unit,
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Confidence threshold", fontWeight = FontWeight.SemiBold)
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                ThresholdOptions.presets.forEach { threshold ->
                    FilterChip(
                        selected = threshold == selected,
                        onClick = { onSelect(threshold) },
                        label = { Text(String.format(java.util.Locale.US, "%.2f", threshold)) },
                    )
                }
            }
        }
    }
}

@Composable
private fun ActionRow(
    hasImage: Boolean,
    onTakePhoto: () -> Unit,
    onPickGallery: () -> Unit,
    onRunAgain: () -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Button(onClick = onTakePhoto, modifier = Modifier.weight(1f)) {
            Text("Take Photo")
        }
        Button(onClick = onPickGallery, modifier = Modifier.weight(1f)) {
            Text("Pick from Gallery")
        }
        OutlinedButton(
            onClick = onRunAgain,
            enabled = hasImage,
            modifier = Modifier.weight(1f),
        ) {
            Text("Run Again")
        }
    }
}

@Composable
private fun ImagePreviewCard(
    state: AppUiState,
    onOverlayModeChange: (OverlayMode) -> Unit,
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(
                    selected = state.overlayMode == OverlayMode.Annotated,
                    onClick = { onOverlayModeChange(OverlayMode.Annotated) },
                    label = { Text("Annotated") },
                )
                FilterChip(
                    selected = state.overlayMode == OverlayMode.Original,
                    onClick = { onOverlayModeChange(OverlayMode.Original) },
                    label = { Text("Original") },
                )
            }

            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(420.dp)
                    .background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(16.dp)),
                contentAlignment = Alignment.Center,
            ) {
                val image = state.selectedImage
                if (image == null) {
                    Text("No image selected")
                } else {
                    AnnotatedImagePreview(
                        bitmap = image.bitmap,
                        boxes = if (state.overlayMode == OverlayMode.Annotated) {
                            state.detectionResult?.boxes.orEmpty()
                        } else {
                            emptyList()
                        },
                    )
                }
            }
        }
    }
}

@Composable
private fun AnnotatedImagePreview(
    bitmap: Bitmap,
    boxes: List<DetectionBox>,
) {
    BoxWithConstraints(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        val imageBitmap = bitmap.asImageBitmap()
        Image(
            bitmap = imageBitmap,
            contentDescription = null,
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Fit,
        )

        Canvas(modifier = Modifier.fillMaxSize()) {
            val canvasWidth = size.width
            val canvasHeight = size.height
            val scale = minOf(
                canvasWidth / bitmap.width.toFloat(),
                canvasHeight / bitmap.height.toFloat(),
            )
            val drawnWidth = bitmap.width * scale
            val drawnHeight = bitmap.height * scale
            val offsetX = (canvasWidth - drawnWidth) / 2f
            val offsetY = (canvasHeight - drawnHeight) / 2f

            boxes.forEach { box ->
                drawRect(
                    color = Color(0xFFFF5722),
                    topLeft = Offset(offsetX + box.left * scale, offsetY + box.top * scale),
                    size = Size(box.width * scale, box.height * scale),
                    style = androidx.compose.ui.graphics.drawscope.Stroke(width = 4f),
                )
            }
        }
    }
}

@Composable
private fun MetadataCard(state: AppUiState) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Result metadata", fontWeight = FontWeight.SemiBold)
            MetadataRow("Image", state.selectedImage?.displayName ?: "—")
            MetadataRow("Model", state.selectedModel?.label ?: "—")
            MetadataRow("Threshold", String.format(java.util.Locale.US, "%.2f", state.selectedThreshold))
            MetadataRow(
                "Latency",
                state.detectionResult?.let { String.format(java.util.Locale.US, "%.2f ms", it.inferenceLatencyMs) } ?: "—",
            )
            MetadataRow("Detected people", state.detectionResult?.boxes?.size?.toString() ?: "—")
            MetadataRow("Status", state.infoMessage ?: "—")
            if (state.isResultStale) {
                Text(
                    "Current result is stale. Tap Run Again to compare this image with the updated model or threshold.",
                    color = MaterialTheme.colorScheme.tertiary,
                    fontSize = 13.sp,
                )
            }
        }
    }
}

@Composable
private fun MetadataRow(label: String, value: String) {
    Row(modifier = Modifier.fillMaxWidth()) {
        Text(label, modifier = Modifier.width(120.dp), fontWeight = FontWeight.Medium)
        Text(value)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SessionLogSheet(
    records: List<RunRecord>,
    onDismiss: () -> Unit,
    onShare: () -> Unit,
    onClear: () -> Unit,
) {
    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Button(onClick = onShare, enabled = records.isNotEmpty()) {
                    Text("Share CSV")
                }
                OutlinedButton(onClick = onClear, enabled = records.isNotEmpty()) {
                    Text("Clear Log")
                }
            }
            Spacer(modifier = Modifier.height(12.dp))
            if (records.isEmpty()) {
                Text("No logged runs yet.")
            } else {
                LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(records) { record ->
                        Card(modifier = Modifier.fillMaxWidth()) {
                            Column(modifier = Modifier.padding(12.dp)) {
                                Text(record.sourceDisplayName, fontWeight = FontWeight.SemiBold)
                                Spacer(modifier = Modifier.height(4.dp))
                                Text("${record.modelKey} | thr ${String.format(java.util.Locale.US, "%.2f", record.threshold)}")
                                Text("${record.detectionCount} detections | ${String.format(java.util.Locale.US, "%.2f ms", record.inferenceLatencyMs)}")
                                Text(record.timestampIso, color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 12.sp)
                            }
                        }
                        HorizontalDivider()
                    }
                }
            }
            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}
