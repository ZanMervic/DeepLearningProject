package si.zanme.dl.modeleval.model

import android.graphics.Bitmap
import android.net.Uri

enum class ImageSourceType {
    Camera,
    Gallery,
}

data class SelectedImage(
    val uri: Uri,
    val displayName: String,
    val sourceType: ImageSourceType,
    val bitmap: Bitmap,
)
