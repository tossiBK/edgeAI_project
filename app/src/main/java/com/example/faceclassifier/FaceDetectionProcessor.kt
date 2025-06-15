import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import androidx.core.graphics.createBitmap

class FaceDetectionProcessor(private val context: Context) {
    private val faceDetector: CascadeClassifier
    // Add a secondary cascade for verification - we sometimes not precise, but still not fully
    // TODO work on better detection if ever needed this project for more and precise results
    private val lbpFaceDetector: CascadeClassifier

    init {
        // Copy the Haar cascade file to a temp file so it can be loaded by OpenCV
        val haarCascadeFile = File(context.cacheDir, "haarcascade_frontalface_default.xml")

        if (!haarCascadeFile.exists()) {
            // Copy from assets to cache directory
            context.assets.open("haarcascade_frontalface_default.xml").use { inputStream ->
                FileOutputStream(haarCascadeFile).use { outputStream ->
                    val buffer = ByteArray(4096)
                    var bytesRead: Int
                    while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                        outputStream.write(buffer, 0, bytesRead)
                    }
                }
            }
        }

        // Copy the LBP cascade file to a temp file
        val lbpCascadeFile = File(context.cacheDir, "lbpcascade_frontalface.xml")

        if (!lbpCascadeFile.exists()) {
            // Copy from assets to cache directory
            context.assets.open("lbpcascade_frontalface.xml").use { inputStream ->
                FileOutputStream(lbpCascadeFile).use { outputStream ->
                    val buffer = ByteArray(4096)
                    var bytesRead: Int
                    while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                        outputStream.write(buffer, 0, bytesRead)
                    }
                }
            }
        }

        faceDetector = CascadeClassifier(haarCascadeFile.absolutePath)
        lbpFaceDetector = CascadeClassifier(lbpCascadeFile.absolutePath)

        if (faceDetector.empty() || lbpFaceDetector.empty()) {
            throw Exception("Could not load face cascade classifiers")
        }
    }

    /**
     * Process an image from URI: convert to grayscale, detect faces,
     * extract faces at two sizes, and add frame to original image
     */
    fun processImage(imageUri: Uri): ProcessedResult {
        // TODO again, we need to adjust maybe more here in general for better detection and extraction
        // Load the image from URI with proper orientation
        val originalBitmap = getBitmapFromUri(imageUri)

        // Convert Bitmap to Mat
        val originalImage = Mat()
        Utils.bitmapToMat(originalBitmap, originalImage)

        // Create a copy of the original image for framing
        val framedImage = originalImage.clone()

        // Convert to grayscale for face detection only
        val grayImage = Mat()
        Imgproc.cvtColor(originalImage, grayImage, Imgproc.COLOR_RGBA2GRAY)

        // Equalize histogram to improve detection in varying lighting conditions
        Imgproc.equalizeHist(grayImage, grayImage)

        // Apply Gaussian blur to reduce noise
        Imgproc.GaussianBlur(grayImage, grayImage, Size(3.0, 3.0), 0.0)

        // Detect faces with stricter parameters to reduce false positives
        val faces = detectFacesWithVerification(grayImage)
        val faceList = faces.toArray()
        println("Found ${faceList.size} faces")

        // Extract faces and add frame to original image
        val extractedFaces = mutableListOf<ExtractedFace>()

        // Define neon green color in RGBA format (0,255,0,255)
        val neonGreen = Scalar(0.0, 255.0, 0.0, 255.0)

        for (i in faceList.indices) {
            val face = faceList[i]

            // Draw a rectangle around each face on the framed image
            Imgproc.rectangle(
                framedImage,
                Point(face.x.toDouble(), face.y.toDouble()),
                Point((face.x + face.width).toDouble(), (face.y + face.height).toDouble()),
                neonGreen,  // Neon green color
                3  // Thickness
            )

            // Extract face region from original image
            val faceRegion = Mat(originalImage, face)

            // Resize to 200x200
            val face200x200 = Mat()
            Imgproc.resize(faceRegion, face200x200, Size(200.0, 200.0))

            // Resize to 48x48
            val face48x48 = Mat()
            Imgproc.resize(faceRegion, face48x48, Size(48.0, 48.0))

            // Add to list
            extractedFaces.add(
                ExtractedFace(
                    face200x200,
                    face48x48,
                    face
                )
            )
        }

        return ProcessedResult(
            originalImage,
            grayImage,
            framedImage,
            extractedFaces
        )
    }

    /**
     * Detect faces with multiple verification steps to reduce false positives.
     * Still not perfect and still need adjustement later
     */
    private fun detectFacesWithVerification(grayImage: Mat): MatOfRect {
        // First detection with Haar cascade
        val initialFaces = MatOfRect()
        faceDetector.detectMultiScale(
            grayImage,
            initialFaces,
            1.2,       // Scale factor
            6,         // Min neighbors
            0,         // Flags
            Size(40.0, 40.0),  // Min face size
            Size(grayImage.width().toDouble() * 0.8, grayImage.height().toDouble() * 0.8)
        )

        // If no faces found, try with more lenient parameters, but not too lenient
        if (initialFaces.empty()) {
            faceDetector.detectMultiScale(
                grayImage,
                initialFaces,
                1.1,
                4,
                0,
                Size(30.0, 30.0),
                Size(grayImage.width().toDouble() * 0.9, grayImage.height().toDouble() * 0.9)
            )
        }

        // Initial faces detected
        val faceList = initialFaces.toArray()

        // If we have too many detected faces (likely false positives), run verification
        if (faceList.size > 3) {
            // Try LBP cascade to verify faces
            val verifiedFaces = MatOfRect()
            lbpFaceDetector.detectMultiScale(
                grayImage,
                verifiedFaces,
                1.1,
                5,
                0,
                Size(40.0, 40.0),
                Size(grayImage.width().toDouble() * 0.8, grayImage.height().toDouble() * 0.8)
            )

            val verifiedFaceList = verifiedFaces.toArray()

            // Create a final list of faces
            val confirmedFaces = mutableListOf<Rect>()

            for (face1 in faceList) {
                var isConfirmed = false

                // The face is confirmed if there's significant overlap with a face detected by LBP
                for (face2 in verifiedFaceList) {
                    val intersection = face1.intersection(face2)
                    val unionArea = (face1.width * face1.height) + (face2.width * face2.height) -
                            (intersection.width * intersection.height)

                    if (intersection.width > 0 && intersection.height > 0) {
                        val overlapRatio = (intersection.width * intersection.height).toDouble() / unionArea

                        if (overlapRatio > 0.3) { // minimum 30% overlap
                            isConfirmed = true
                            break
                        }
                    }
                }

                // Add to the confirmed list if verified or if there were no LBP faces detected
                if (isConfirmed || verifiedFaceList.isEmpty()) {
                    confirmedFaces.add(face1)
                }
            }

            // Apply face proportion check to remove unlikely face shapes
            val filteredFaces = confirmedFaces.filter { face ->
                val aspectRatio = face.width.toDouble() / face.height
                // ratios between 0.75 and 1.3 seems to be face proportions for humans
                // lets test with this research and code
                aspectRatio in 0.75..1.3
            }

            // Return the filtered faces
            val result = MatOfRect()
            if (filteredFaces.isNotEmpty()) {
                result.fromList(filteredFaces)
            } else if (faceList.size <= 3) {
                // If we had 3 or fewer faces originally, just use those
                result.fromList(faceList.toList())
            } else {
                // If we had many faces but none passed verification, take at most 3 largest faces
                val sortedBySize = faceList.sortedByDescending { it.width * it.height }
                result.fromList(sortedBySize.take(3))
            }

            return result
        }

        // If we detected a reasonable number of faces initially, just return those
        return initialFaces
    }

    /**
     * Get bitmap from URI with proper orientation
     */
    private fun getBitmapFromUri(uri: Uri): Bitmap {
        val inputStream: InputStream = context.contentResolver.openInputStream(uri)
            ?: throw IOException("Couldn't open input stream for URI: $uri")

        // Decode image dimensions first
        val options = BitmapFactory.Options().apply {
            inJustDecodeBounds = true
        }
        BitmapFactory.decodeStream(inputStream, null, options)
        inputStream.close()

        // Reopen stream for actual decoding
        val inputStream2 = context.contentResolver.openInputStream(uri)
            ?: throw IOException("Couldn't open input stream for URI: $uri")

        // Decode the actual bitmap
        val options2 = BitmapFactory.Options().apply {
            inJustDecodeBounds = false
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val bitmap = BitmapFactory.decodeStream(inputStream2, null, options2)
        inputStream2.close()

        if (bitmap == null) {
            throw IOException("Couldn't decode bitmap from URI: $uri")
        }

        // Get the orientation from Exif data
        val orientation = getImageOrientation(uri)

        // Rotate the bitmap if needed
        return if (orientation != 0) {
            val matrix = Matrix()
            matrix.postRotate(orientation.toFloat())
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } else {
            bitmap
        }
    }

    /**
     * Get the orientation of an image from its EXIF data
     */
    private fun getImageOrientation(uri: Uri): Int {
        try {
            val inputStream = context.contentResolver.openInputStream(uri)
            if (inputStream != null) {
                val exif = ExifInterface(inputStream)
                val orientation = exif.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
                )
                inputStream.close()

                return when (orientation) {
                    ExifInterface.ORIENTATION_ROTATE_90 -> 90
                    ExifInterface.ORIENTATION_ROTATE_180 -> 180
                    ExifInterface.ORIENTATION_ROTATE_270 -> 270
                    else -> 0
                }
            }
        } catch (e: IOException) {
            e.printStackTrace()
        }
        return 0
    }

    /**
     * Extension function to get the intersection of two rectangles
     */
    private fun Rect.intersection(other: Rect): Rect {
        val left = maxOf(this.x, other.x)
        val top = maxOf(this.y, other.y)
        val right = minOf(this.x + this.width, other.x + other.width)
        val bottom = minOf(this.y + this.height, other.y + other.height)

        val width = maxOf(0, right - left)
        val height = maxOf(0, bottom - top)

        return Rect(left, top, width, height)
    }

    /**
     * Convert the processed results to bitmaps for use in Android
     */
    fun getResultBitmaps(result: ProcessedResult): BitmapResults {
        // Convert framed image to bitmap
        val framedBitmap = createBitmap(result.framedImage.cols(), result.framedImage.rows())
        Utils.matToBitmap(result.framedImage, framedBitmap)

        // Convert grayscale image to bitmap
        val grayBitmap = createBitmap(result.grayImage.cols(), result.grayImage.rows())
        Utils.matToBitmap(result.grayImage, grayBitmap)

        // Convert all extracted faces to bitmaps
        val facesBitmaps = mutableListOf<ExtractedFaceBitmaps>()

        for (face in result.extractedFaces) {
            val bitmap200x200 = createBitmap(200, 200)
            Utils.matToBitmap(face.face200x200, bitmap200x200)

            val bitmap48x48 = createBitmap(48, 48)
            Utils.matToBitmap(face.face48x48, bitmap48x48)

            facesBitmaps.add(
                ExtractedFaceBitmaps(
                    bitmap200x200,
                    bitmap48x48,
                    face.originalRect
                )
            )
        }

        return BitmapResults(
            framedBitmap,
            grayBitmap,
            facesBitmaps
        )
    }

    /**
     * Save the bitmap results to the app's storage
     */
    fun saveResults(results: BitmapResults, outputDir: String): List<Uri> {
        val dir = File(context.getExternalFilesDir(null), outputDir)
        if (!dir.exists()) {
            dir.mkdirs()
        }

        val savedUris = mutableListOf<Uri>()

        try {
            // Save framed image
            val framedFile = File(dir, "framed_image.jpg")
            FileOutputStream(framedFile).use { out ->
                results.framedBitmap.compress(Bitmap.CompressFormat.JPEG, 100, out)
            }
            savedUris.add(Uri.fromFile(framedFile))

            // Save grayscale image
            val grayFile = File(dir, "grayscale_image.jpg")
            FileOutputStream(grayFile).use { out ->
                results.grayBitmap.compress(Bitmap.CompressFormat.JPEG, 100, out)
            }
            savedUris.add(Uri.fromFile(grayFile))

            // Save each extracted face
            for (i in results.extractedFaces.indices) {
                val face = results.extractedFaces[i]

                // Save 200x200 face
                val face200File = File(dir, "face_${i}_200x200.jpg")
                FileOutputStream(face200File).use { out ->
                    face.face200x200.compress(Bitmap.CompressFormat.JPEG, 100, out)
                }
                savedUris.add(Uri.fromFile(face200File))

                // Save 48x48 face
                val face48File = File(dir, "face_${i}_48x48.jpg")
                FileOutputStream(face48File).use { out ->
                    face.face48x48.compress(Bitmap.CompressFormat.JPEG, 100, out)
                }
                savedUris.add(Uri.fromFile(face48File))
            }
        } catch (e: IOException) {
            e.printStackTrace()
        }

        return savedUris
    }

    /**
     * Class to hold the extracted faces as Mat objects
     */
    data class ExtractedFace(
        val face200x200: Mat,
        val face48x48: Mat,
        val originalRect: Rect
    )

    /**
     * Class to hold the extracted faces as Bitmap objects
     */
    data class ExtractedFaceBitmaps(
        val face200x200: Bitmap,
        val face48x48: Bitmap,
        val originalRect: Rect
    )

    /**
     * Class to hold the processing results as Mat objects
     */
    data class ProcessedResult(
        val originalImage: Mat,
        val grayImage: Mat,
        val framedImage: Mat,
        val extractedFaces: List<ExtractedFace>
    )

    /**
     * Class to hold the processing results as Bitmap objects
     */
    data class BitmapResults(
        val framedBitmap: Bitmap,
        val grayBitmap: Bitmap,
        val extractedFaces: List<ExtractedFaceBitmaps>
    )
}