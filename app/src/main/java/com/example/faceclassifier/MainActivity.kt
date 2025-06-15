package com.example.faceclassifier

import FaceDetectionProcessor
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import org.opencv.android.OpenCVLoader
import java.io.File

class MainActivity : AppCompatActivity() {

    lateinit var imgV:ImageView
    lateinit var btnSelect:Button
    lateinit var btnTakePicture:Button

    lateinit var imageUri:Uri

    lateinit var textView: TextView
    lateinit var mLineTextView: TextView

    private lateinit var faceProcessor: FaceDetectionProcessor

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        OpenCVLoader.initLocal()


        // Initialize the face processor
        faceProcessor = FaceDetectionProcessor(this)

        imgV = findViewById(R.id.imageView)
        btnSelect = findViewById(R.id.button)
        btnTakePicture = findViewById(R.id.button3)
        mLineTextView = findViewById(R.id.editTextTextMultiLine)

        val imagePicker = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) {uri->
            if (uri != null) {
                val processedResult = faceProcessor.processImage(uri)


                // Convert to bitmaps
                val bitmapResults = faceProcessor.getResultBitmaps(processedResult)


                // Display results
                val framedBitmap = bitmapResults.framedBitmap
                imgV.setImageBitmap(framedBitmap)

                // Now you can use the extracted faces as needed
                val extractedFaces = bitmapResults.extractedFaces.get(0).face200x200


                val model1Labels = arrayOf("0-2", "3-9", "10-20", "21-27", "28-45", "46-65", "66+")
                val model2Labels = arrayOf("male", "female")
                val model3Labels = arrayOf("happy", "sad")

                val processor = FaceModelProcessor(
                    context = this,
                    model1Name = "model_age.tflite",
                    model2Name = "model_gender.tflite",
                    model3Name = "model_emotion.tflite",
                    outputSize1 = 7,  // Number of age classes
                    outputSize2 = 2,  // Number of gender classes
                    outputSize3 = 2,  // Number of emotion classes
                    model1ClassLabels = model1Labels,
                    model2ClassLabels = model2Labels,
                    model3ClassLabels = model3Labels
                )

                // Analyze a face and get the top class predictions
                val result = processor.analyzeFace(extractedFaces)

                // Access the top class predictions
                if (result != null) {
                    val topAgeGroup = result.topClass1
                    val topGender = result.topClass2
                    val topEmotion = result.topClass3

                    // Use the top class information
                    Log.d("FaceAnalysis", "Emotion: ${topEmotion.className} (${topEmotion.confidence})")
                    Log.d("FaceAnalysis", "Gender: ${topGender.className} (${topGender.confidence})")
                    Log.d("FaceAnalysis", "Age: ${topAgeGroup.className} (${topAgeGroup.confidence})")

                    mLineTextView.text = buildString {
                        append("Emotion: ")
                        append(topEmotion.className)
                        append(" (")
                        append(topEmotion.confidence)
                        append(") \nGender: ")
                        append(topGender.className)
                        append(" (")
                        append(topGender.confidence)
                        append(") \nAge: ")
                        append(topAgeGroup.className)
                        append(" (")
                        append(topAgeGroup.confidence)
                        append(")\" ")
                    }
                }
            }
        }

        btnSelect.setOnClickListener {
            imagePicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }

        imageUri = createImageUri()
        val takePicture = registerForActivityResult(ActivityResultContracts.TakePicture()) {
            if (imageUri != null) {
                val processedResult = faceProcessor.processImage(imageUri )


                // Convert to bitmaps
                val bitmapResults = faceProcessor.getResultBitmaps(processedResult)


                // Display results
                val framedBitmap = bitmapResults.framedBitmap
                imgV.setImageBitmap(framedBitmap)

                // extracted faces as needed
                val extractedFaces = bitmapResults.extractedFaces.get(0).face200x200


                val model1Labels = arrayOf("0-2", "3-9", "10-20", "21-27", "28-45", "46-65", "66+")
                val model2Labels = arrayOf("male", "female")
                val model3Labels = arrayOf("happy", "sad")

                val processor = FaceModelProcessor(
                    context = this,
                    model1Name = "model_age.tflite",
                    model2Name = "model_gender.tflite",
                    model3Name = "model_emotion.tflite",
                    outputSize1 = 7,  // Number of emotion classes
                    outputSize2 = 2,  // Number of gender classes
                    outputSize3 = 2,  // Number of age classes
                    model1ClassLabels = model1Labels,
                    model2ClassLabels = model2Labels,
                    model3ClassLabels = model3Labels
                )

                // Analyze a face and get the top class predictions
                val result = processor.analyzeFace(extractedFaces)

                // Access the top class predictions
                if (result != null) {
                    val topEmotion = result.topClass1
                    val topGender = result.topClass2
                    val topAgeGroup = result.topClass3

                    // Use the top class information
                    Log.d("FaceAnalysis", "Emotion: ${topEmotion.className} (${topEmotion.confidence})")
                    Log.d("FaceAnalysis", "Gender: ${topGender.className} (${topGender.confidence})")
                    Log.d("FaceAnalysis", "Age: ${topAgeGroup.className} (${topAgeGroup.confidence})")

                    mLineTextView.text = buildString {
                        append("Emotion: ")
                        append(topEmotion.className)
                        append(" (")
                        append(topEmotion.confidence)
                        append(") \nGender: ")
                        append(topGender.className)
                        append(" (")
                        append(topGender.confidence)
                        append(") \nAge: ")
                        append(topAgeGroup.className)
                        append(" (")
                        append(topAgeGroup.confidence)
                        append(")\" ")
                    }

                }


            }
        }
        btnTakePicture.setOnClickListener {
            takePicture.launch(imageUri)
            true
        }
    }

    fun createImageUri(): Uri {
        val image = File(applicationContext.filesDir, "camera_image.pmg")
        return FileProvider.getUriForFile(applicationContext, "com.example.faceclassifier.fileprovider", image)
    }
}