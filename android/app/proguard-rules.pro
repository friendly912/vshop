# LiteRT / TFLite (JNI-bound classes)
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.**

# MediaPipe Tasks (JNI + protobuf-lite)
-keep class com.google.mediapipe.** { *; }
-keep class com.google.protobuf.** { *; }
-dontwarn com.google.mediapipe.**
-dontwarn com.google.auto.value.**
