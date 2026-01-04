import 'package:flutter/material.dart';
import 'package:image_classification/image_classifier_screen.dart';
import 'package:image_classification/tflite_helper.dart';
import 'package:flutter/services.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await verifyAssets();
  await TFLiteHelper.init();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Image Classifier',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: ImageClassifierScreen(), // Your screen here
    );
  }
}

Future<void> verifyAssets() async {
  try {
    // Check TFLite model
    final modelData = await rootBundle.load('assets/mobilenet_224.tflite');
    if (modelData.lengthInBytes == 0) {
      print("Error: mobilenet_224.tflite is empty!");
    } else {
      print("mobilenet_224.tflite exists and has size: ${modelData.lengthInBytes} bytes");
    }

    // Check labels
    final labelsData = await rootBundle.loadString('assets/labels.txt');
    if (labelsData.trim().isEmpty) {
      print("Error: labels.txt is empty!");
    } else {
      print("labels.txt exists and has ${labelsData.split('\n').length} lines");
    }
  } catch (e) {
    print("Failed to load asset: $e");
  }
}
