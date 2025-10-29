package com.example.healthcare;

import android.content.Context;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Arrays;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public class ONNXModelPredictor {

    private OrtEnvironment ortEnv;
    private OrtSession ortSession;
    private List<String> allSymptoms;
    private List<String> labels;
    private static final String MODEL_FILE = "healthcare_model_quantized.onnx";

    // ------------------ CONSTRUCTEUR ------------------
    public ONNXModelPredictor(Context context) throws Exception {
        // Charger et nettoyer symptômes et labels
        allSymptoms = cleanSymptoms(loadSymptoms(context));
        labels = cleanLabels(loadLabels(context));

        // Initialiser ONNX Runtime
        ortEnv = OrtEnvironment.getEnvironment();
        try (InputStream inputStream = context.getAssets().open(MODEL_FILE)) {
            byte[] modelBytes = new byte[inputStream.available()];
            inputStream.read(modelBytes);
            ortSession = ortEnv.createSession(modelBytes, new OrtSession.SessionOptions());
        }

        System.out.println("=== Infos sur le modèle ONNX ===");
        System.out.println(ortSession.getInputInfo());
        System.out.println("================================");
    }

    // ------------------ CHARGEMENT DES FICHIERS ------------------
    private List<String> loadLabels(Context context) {
        List<String> labels = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(context.getAssets().open("label.txt")))) {
            String line;
            while ((line = reader.readLine()) != null) {
                labels.add(line.trim());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return labels;
    }

    private List<String> loadSymptoms(Context context) {
        List<String> symptoms = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(context.getAssets().open("symptomes.txt")))) {
            String line;
            while ((line = reader.readLine()) != null) {
                symptoms.add(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return symptoms;
    }

    // ------------------ NETTOYAGE DES SYMPTÔMES ET LABELS ------------------
    private List<String> cleanSymptoms(List<String> symptoms) {
        List<String> cleaned = new ArrayList<>();
        for (String s : symptoms) {
            if (s != null && !s.isEmpty()) {
                cleaned.add(s.trim().replaceAll(" +", "_").toLowerCase());
            }
        }
        return cleaned;
    }

    private List<String> cleanLabels(List<String> labels) {
        List<String> cleaned = new ArrayList<>();
        for (String s : labels) {
            if (s != null && !s.isEmpty()) {
                cleaned.add(s.trim());
            }
        }
        return cleaned;
    }

    // ------------------ ENCODAGE DES SYMPTÔMES ------------------
    private float[] encodeSymptoms(List<String> selectedSymptoms) {
        float[] inputVector = new float[allSymptoms.size()];
        for (int i = 0; i < allSymptoms.size(); i++) {
            inputVector[i] = selectedSymptoms.contains(allSymptoms.get(i)) ? 1.0f : 0.0f;
        }
        return inputVector;
    }

    // ------------------ PRÉDICTION ------------------
    public String predict(List<String> selectedSymptoms) {
        if (ortSession == null) return "Erreur: Session ONNX non initialisée.";

        // Nettoyer et filtrer les symptômes avant encodage
        List<String> cleanedSymptoms = cleanSymptoms(selectedSymptoms);
        List<String> validSymptoms = new ArrayList<>();
        for (String s : cleanedSymptoms) {
            if (allSymptoms.contains(s)) validSymptoms.add(s);
            else System.out.println("Symptôme ignoré : " + s);
        }

        float[] inputVector = encodeSymptoms(validSymptoms);
        long[] shape = new long[]{1, inputVector.length};

        try (OnnxTensor inputTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(inputVector), shape)) {
            String inputName = ortSession.getInputInfo().keySet().iterator().next();
            Map<String, OnnxTensor> inputs = Collections.singletonMap(inputName, inputTensor);

            try (OrtSession.Result result = ortSession.run(inputs)) {
                long[] outputData = (long[]) result.get(0).getValue();
                int predictedIndex = (int) outputData[0];

                // DEBUG
                System.out.println("Valid symptoms: " + validSymptoms);
                System.out.println("Encoded vector: " + Arrays.toString(inputVector));
                System.out.println("Predicted index: " + predictedIndex);
                if (predictedIndex >= 0 && predictedIndex < labels.size()) {
                    System.out.println("Predicted label: " + labels.get(predictedIndex));
                    return labels.get(predictedIndex);
                } else {
                    return "Erreur: index label invalide (" + predictedIndex + ")";
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            return "Erreur lors de l'inférence: " + e.getMessage();
        }
    }
}
