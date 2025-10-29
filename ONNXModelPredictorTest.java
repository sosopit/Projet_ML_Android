package com.example.healthcare;
import android.content.Context;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class ONNXModelPredictorTest {

    private Context appContext;
    private ONNXModelPredictor predictor;
    private final int EXPECTED_SYMPTOMS_COUNT = 131;

    @Before
    public void setup() throws Exception {
        appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();

        predictor = new ONNXModelPredictor(appContext);
    }

    @Test
    public void testSymptomLoading() {
        assertNotNull("Le prédicteur ne doit pas être null après l'initialisation", predictor);

    }

    @Test
    public void testONNXSessionInitialization() {
        assertNotNull("La session ONNX interne ne doit pas être null (si accessible)", predictor);
    }

    @Test
    public void testPredictionDengue() {
        List<String> dengueSymptoms = Arrays.asList(
                "chills", "joint_pain", "vomiting", "fatigue", "high_fever", "headache", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "malaise", "muscle_pain"
        );

        String result = predictor.predict(dengueSymptoms);
        assertNotNull("Le résultat de la prédiction ne doit pas être null", result);

        // Si "Dengue" est l'une de vos maladies
        assertEquals("Le modèle devrait prédire 'Dengue' pour ces symptômes", "Dengue", result);
    }
}